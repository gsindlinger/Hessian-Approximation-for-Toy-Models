import argparse
import copy
import time
from typing import List, Literal

import numpy as np
from jax import numpy as jnp

from config.config import Config
from config.dataset_config import RandomClassificationConfig
from config.hessian_approximation_config import (
    FisherInformationConfig,
    GaussNewtonHessianConfig,
    HessianApproximationConfig,
    HessianName,
    KFACBuildConfig,
    KFACConfig,
    KFACRunConfig,
)
from config.model_config import LinearModelConfig, MLPModelConfig
from config.training_config import TrainingConfig
from hessian_approximations.fim.fisher_information import FisherInformation
from hessian_approximations.gauss_newton.gauss_newton import GaussNewton
from hessian_approximations.hessian.hessian import Hessian
from hessian_approximations.kfac.kfac import KFAC
from metrics.full_matrix_metrics import FullMatrixMetric
from utils.utils import (
    get_peak_bytes_in_use,
    plot_metric_results_with_seeds,
    print_device_memory_stats,
    save_results,
    write_global_markdown_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU Memory Usage Tests for Hessian Approximations"
    )
    parser.add_argument(
        "--n_params_min",
        type=int,
        help="Minimum number of parameters for the model",
        default=100,
    )
    parser.add_argument(
        "--n_params_max",
        type=int,
        help="Maximum number of parameters for the model",
        default=10000,
    )
    parser.add_argument(
        "--reps",
        type=int,
        help="Number of repetitions between min and max parameters",
        default=1,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        help="Number of different random seeds to average results over",
        default=5,
    )

    return parser.parse_args()


def compare_approximation_with_true_hessian(
    full_config: Config,
    hessian_approxs_configs: List[HessianApproximationConfig],
    ground_truth_hessian_approach: HessianName = HessianName.HESSIAN,
):
    results_dict = {}

    config = copy.deepcopy(full_config)
    config.hessian_approximation = HessianApproximationConfig(
        ground_truth_hessian_approach
    )

    damping = KFAC.setup_with_run_and_build_config(
        full_config=full_config,
        build_config=KFACBuildConfig(),
        run_config=KFACRunConfig(use_eigenvalue_correction=False),
    ).damping()

    if ground_truth_hessian_approach == HessianName.HESSIAN:
        true_hessian_model = Hessian(full_config=config)
    else:
        true_hessian_model = GaussNewton(full_config=config)

    start_time_hessian = time.time()
    true_hessian = true_hessian_model.compute_hessian(damping=damping)
    end_time_hessian = time.time()
    results_dict["true_hessian_time"] = end_time_hessian - start_time_hessian

    for approx_config in hessian_approxs_configs:
        if isinstance(approx_config, KFACConfig):
            approx_name = (
                "ekfac"
                if approx_config.run_config.use_eigenvalue_correction
                else "kfac"
            )

            approx_method = KFAC.setup_with_run_and_build_config(
                full_config=full_config,
                build_config=approx_config.build_config,
                run_config=approx_config.run_config,
            )
        elif isinstance(approx_config, FisherInformationConfig):
            approx_name = "fim"
            approx_method = FisherInformation(full_config=full_config)
        elif isinstance(approx_config, GaussNewtonHessianConfig):
            approx_name = "gauss_newton"
            approx_method = GaussNewton(full_config=full_config)
        else:
            raise ValueError(
                f"Unsupported Hessian approximation config type: {type(approx_config)}"
            )

        results_dict[approx_name] = {}

        print_device_memory_stats(
            f"Before computing and comparing {approx_name} Hessian"
        )
        start_time_approx = time.time()
        relative_frobenius = approx_method.compare_hessians(
            damping=damping,
            comparison_matrix=true_hessian,
            metric=FullMatrixMetric.RELATIVE_FROBENIUS,
        )
        print_device_memory_stats(
            f"After comparing {approx_name} Hessian for Relative Frobenius"
        )
        end_time_approx = time.time()
        results_dict[approx_name]["relative_frobenius"] = float(relative_frobenius)
        cosine_similarity = approx_method.compare_hessians(
            damping=damping,
            comparison_matrix=true_hessian,
            metric=FullMatrixMetric.COSINE_SIMILARITY,
        )
        print_device_memory_stats(
            f"After comparing {approx_name} Hessian for Cosine Similarity"
        )
        results_dict[approx_name]["cosine_similarity"] = float(cosine_similarity)
        trace_distance = approx_method.compare_hessians(
            damping=damping,
            comparison_matrix=true_hessian,
            metric=FullMatrixMetric.TRACE_DISTANCE,
        )
        results_dict[approx_name]["trace_distance"] = float(trace_distance)
        print_device_memory_stats(
            f"After comparing {approx_name} Hessian for Trace Distance"
        )

        results_dict[approx_name]["time"] = end_time_approx - start_time_approx
        results_dict[approx_name]["num_params"] = (
            approx_method.model_context.model.get_num_params(
                approx_method.model_context.params
            )
        )
        print_device_memory_stats(
            f"After computing and comparing {approx_name} Hessian"
        )

    return results_dict


def _linear_setup(
    min_params: int,
    max_params: int,
    num_reps: int,
    n_samples: int = 3000,
    layer_type: Literal["single_layer", "multi_layer"] = "single_layer",
):
    if layer_type == "multi_layer":
        dataset_nums = _mlp_dataset_nums(min_params, max_params, num_reps, n_samples)
    else:
        target_param_sizes = jnp.linspace(
            min_params, max_params, num=num_reps, dtype=int
        ).tolist()
        n_classes_list = [10] * num_reps

        dataset_nums = []

        for i, (target_params, n_classes) in enumerate(
            zip(target_param_sizes, n_classes_list)
        ):
            n_features = max(10, int(target_params / n_classes))
            n_informative = min(max(2, int(0.6 * n_features)), 100)

            dataset_nums.append(
                {
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "n_classes": n_classes,
                    "n_informative": n_informative,
                    "total_params": n_features * n_classes,
                    "hidden_dims": [],
                }
            )

    dataset_configs = {}
    training_configs = {}
    model_configs = {}

    for dataset_num in dataset_nums:
        n_samples = dataset_num["n_samples"]
        n_features = dataset_num["n_features"]
        n_classes = dataset_num["n_classes"]
        n_informative = dataset_num["n_informative"]
        hidden_dims = dataset_num["hidden_dims"]

        dataset_configs[
            f"random_classification_linear_only_{n_samples}_{n_features}_{hidden_dims}_{n_classes}"
        ] = RandomClassificationConfig(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            train_test_split=1,
        )
        training_configs[
            f"random_classification_linear_only_{n_samples}_{n_features}_{hidden_dims}_{n_classes}"
        ] = TrainingConfig(
            epochs=1000,
            lr=0.001,
            optimizer="sgd",
            batch_size=100,
            loss="cross_entropy",
        )
        model_configs[
            f"random_classification_linear_only_{n_samples}_{n_features}_{hidden_dims}_{n_classes}"
        ] = LinearModelConfig(loss="cross_entropy", hidden_dim=hidden_dims)

    return dataset_configs, training_configs, model_configs


def _mlp_dataset_nums(
    min_params: int, max_params: int, num_reps: int, n_samples: int = 3000
):
    target_param_sizes = jnp.linspace(
        min_params, max_params, num=num_reps, dtype=int
    ).tolist()

    n_classes = 10
    dataset_nums = []

    def choose_n_features(target):
        base = 8
        scale = int(jnp.log2(max(target, 8)))
        return int(min(500, base + 6 * scale))

    def candidate_hidden_dims(target):
        base = max(4, int(jnp.sqrt(target) / 6))
        return sorted({base, max(4, base // 2), min(512, base * 2), min(512, base * 4)})

    def compute_params(n_features, hidden_dims):
        total = 0
        d = n_features
        for h in hidden_dims:
            total += d * h
            d = h
        total += d * n_classes
        return total

    for target in target_param_sizes:
        n_features = choose_n_features(target)
        n_informative = min(int(0.6 * n_features), n_features)

        hidden_candidates = candidate_hidden_dims(target)
        best = None

        for h1 in hidden_candidates:
            dims1 = [h1]
            p1 = compute_params(n_features, dims1)
            diff1 = abs(p1 - target)
            if best is None or diff1 < best[0]:
                best = (diff1, dims1, p1)

            for h2 in hidden_candidates:
                if h2 > h1:
                    continue
                dims2 = [h1, h2]
                p2 = compute_params(n_features, dims2)
                diff2 = abs(p2 - target)
                if diff2 < best[0]:
                    best = (diff2, dims2, p2)

        diff, hidden_dims, total_params = best  # type: ignore

        dataset_nums.append(
            {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
                "n_informative": n_informative,
                "hidden_dims": hidden_dims,
                "total_params": total_params,
                "target_params": target,
                "difference": diff,
            }
        )

    return dataset_nums


def _mlp_setup(min_params: int, max_params: int, num_reps: int, n_samples: int = 3000):
    dataset_nums = _mlp_dataset_nums(min_params, max_params, num_reps, n_samples)

    dataset_configs = {}
    training_configs = {}
    model_configs = {}

    for dataset_num in dataset_nums:
        n_samples = dataset_num["n_samples"]
        n_features = dataset_num["n_features"]
        n_classes = dataset_num["n_classes"]
        n_informative = dataset_num["n_informative"]
        hidden_dims_list = dataset_num["hidden_dims"]
        dataset_configs[
            f"random_classification_mlp_{n_samples}_{n_features}_{hidden_dims_list}_{n_classes}"
        ] = RandomClassificationConfig(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            train_test_split=1,
        )
        training_configs[
            f"random_classification_mlp_{n_samples}_{n_features}_{hidden_dims_list}_{n_classes}"
        ] = TrainingConfig(
            epochs=1000,
            lr=0.001,
            optimizer="sgd",
            batch_size=100,
            loss="cross_entropy",
        )
        model_configs[
            f"random_classification_mlp_{n_samples}_{n_features}_{hidden_dims_list}_{n_classes}"
        ] = MLPModelConfig(loss="cross_entropy", hidden_dim=hidden_dims_list)

    mlp_collection = (dataset_configs, training_configs, model_configs)
    return mlp_collection


def run_gpu_tests():
    args = parse_args()
    min_params = args.n_params_min
    max_params = args.n_params_max
    num_reps = args.reps
    num_seeds = args.seeds

    # Define seeds internally depending on number of repetitions
    seeds = [554 + i * 2 for i in range(num_seeds)]

    linear_setup = _linear_setup(
        min_params, max_params, num_reps, layer_type="single_layer"
    )
    linear_setup_multi_layer = _linear_setup(
        min_params, max_params, num_reps, layer_type="multi_layer"
    )
    mlp_setup = _mlp_setup(min_params, max_params, num_reps)

    # Compute for both single linear layer and MLP layer datasets
    all_collection_results = {}
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    for collection in [linear_setup, linear_setup_multi_layer, mlp_setup]:
        dataset_config, training_config, model_config = collection

        # Store results across all seeds
        all_seeds_results = {}

        for seed in seeds:
            print(f"\n{'=' * 60}")
            print(f"Running with seed: {seed}")
            print(f"{'=' * 60}\n")

            results_dict = {}

            for dataset_name, dataset_cfg in dataset_config.items():
                full_config = Config(
                    dataset=dataset_cfg,
                    model=model_config[dataset_name],
                    training=training_config[dataset_name],
                    seed=seed,
                )
                kfac_config = KFACConfig(
                    build_config=KFACBuildConfig(use_pseudo_targets=True),
                    run_config=KFACRunConfig(use_eigenvalue_correction=False),
                )
                ekfac_config = KFACConfig(
                    build_config=KFACBuildConfig(use_pseudo_targets=True),
                    run_config=KFACRunConfig(use_eigenvalue_correction=True),
                )

                if collection == mlp_setup:
                    ground_truth_hessian_approach = HessianName.GAUSS_NEWTON
                    comparison_configs = [
                        kfac_config,
                        ekfac_config,
                        FisherInformationConfig(fisher_type="true"),
                    ]
                else:
                    ground_truth_hessian_approach = HessianName.HESSIAN
                    comparison_configs = [
                        kfac_config,
                        ekfac_config,
                        FisherInformationConfig(fisher_type="true"),
                        GaussNewtonHessianConfig(),
                    ]

                print(f"Comparing Hessians for dataset: {dataset_name}")

                start_time = time.time()
                results_dict[dataset_name] = compare_approximation_with_true_hessian(
                    full_config=full_config,
                    hessian_approxs_configs=comparison_configs,
                    ground_truth_hessian_approach=ground_truth_hessian_approach,
                )
                end_time = time.time()
                print(
                    f"Time taken for dataset {dataset_name}: {end_time - start_time} seconds"
                )
                results_dict[dataset_name]["time_taken_overall"] = end_time - start_time
                print(f"Completed dataset: {dataset_name}")
                results_dict[dataset_name]["peak_memory"] = get_peak_bytes_in_use()

            all_seeds_results[seed] = results_dict

        # Aggregate results across seeds
        aggregated_results = aggregate_results_across_seeds(all_seeds_results)

        # Save results
        folder_parent = f"data/artifacts/results/hessian/{timestamp}"
        if collection == linear_setup:
            folder = f"{folder_parent}/single_linear"
        elif collection == linear_setup_multi_layer:
            folder = f"{folder_parent}/multi_linear"
        else:
            folder = f"{folder_parent}/mlp"
        # Save individual seed results
        save_results(
            all_seeds_results,
            f"{folder}/kfac_hessian_comparison_all_seeds_{timestamp}.json",
        )

        # Save aggregated results
        save_results(
            aggregated_results,
            f"{folder}/kfac_hessian_comparison_aggregated_{timestamp}.json",
        )

        # Plot with error bars
        plot_metric_results_with_seeds(
            results=aggregated_results,
            filename=f"{folder}/kfac_hessian_comparison_metrics_{timestamp}",
            metrics=[
                "relative_frobenius",
                "cosine_similarity",
                "trace_distance",
            ],
        )
        print(
            f"Saved results to {folder}/kfac_hessian_comparison_results_{timestamp}.json"
        )

        if collection == linear_setup:
            collection_name = "single_linear"
        elif collection == linear_setup_multi_layer:
            collection_name = "multi_linear"
        else:
            collection_name = "mlp"

        all_collection_results[collection_name] = {
            "folder": folder,
            "aggregated": aggregated_results,
            "all_seeds": all_seeds_results,
        }

    write_global_markdown_summary(
        all_collections_results=all_collection_results,
        timestamp=timestamp,
        title="Comparison of Full Hessians",
        folder=folder_parent,
        overview_metric=FullMatrixMetric.RELATIVE_FROBENIUS.value,
    )
    print("Saved data to:")
    print(f"{folder_parent}")


def aggregate_results_across_seeds(all_seeds_results):
    """
    Aggregate results across multiple seeds, computing mean and std for each metric.

    Structure:
    aggregated_results[dataset_name][approx_method][metric] = {
        'mean': float,
        'std': float,
        'values': list
    }
    """
    aggregated = {}

    # Get all dataset names from first seed
    first_seed = list(all_seeds_results.keys())[0]
    dataset_names = list(all_seeds_results[first_seed].keys())

    for dataset_name in dataset_names:
        aggregated[dataset_name] = {}

        # Get all approximation methods
        approx_methods = [
            k
            for k in all_seeds_results[first_seed][dataset_name].keys()
            if k in ["kfac", "ekfac", "fim", "gauss_newton"]
        ]

        for approx_method in approx_methods:
            aggregated[dataset_name][approx_method] = {}

            # Get all metrics for this approximation method
            metrics = list(
                all_seeds_results[first_seed][dataset_name][approx_method].keys()
            )

            for metric in metrics:
                values = []
                for seed in all_seeds_results.keys():
                    value = all_seeds_results[seed][dataset_name][approx_method][metric]
                    values.append(value)

                aggregated[dataset_name][approx_method][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }

    return aggregated


if __name__ == "__main__":
    run_gpu_tests()
