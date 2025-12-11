import argparse
import copy
import time
from dataclasses import asdict
from typing import List

import jax
from config.config import Config
from config.hessian_approximation_config import (
    FisherInformationConfig,
    GaussNewtonHessianConfig,
    HessianApproximationConfig,
    HessianName,
    KFACBuildConfig,
    KFACConfig,
    KFACRunConfig,
)
from hessian_approximations.fim.fisher_information import FisherInformation
from hessian_approximations.gauss_newton.gauss_newton import GaussNewton
from hessian_approximations.hessian.hessian import Hessian
from hessian_approximations.kfac.kfac import KFAC
from jaxtyping import Array, Float
from models.train import train_or_load

from metrics.vector_metrics import VectorMetric
from scripts.run_gpu_tests import (
    _linear_setup,
    _mlp_setup,
    aggregate_results_across_seeds,
)
from src.utils.utils import (
    get_peak_bytes_in_use,
    plot_metric_results_with_seeds,
    sample_gradient_from_output_distribution_batched,
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
        default=3,
    )

    return parser.parse_args()


def compare_approximation_with_true_hessian(
    full_config: Config,
    hessian_approxs_configs: List[HessianApproximationConfig],
    vectors_1: Float[Array, "num_vectors num_params"],
    vectors_2: Float[Array, "num_vectors num_params"],
    ground_truth_hessian_approach: HessianName = HessianName.HESSIAN,
):
    results_dict = {}

    config = copy.deepcopy(full_config)
    config.hessian_approximation = HessianApproximationConfig(
        ground_truth_hessian_approach
    )

    damping = KFAC.setup_with_run_and_build_config(
        full_config=full_config,
        build_config=KFACBuildConfig(use_pseudo_targets=True),
        run_config=KFACRunConfig(use_eigenvalue_correction=False),
    ).damping()

    results_dict["damping"] = float(damping)

    if ground_truth_hessian_approach == HessianName.HESSIAN:
        true_hessian_model = Hessian(full_config=config)
    else:
        true_hessian_model = GaussNewton(full_config=config)

    start_time_hessian = time.time()
    true_hessian_ihvps = true_hessian_model.compute_ihvp(
        vectors=vectors_1, damping=damping
    )
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

        start_time_approx = time.time()

        vector_metrics = VectorMetric.all_metrics()
        for vector_metric in vector_metrics:
            ihvp_approx = approx_method.compute_ihvp(vectors=vectors_1, damping=damping)
            metric_value = vector_metric.compute(
                v1=true_hessian_ihvps,
                v2=ihvp_approx,
                x=vectors_2,
                reduction="mean",
            )
            results_dict[approx_name][vector_metric.value] = float(metric_value)

        end_time_approx = time.time()
        results_dict[approx_name]["time"] = end_time_approx - start_time_approx
        results_dict[approx_name]["num_params"] = (
            approx_method.model_context.model.get_num_params(
                approx_method.model_context.params
            )
        )
    return results_dict


def run_gpu_tests_vectors():
    args = parse_args()
    min_params = args.n_params_min
    max_params = args.n_params_max
    num_reps = args.reps
    num_seeds = args.seeds

    # Define seeds internally depending on number of repetitions
    seeds = [554 + i * 2 for i in range(num_seeds)]

    linear_setup = _linear_setup(
        min_params=min_params,
        max_params=max_params,
        num_reps=num_reps,
        layer_type="single_layer",
    )
    linear_setup_multi_layer = _linear_setup(
        min_params=min_params,
        max_params=max_params,
        num_reps=num_reps,
        layer_type="multi_layer",
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
                model_data = train_or_load(full_config)
                vectors_1 = sample_gradient_from_output_distribution_batched(
                    model_data=model_data,
                    n_vectors=500,
                    rng_key=jax.random.PRNGKey(7833),
                )

                vectors_2 = sample_gradient_from_output_distribution_batched(
                    model_data=model_data,
                    n_vectors=500,
                    rng_key=jax.random.PRNGKey(10112),
                )
                kfac_config = KFACConfig(
                    build_config=KFACBuildConfig(use_pseudo_targets=True),
                    run_config=KFACRunConfig(use_eigenvalue_correction=False),
                )
                ekfac_config = KFACConfig(
                    build_config=KFACBuildConfig(use_pseudo_targets=True),
                    run_config=KFACRunConfig(use_eigenvalue_correction=True),
                )

                if collection in [linear_setup, linear_setup_multi_layer]:
                    ground_truth_hessian_approach = HessianName.HESSIAN
                    comparison_configs = [
                        kfac_config,
                        ekfac_config,
                        FisherInformationConfig(fisher_type="true"),
                        GaussNewtonHessianConfig(),
                    ]
                else:
                    ground_truth_hessian_approach = HessianName.GAUSS_NEWTON
                    comparison_configs = [
                        kfac_config,
                        ekfac_config,
                        FisherInformationConfig(fisher_type="true"),
                    ]

                print(f"Comparing Hessians for dataset: {dataset_name}")

                start_time = time.time()
                results_dict[dataset_name] = compare_approximation_with_true_hessian(
                    full_config=full_config,
                    hessian_approxs_configs=comparison_configs,
                    vectors_1=vectors_1,
                    vectors_2=vectors_2,
                    ground_truth_hessian_approach=ground_truth_hessian_approach,
                )
                end_time = time.time()
                print(
                    f"Time taken for dataset {dataset_name}: {end_time - start_time} seconds"
                )
                results_dict[dataset_name]["time_taken_overall"] = end_time - start_time
                print(f"Completed dataset: {dataset_name}")
                results_dict[dataset_name]["peak_memory"] = get_peak_bytes_in_use()
                results_dict[dataset_name]["training_config"] = asdict(
                    training_config[dataset_name]
                )
                results_dict[dataset_name]["model_config"] = asdict(
                    model_config[dataset_name]
                )
                results_dict[dataset_name]["dataset_config"] = asdict(
                    dataset_config[dataset_name]
                )

            all_seeds_results[seed] = results_dict

        # Aggregate results across seeds
        aggregated_results = aggregate_results_across_seeds(all_seeds_results)

        # Save results
        folder_parent = f"data/artifacts/results/ihvp/{timestamp}"
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
                VectorMetric.RELATIVE_ERROR.value,
                VectorMetric.COSINE_SIMILARITY.value,
                VectorMetric.INNER_PRODUCT_DIFF.value,
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
        title="Comparison of IHVPs",
        folder=folder_parent,
        overview_metric=VectorMetric.RELATIVE_ERROR.value,
    )

    print("Saved data to:")
    print(f"{folder_parent}")


if __name__ == "__main__":
    run_gpu_tests_vectors()
