import argparse

import jax
from jax import numpy as jnp

from config.config import Config
from config.dataset_config import RandomClassificationConfig
from config.hessian_approximation_config import (
    KFACBuildConfig,
    KFACConfig,
    KFACRunConfig,
)
from config.model_config import LinearModelConfig
from config.training_config import TrainingConfig
from hessian_approximations.kfac.kfac import KFAC
from metrics.vector_metrics import VectorMetric
from models.train import train_or_load
from utils.utils import (
    print_device_memory_stats,
    sample_gradient_from_output_distribution_batched,
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
        default=20000,
    )
    parser.add_argument(
        "--reps",
        type=int,
        help="Number of repetitions between min and max parameters",
        default=1,
    )

    parser.add_argument(
        "--num_vectors",
        type=int,
        help="Number of random vectors for IHVP computation",
        default=10,
    )

    return parser.parse_args()


def check_memory():
    #### PARSE ARGUMENTS ####
    args = parse_args()
    reps = args.reps
    min_params = args.n_params_min
    max_params = args.n_params_max
    num_vectors = args.num_vectors

    #### DEVICE DIAGNOSTICS ####
    print("\n" + "=" * 60)
    print("JAX DEVICE CONFIGURATION")
    print("=" * 60)
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Number of devices: {jax.device_count()}")
    print("=" * 60 + "\n")

    # Print initial memory state
    print_device_memory_stats("Initial State")

    #### SINGLE LINEAR LAYER ####

    # 10 balanced target total parameter sizes (≤ 5k)
    n_samples = 2000
    random_state = 42

    # --- Gradual scaling from simple to complex ---
    # total parameter size increases smoothly up to ~20k
    num_reps = reps if reps is not None else 1
    min_params = min_params if min_params is not None else 1000
    max_params = max_params if max_params is not None else 20000

    target_param_sizes = jnp.linspace(
        min_params, max_params, num=num_reps, dtype=int
    ).tolist()
    n_classes_list = [10] * num_reps  # keep n_classes constant for simplicity

    dataset_nums = []

    for i, (target_params, n_classes) in enumerate(
        zip(target_param_sizes, n_classes_list)
    ):
        # derive n_features
        n_features = max(10, int(target_params / n_classes))

        # scale informative features proportionally to n_features (min 2, max 100)
        n_informative = min(max(2, int(0.6 * n_features)), 100)

        dataset_nums.append(
            {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
                "n_informative": n_informative,
                "total_params": n_features * n_classes,
            }
        )

    dataset_configs = {}
    training_configs = {}

    for dataset_num in dataset_nums:
        n_samples = dataset_num["n_samples"]
        n_features = dataset_num["n_features"]
        n_classes = dataset_num["n_classes"]
        n_informative = dataset_num["n_informative"]
        dataset_configs[
            f"random_classification_linear_only_{n_samples}_{n_features}_{n_classes}"
        ] = RandomClassificationConfig(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            random_state=42,
            train_test_split=1,
        )
        training_configs[
            f"random_classification_linear_only_{n_samples}_{n_features}_{n_classes}"
        ] = TrainingConfig(
            epochs=100,
            lr=0.001,
            optimizer="sgd",
            batch_size=100,
            loss="cross_entropy",
        )

        single_linear_collection = (dataset_configs, training_configs)

        linear_model_config = LinearModelConfig(
            loss="cross_entropy",
            hidden_dim=[],  # no hidden layers
        )

        full_config = Config(
            dataset=dataset_configs[
                f"random_classification_linear_only_{n_samples}_{n_features}_{n_classes}"
            ],
            model=linear_model_config,
            training=training_configs[
                f"random_classification_linear_only_{n_samples}_{n_features}_{n_classes}"
            ],
        )
        model_data = train_or_load(full_config)

        kfac_config = KFACConfig(
            build_config=KFACBuildConfig(use_pseudo_targets=True),
            run_config=KFACRunConfig(
                use_eigenvalue_correction=False, damping_lambda=0.1
            ),
        )
        ekfac_config = KFACConfig(
            build_config=KFACBuildConfig(use_pseudo_targets=True),
            run_config=KFACRunConfig(
                use_eigenvalue_correction=True, damping_lambda=0.1
            ),
        )

        results_dict = {}

        test_vectors = sample_gradient_from_output_distribution_batched(
            model_data=model_data,
            n_vectors=num_vectors,
            rng_key=jax.random.PRNGKey(123),
        )

        for kfac_config in [kfac_config, ekfac_config]:
            kfac_string = (
                "K-FAC"
                if not kfac_config.run_config.use_eigenvalue_correction
                else "E-KFAC"
            )

            print(f"\n{'#' * 60}")
            print(f"Processing: {kfac_string}")
            print(f"{'#' * 60}")

            kfac_model = KFAC.setup_with_run_and_build_config(
                full_config=full_config,
                build_config=kfac_config.build_config,
                run_config=kfac_config.run_config,
            )

            print_device_memory_stats(f"After {kfac_string} Setup")

            damping = kfac_model.damping()
            diff, _ = kfac_model.compute_ihvp(
                vector=test_vectors, damping=damping, metric=VectorMetric.RELATIVE_ERROR
            )

            print_device_memory_stats(f"After {kfac_string} IHVP Computation")

            # # plot the first four ihvp for both in line plots using subfigures in matplotlib
            # import matplotlib.pyplot as plt

            # fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            # for i in range(2):
            #     for j in range(2):
            #         idx = i * 2 + j
            #         axs[i, j].plot(
            #             hessian_result[idx],
            #             label="True Hessian IHVP",
            #             color="blue",
            #         )
            #         axs[i, j].plot(
            #             kfac_result[idx],
            #             label=f"{kfac_string} IHVP",
            #             color="orange",
            #         )
            #         axs[i, j].set_title(f"IHVP Comparison for Vector {idx + 1}")
            #         axs[i, j].legend()
            # plt.tight_layout()
            # plt.show()

            # diff = VectorMetric.RELATIVE_ERROR.compute(
            #     kfac_result, hessian_result, reduction="mean"
            # )
            # print_device_memory_stats(f"After {kfac_string} IHVP Comparison")

            results_dict[kfac_string] = {}
            results_dict[kfac_string]["relative_error"] = diff

            kfac_model.model_data.params
            results_dict[kfac_string]["num_params"] = (
                kfac_model.model_data.model.get_num_params(kfac_model.model_data.params)
            )

            # del kfac_hessian
            del kfac_model
            # del hessian_result
            # del kfac_result
            del diff

            # Force garbage collection to see memory release
            import gc

            gc.collect()
            print_device_memory_stats(f"After Deleting {kfac_string} Model & Hessian")

        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        for method, metrics in results_dict.items():
            print(f"\nMethod: {method}")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value}")
            print()

        print_device_memory_stats("Final State")


if __name__ == "__main__":
    check_memory()
