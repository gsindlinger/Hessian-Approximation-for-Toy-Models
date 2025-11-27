import argparse

import jax
from jax import numpy as jnp

from src.config.config import Config
from src.config.dataset_config import RandomClassificationConfig
from src.config.hessian_approximation_config import (
    KFACBuildConfig,
    KFACConfig,
    KFACRunConfig,
)
from src.config.model_config import LinearModelConfig
from src.config.training_config import TrainingConfig
from src.hessian_approximations.hessian.hessian import Hessian
from src.hessian_approximations.kfac.kfac_service import KFAC
from src.utils.utils import (
    get_device_memory_stats,
    get_total_jax_memory,
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

    return parser.parse_args()


def check_memory():
    #### PARSE ARGUMENTS ####
    args = parse_args()
    reps = args.reps
    min_params = args.n_params_min
    max_params = args.n_params_max

    #### DEVICE DIAGNOSTICS ####
    print("\n" + "=" * 60)
    print("JAX DEVICE CONFIGURATION")
    print("=" * 60)
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Number of devices: {jax.device_count()}")
    print("=" * 60 + "\n")

    # Print initial memory state
    print(get_device_memory_stats("Initial State"))

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
            epochs=1000,
            lr=0.001,
            optimizer="sgd",
            batch_size=100,
            loss="cross_entropy",
        )

        single_linear_collection = (dataset_configs, training_configs)

        linear_model_config = LinearModelConfig(
            loss="cross_entropy",
            hidden_dim=[10, 3],  # no hidden layers
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

        # print(get_device_memory_stats("Before Hessian Computation")

        # hessian = Hessian(full_config).compute_hessian()
        # hessian_bytes = get_total_jax_memory(hessian)

        # print("\n### True Hessian Memory ###")
        # print(f"Stored array size: {hessian_bytes / (1024**2):.2f} MB")
        # print("Data Type:", hessian.dtype)
        # print_array_device_info(hessian, "Hessian")
        # print(get_device_memory_stats("After Hessian Computation")

        kfac_config = KFACConfig(
            build_config=KFACBuildConfig(use_pseudo_targets=True),
            run_config=KFACRunConfig(use_eigenvalue_correction=False),
        )
        ekfac_config = KFACConfig(
            build_config=KFACBuildConfig(use_pseudo_targets=True),
            run_config=KFACRunConfig(use_eigenvalue_correction=True),
        )

        results_dict = {}

        for kfac_config in [kfac_config, ekfac_config]:
            kfac_string = (
                "K-FAC"
                if not kfac_config.run_config.use_eigenvalue_correction
                else "E-KFAC"
            )

            print(f"\n{'#' * 60}")
            print(f"Processing: {kfac_string}")
            print(f"{'#' * 60}")

            print(get_device_memory_stats(f"Before {kfac_string} Computation"))

            kfac_model = KFAC.setup_with_run_and_build_config(
                full_config=full_config,
                build_config=kfac_config.build_config,
                run_config=kfac_config.run_config,
            )

            kfac_bytes = get_total_jax_memory(kfac_model)
            print(f"\n### {kfac_string} Model Memory ###")
            print(f"Stored data size: {kfac_bytes / (1024**2):.2f} MB")
            print(get_device_memory_stats(f"After {kfac_string} Setup"))

            damping = kfac_model.damping()
            comparison_matrix = Hessian(full_config).compute_hessian(damping=damping)
            norm_result = kfac_model.compare_hessians(
                comparison_matrix=comparison_matrix, damping=damping
            )
            kfac_hessian = kfac_model.compute_hessian(damping=damping)

            # plot eigenvalues for visual comparison
            import matplotlib.pyplot as plt

            eigs_comp = jnp.linalg.eigvalsh(comparison_matrix)
            eigs_kfac = jnp.linalg.eigvalsh(kfac_hessian)
            plt.figure(figsize=(10, 6))
            plt.plot(
                eigs_comp,
                label="True Hessian Eigenvalues",
                marker="o",
                linestyle="None",
                markersize=4,
            )
            plt.plot(
                eigs_kfac,
                label=f"{kfac_string} Eigenvalues",
                marker="x",
                linestyle="None",
                markersize=4,
            )
            plt.yscale("log")
            plt.xlabel("Index")
            plt.ylabel("Eigenvalue (log scale)")
            plt.title(f"Eigenvalue Comparison: True Hessian vs {kfac_string}")
            plt.legend()
            plt.grid(True)
            plt.show()

            results_dict[kfac_string] = {}
            results_dict[kfac_string]["l2_norm_difference"] = norm_result

            kfac_model.model_context.params
            results_dict[kfac_string]["num_params"] = (
                kfac_model.model_context.model.get_num_params(
                    kfac_model.model_context.params
                )
            )

            # del kfac_hessian
            del kfac_model

            # Force garbage collection to see memory release
            import gc

            gc.collect()
            print(
                get_device_memory_stats(f"After Deleting {kfac_string} Model & Hessian")
            )

        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        for method, metrics in results_dict.items():
            print(f"\nMethod: {method}")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value}")
            print()

        print(get_device_memory_stats("Final State"))


if __name__ == "__main__":
    check_memory()
