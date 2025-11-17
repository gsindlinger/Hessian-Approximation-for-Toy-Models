import copy
import json
import os
import time
from typing import Dict

import jax
from jax import numpy as jnp

from config.config import Config
from config.dataset_config import RandomClassificationConfig
from config.hessian_approximation_config import (
    HessianApproximationConfig,
    HessianName,
    KFACBuildConfig,
    KFACRunConfig,
)
from config.model_config import LinearModelConfig
from config.training_config import TrainingConfig
from hessian_approximations.hessian.hessian import Hessian
from hessian_approximations.kfac.kfac import KFAC
from metrics.full_matrix_metrics import MATRIX_METRICS, compare_matrices
from models.train import train_or_load


def compare_approximation_with_true_hessian(
    hessian_approxs: Dict[str, KFAC],
):
    results_dict = {}

    # assert that damping for all approximations is the same
    damping_value_1 = hessian_approxs["kfac"].damping()
    damping_value_2 = hessian_approxs["ekfac"].damping()

    assert jnp.isclose(damping_value_1, damping_value_2), (
        "Damping values for approximations are not the same."
    )

    config = copy.deepcopy(list(hessian_approxs.values())[0].full_config)
    config.hessian_approximation = HessianApproximationConfig(HessianName.HESSIAN)

    true_hessian = Hessian(full_config=config).compute_hessian()
    true_hessian = true_hessian + damping_value_1 * jnp.eye(true_hessian.shape[0])

    approximation_matrices = {}

    for approx_name, approx_method in hessian_approxs.items():
        approximation_matrix = approx_method.compute_hessian()
        approximation_matrices[approx_name] = approximation_matrix

        _, _, params, _ = train_or_load(approx_method.full_config)

        results_dict[approx_name] = compare_matrices(
            matrix_1=true_hessian,
            matrix_2=approximation_matrix,
            metrics=MATRIX_METRICS["all_matrix"],
        )

        # add number of parameters
        num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
        results_dict[approx_name]["num_params"] = num_params

    return results_dict


# 10 balanced target total parameter sizes (≤ 5k)
n_samples = 2000
random_state = 42

# --- Gradual scaling from simple to complex ---
# total parameter size increases smoothly up to ~5k
target_param_sizes = jnp.linspace(5000, 10000, num=10, dtype=int).tolist()
n_classes_list = [10] * 10  # keep n_classes constant at 10

dataset_nums = []

for i, (target_params, n_classes) in enumerate(zip(target_param_sizes, n_classes_list)):
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
    dataset_configs[f"random_classification_{n_samples}_{n_features}_{n_classes}"] = (
        RandomClassificationConfig(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            random_state=42,
            train_test_split=1,
        )
    )
    training_configs[f"random_classification_{n_samples}_{n_features}_{n_classes}"] = (
        TrainingConfig(
            epochs=100,
            lr=0.001,
            optimizer="sgd",
            batch_size=100,
            loss="cross_entropy",
        )
    )


# Linear Model Single Layer
results_dict = {}

for dataset_name, dataset_config in dataset_configs.items():
    linear_model_config = LinearModelConfig(loss="cross_entropy", hidden_dim=[])

    full_config = Config(
        dataset=dataset_config,
        model=linear_model_config,
        training=training_configs[dataset_name],
    )

    linear_hessian_configs_kfac: Dict[str, KFAC] = {
        "kfac": KFAC.setup_with_run_and_build_config(
            full_config=full_config,
            build_config=KFACBuildConfig(
                use_pseudo_targets=True,
            ),
            run_config=KFACRunConfig(
                use_eigenvalue_correction=False,
            ),
        ),
        "ekfac": KFAC.setup_with_run_and_build_config(
            full_config=full_config,
            build_config=KFACBuildConfig(
                use_pseudo_targets=True,
            ),
            run_config=KFACRunConfig(
                use_eigenvalue_correction=True,
            ),
        ),
    }

    print(f"Comparing Hessians for dataset: {dataset_name}")
    # add time taken for each approximation

    start_time = time.time()
    results_dict[dataset_name] = compare_approximation_with_true_hessian(
        hessian_approxs=linear_hessian_configs_kfac
    )
    end_time = time.time()
    print(f"Time taken for dataset {dataset_name}: {end_time - start_time} seconds")
    results_dict[dataset_name]["time_taken"] = end_time - start_time
    print(f"Completed dataset: {dataset_name}")

# save results to a json file
# folder path
folder = "data/artifacts/results/"
# check if folder exists, if not create it

# timestamp for unique filename
timestamp = time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(folder):
    os.makedirs(folder)
with open(f"{folder}/kfac_hessian_comparison_results_{timestamp}.json", "w") as f:
    json.dump(results_dict, f, indent=4)
