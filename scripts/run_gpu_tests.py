import copy
import time

from jax import numpy as jnp

from config.config import Config
from config.dataset_config import RandomClassificationConfig
from config.hessian_approximation_config import (
    HessianApproximationConfig,
    HessianName,
    KFACBuildConfig,
    KFACConfig,
    KFACRunConfig,
)
from config.model_config import MLPModelConfig
from config.training_config import TrainingConfig
from hessian_approximations.hessian.hessian import Hessian
from hessian_approximations.kfac.kfac import KFAC
from metrics.full_matrix_metrics import MATRIX_METRICS, compare_matrices
from utils.utils import plot_metric_results, save_results


def compare_approximation_with_true_hessian(
    full_config: Config,
    hessian_approxs_configs: tuple[KFACConfig, KFACConfig],
):
    results_dict = {}

    config = copy.deepcopy(full_config)
    config.hessian_approximation = HessianApproximationConfig(HessianName.HESSIAN)

    true_hessian_model = Hessian(full_config=config)

    start_time_hessian = time.time()
    true_hessian = true_hessian_model.compute_hessian()
    print("Data type of true Hessian:", true_hessian.dtype)
    end_time_hessian = time.time()
    results_dict["true_hessian_time"] = end_time_hessian - start_time_hessian

    approximation_matrices = {}

    for approx_config in hessian_approxs_configs:
        approx_name = (
            "ekfac" if approx_config.run_config.use_eigenvalue_correction else "kfac"
        )
        start_time_approx = time.time()
        approx_method = KFAC.setup_with_run_and_build_config(
            full_config=full_config,
            build_config=approx_config.build_config,
            run_config=approx_config.run_config,
        )

        approximation_matrix = approx_method.compute_hessian()
        end_time_approx = time.time()
        end_time_approx = time.time()
        approximation_matrices[approx_name] = approximation_matrix

        results_dict[approx_name] = compare_matrices(
            matrix_1=true_hessian + approx_method.damping(),
            matrix_2=approximation_matrix,
            metrics=MATRIX_METRICS["comprehensive"],
        )
        results_dict[approx_name]["time"] = end_time_approx - start_time_approx
        results_dict[approx_name]["num_params"] = (
            approx_method.model_data.model.get_num_params(
                approx_method.model_data.params
            )
        )

    return results_dict


#### SINGLE LINEAR LAYER ####

#### SINGLE LINEAR LAYER ####

# 10 balanced target total parameter sizes (≤ 5k)
n_samples = 2000
random_state = 42

# --- Gradual scaling from simple to complex ---
# total parameter size increases smoothly up to ~20k
num_reps = 1
min_params = 17000
max_params = 20000

target_param_sizes = jnp.linspace(
    min_params, max_params, num=num_reps, dtype=int
).tolist()
n_classes_list = [3] * num_reps  # keep n_classes constant for simplicity

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

#### MLP LAYERS ####

# 10 balanced target total parameter sizes (≤ 5k)
n_samples = 2000
random_state = 42

target_param_sizes = jnp.linspace(
    min_params, max_params, num=num_reps, dtype=int
).tolist()
n_classes_list = [10] * num_reps  # keep n_classes constant for simplicity


def choose_hidden_dims(target_params: int):
    """
    Choose hidden layer sizes dynamically so that total params ≈ target_params.
    Formula for params: n_features*h1 + h1*h2 + ... + hL*n_classes
    """
    # simple heuristic: one or two hidden layers that fit target budget
    if target_params <= 10000:
        return [20]
    elif target_params <= 15000:
        return [30]
    else:
        return [50]


dataset_nums = []

for target_params, n_classes in zip(target_param_sizes, n_classes_list):
    # derive n_features
    n_features = max(10, int(target_params / n_classes))
    n_informative = min(max(2, int(0.6 * n_features)), 100)

    hidden_dims = choose_hidden_dims(target_params)

    total_params = n_features * hidden_dims[0]
    for i in range(len(hidden_dims) - 1):
        total_params += hidden_dims[i] * hidden_dims[i + 1]
    total_params += hidden_dims[-1] * n_classes

    dataset_nums.append(
        {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "n_informative": n_informative,
            "hidden_dims": hidden_dims,
            "total_params": total_params,
        }
    )


dataset_configs = {}
training_configs = {}
hidden_dims = {}

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
        random_state=42,
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
    hidden_dims[
        f"random_classification_mlp_{n_samples}_{n_features}_{hidden_dims_list}_{n_classes}"
    ] = hidden_dims_list

mlp_collection = (dataset_configs, training_configs, hidden_dims)


# Compute for both single linear layer and MLP layer datasets
for collection in [single_linear_collection, mlp_collection]:
    dataset_configs, training_configs = collection[:2]
    if len(collection) == 3:
        hidden_dims = collection[2]
    else:
        hidden_dims = {
            name: [] for name in dataset_configs.keys()
        }  # empty list for linear models

    results_dict = {}
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

#### MLP LAYERS ####

# 10 balanced target total parameter sizes (≤ 5k)
n_samples = 2000
random_state = 42

target_param_sizes = jnp.linspace(
    min_params, max_params, num=num_reps, dtype=int
).tolist()
n_classes_list = [10] * num_reps  # keep n_classes constant for simplicity


def choose_hidden_dims(target_params: int):
    """
    Choose hidden layer sizes dynamically so that total params ≈ target_params.
    Formula for params: n_features*h1 + h1*h2 + ... + hL*n_classes
    """
    # simple heuristic: one or two hidden layers that fit target budget
    if target_params <= 10000:
        return [20]
    elif target_params <= 15000:
        return [30]
    else:
        return [50]


dataset_nums = []

for target_params, n_classes in zip(target_param_sizes, n_classes_list):
    # derive n_features
    n_features = max(10, int(target_params / n_classes))
    n_informative = min(max(2, int(0.6 * n_features)), 100)

    hidden_dims = choose_hidden_dims(target_params)

    total_params = n_features * hidden_dims[0]
    for i in range(len(hidden_dims) - 1):
        total_params += hidden_dims[i] * hidden_dims[i + 1]
    total_params += hidden_dims[-1] * n_classes

    dataset_nums.append(
        {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "n_informative": n_informative,
            "hidden_dims": hidden_dims,
            "total_params": total_params,
        }
    )


dataset_configs = {}
training_configs = {}
hidden_dims = {}

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
        random_state=42,
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
    hidden_dims[
        f"random_classification_mlp_{n_samples}_{n_features}_{hidden_dims_list}_{n_classes}"
    ] = hidden_dims_list

mlp_collection = (dataset_configs, training_configs, hidden_dims)


# Compute for both single linear layer and MLP layer datasets
for collection in [single_linear_collection, mlp_collection]:
    dataset_configs, training_configs = collection[:2]
    if len(collection) == 3:
        hidden_dims = collection[2]
    else:
        hidden_dims = {
            name: [] for name in dataset_configs.keys()
        }  # empty list for linear models

    results_dict = {}

    for dataset_name, dataset_config in dataset_configs.items():
        linear_model_config = MLPModelConfig(
            loss="cross_entropy", hidden_dim=hidden_dims[dataset_name]
        )
    for dataset_name, dataset_config in dataset_configs.items():
        linear_model_config = MLPModelConfig(
            loss="cross_entropy", hidden_dim=hidden_dims[dataset_name]
        )

        full_config = Config(
            dataset=dataset_config,
            model=linear_model_config,
            training=training_configs[dataset_name],
        )
        full_config = Config(
            dataset=dataset_config,
            model=linear_model_config,
            training=training_configs[dataset_name],
        )

        kfac_config = KFACConfig(
            build_config=KFACBuildConfig(),
            run_config=KFACRunConfig(use_eigenvalue_correction=False),
        )

        ekfac_config = KFACConfig(
            build_config=KFACBuildConfig(),
            run_config=KFACRunConfig(use_eigenvalue_correction=True),
        )

        linear_hessian_configs_kfac = (kfac_config, ekfac_config)

        print(f"Comparing Hessians for dataset: {dataset_name}")
        # add time taken for each approximation
        print(f"Comparing Hessians for dataset: {dataset_name}")
        # add time taken for each approximation

        start_time = time.time()
        results_dict[dataset_name] = compare_approximation_with_true_hessian(
            full_config=full_config, hessian_approxs_configs=linear_hessian_configs_kfac
        )
        end_time = time.time()
        print(f"Time taken for dataset {dataset_name}: {end_time - start_time} seconds")
        results_dict[dataset_name]["time_taken_overall"] = end_time - start_time
        print(f"Completed dataset: {dataset_name}")

    # save results to a json file
    # folder path
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if collection == single_linear_collection:
        folder = f"data/artifacts/results/single_linear/{timestamp}/"
    else:
        folder = f"data/artifacts/results/mlp/{timestamp}/"
    save_results(
        results_dict, f"{folder}/kfac_hessian_comparison_results_{timestamp}.json"
    )

    plot_metric_results(
        results=results_dict,
        filename=f"{folder}/kfac_hessian_comparison_metrics_{timestamp}",
        metrics=[
            "relative_frobenius",
            "cosine_similarity",
            "trace_distance",
        ],
    )
    print(f"Saved results to {folder}/kfac_hessian_comparison_results_{timestamp}.json")
