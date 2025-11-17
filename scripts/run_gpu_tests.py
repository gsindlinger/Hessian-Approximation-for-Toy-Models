import copy
import json
import os
import time
from typing import Dict

from jax import numpy as jnp

from config.config import Config
from config.dataset_config import RandomClassificationConfig
from config.hessian_approximation_config import (
    HessianApproximationConfig,
    HessianName,
    KFACBuildConfig,
    KFACRunConfig,
)
from config.model_config import MLPModelConfig
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

    true_hessian_model = Hessian(full_config=config)

    start_time_hessian = time.time()
    true_hessian = true_hessian_model.compute_hessian()
    true_hessian = true_hessian + damping_value_1 * jnp.eye(true_hessian.shape[0])
    end_time_hessian = time.time()
    results_dict["true_hessian_time"] = end_time_hessian - start_time_hessian

    approximation_matrices = {}

    for approx_name, approx_method in hessian_approxs.items():
        start_time_approx = time.time()
        approximation_matrix = approx_method.compute_hessian()
        end_time_approx = time.time()
        approximation_matrices[approx_name] = approximation_matrix

        results_dict[approx_name] = compare_matrices(
            matrix_1=true_hessian,
            matrix_2=approximation_matrix,
            metrics=MATRIX_METRICS["all_matrix"],
        )
        results_dict[approx_name]["time"] = end_time_approx - start_time_approx

        model, _, params, _ = train_or_load(approx_method.full_config)
        results_dict[approx_name]["num_params"] = model.get_num_params(params)

    return results_dict


#### SINGLE LINEAR LAYER ####

# 10 balanced target total parameter sizes (≤ 5k)
n_samples = 2000
random_state = 42

# --- Gradual scaling from simple to complex ---
# total parameter size increases smoothly up to ~20k
num_reps = 10
min_params = 1000
max_params = 20000

target_param_sizes = jnp.linspace(
    min_params, max_params, num=num_reps, dtype=int
).tolist()
n_classes_list = [10] * num_reps  # keep n_classes constant for simplicity

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

    for dataset_name, dataset_config in dataset_configs.items():
        linear_model_config = MLPModelConfig(
            loss="cross_entropy", hidden_dim=hidden_dims[dataset_name]
        )

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
        results_dict[dataset_name]["time_taken_overall"] = end_time - start_time
        print(f"Completed dataset: {dataset_name}")

    # save results to a json file
    # folder path
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if collection == single_linear_collection:
        folder = f"data/artifacts/results/single_linear/{timestamp}/"
    else:
        folder = f"data/artifacts/results/mlp/{timestamp}/"
    # check if folder exists, if not create it
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(f"{folder}/kfac_hessian_comparison_results.json", "w") as f:
        json.dump(results_dict, f, indent=4)

    # create subplot with relative frobenius, relative spectral, cosine similarity, trace distance
    # for each dataset and each approximation method by parameter size
    # save the plot as pdf and png

    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 4))
    metrics_to_plot = [
        "relative_frobenius",
        "spectral_relative",
        "cosine_similarity",
        "trace_distance",
    ]

    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "figure.figsize": (16, 4),
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.5,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "legend.framealpha": 1.0,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Define colors for consistency
    colors = {"kfac": "#1f77b4", "ekfac": "#ff7f0e"}
    markers = {"kfac": "o", "ekfac": "s"}

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]

        for approx_method in ["kfac", "ekfac"]:
            x = []
            y = []
            for dataset_name in results_dict.keys():
                param_size = results_dict[dataset_name][approx_method]["num_params"]
                metric_value = results_dict[dataset_name][approx_method][metric]
                x.append(param_size)
                y.append(metric_value)

            # Sort x and y based on x
            x, y = zip(*sorted(zip(x, y)))

            label = "KFAC" if approx_method == "kfac" else "EKFAC"
            ax.plot(
                x,
                y,
                marker=markers[approx_method],
                label=label,
                color=colors[approx_method],
                linewidth=2,
                markersize=6,
                markerfacecolor=colors[approx_method],
                markeredgecolor="white",
                markeredgewidth=0.5,
            )

        # Format metric name for LaTeX
        metric_latex = (
            metric.replace("_", " ")
            .replace("relative", "Relative")
            .replace("spectral", "Spectral")
            .replace("frobenius", "Frobenius")
            .replace("cosine", "Cosine")
            .replace("similarity", "Similarity")
            .replace("trace", "Trace")
            .replace("distance", "Distance")
        )

        ax.set_xlabel(r"Number of Parameters", fontsize=12)
        # ax.set_ylabel(metric_latex, fontsize=12)
        ax.set_title(metric_latex, fontsize=12, pad=10)

        # Improve grid
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Set scientific notation for y-axis if values are very small
        if (
            max(
                [
                    results_dict[dataset_name][approx_method][metric]
                    for dataset_name in results_dict.keys()
                    for approx_method in ["kfac", "ekfac"]
                ]
            )
            < 0.01
        ):
            ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # Add legend only to the first subplot
        if i == 0:
            ax.legend(loc="upper left", fontsize=10)

        # Improve tick formatting
        ax.tick_params(axis="both", which="major", labelsize=10)

        # Set spine properties
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("black")

    plt.suptitle(
        r"KFAC vs EKFAC Comparison to Ground Truth Hessian", fontsize=14, y=1.02
    )
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3)

    # Save with high DPI for publication quality
    plt.savefig(
        f"{folder}/kfac_hessian_comparison_{timestamp}.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.savefig(
        f"{folder}/kfac_hessian_comparison_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )

    print(f"Saved results to {folder}/kfac_hessian_comparison_results_{timestamp}.json")
