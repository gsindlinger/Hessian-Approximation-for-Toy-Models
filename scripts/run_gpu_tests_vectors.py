import copy
import json
import os
import time
from typing import List

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

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
from metrics.vector_metrics import VectorMetric


def generate_out_of_distribution_vectors(
    n_params: int,
    n_vectors: int,
    rng_key: Array,
    vector_type: str = "random_normal",
) -> Float[Array, "n_vectors n_params"]:
    """
    Generate test vectors that are out of the training data distribution.

    Args:
        n_params: Number of parameters
        n_vectors: Number of test vectors to generate
        rng_key: JAX random key
        vector_type: Type of vectors to generate
            - "random_normal": Standard normal vectors
            - "random_uniform": Uniform [-1, 1] vectors
            - "sparse": Sparse vectors (90% zeros)
            - "structured": Structured patterns (alternating signs)
    """
    if vector_type == "random_normal":
        vectors = jax.random.normal(rng_key, (n_vectors, n_params))
    elif vector_type == "random_uniform":
        vectors = jax.random.uniform(
            rng_key, (n_vectors, n_params), minval=-1.0, maxval=1.0
        )
    elif vector_type == "sparse":
        vectors = jax.random.normal(rng_key, (n_vectors, n_params))
        # Make 90% of entries zero
        mask_key = jax.random.split(rng_key)[0]
        mask = jax.random.bernoulli(mask_key, p=0.1, shape=(n_vectors, n_params))
        vectors = vectors * mask
    elif vector_type == "structured":
        # Create alternating pattern vectors
        vectors = jnp.ones((n_vectors, n_params))
        for i in range(n_vectors):
            pattern = jnp.array([(-1) ** (j + i) for j in range(n_params)])
            vectors = vectors.at[i].set(pattern)
    else:
        raise ValueError(f"Unknown vector_type: {vector_type}")

    # Normalize vectors
    norms = jnp.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)


def compare_hvp_ihvp(
    full_config: Config,
    hessian_approxs_configs: tuple[KFACConfig, KFACConfig],
    n_test_vectors: int = 100,
    vector_types: List[str] | None = None,
    rng_seed: int = 123,
):
    """
    Compare HVP and IHVP between true Hessian and approximations.

    Args:
        full_config: Configuration for the experiment
        hessian_approxs_configs: Tuple of (KFAC, EKFAC) configurations
        n_test_vectors: Number of test vectors per vector type
        vector_types: List of vector types to test
        rng_seed: Random seed for reproducibility
    """
    if vector_types is None:
        vector_types = ["random_normal"]

    results_dict = {}

    # Setup true Hessian
    config = copy.deepcopy(full_config)
    config.hessian_approximation = HessianApproximationConfig(HessianName.HESSIAN)

    print("Setting up true Hessian...")
    true_hessian_model = Hessian(full_config=config)

    # Get number of parameters
    n_params = true_hessian_model.model_data.model.get_num_params(
        true_hessian_model.model_data.params
    )
    print(f"Number of parameters: {n_params}")

    # Initialize approximation methods
    approximation_methods = {}
    for approx_config in hessian_approxs_configs:
        approx_name = (
            "ekfac" if approx_config.run_config.use_eigenvalue_correction else "kfac"
        )
        print(f"Setting up {approx_name.upper()}...")
        approx_method = KFAC.setup_with_run_and_build_config(
            full_config=full_config,
            build_config=approx_config.build_config,
            run_config=approx_config.run_config,
        )
        approximation_methods[approx_name] = approx_method

    # Define metrics to compute
    hvp_metrics = [
        VectorMetric.RELATIVE_ERROR,
        VectorMetric.COSINE_SIMILARITY,
        VectorMetric.ABSOLUTE_L2_DIFF,
        VectorMetric.SIGN_AGREEMENT,
        VectorMetric.RELATIVE_ENERGY_DIFF,
    ]

    ihvp_metrics = [
        VectorMetric.RELATIVE_ERROR,
        VectorMetric.COSINE_SIMILARITY,
        VectorMetric.ABSOLUTE_L2_DIFF,
        VectorMetric.SIGN_AGREEMENT,
        VectorMetric.INNER_PRODUCT_DIFF,
        VectorMetric.INNER_PRODUCT_RATIO,
    ]

    # Test each vector type
    for vector_type in vector_types:
        print(f"\nTesting with {vector_type} vectors...")
        results_dict[vector_type] = {}

        # Generate test vectors
        rng_key = jax.random.PRNGKey(rng_seed)
        test_vectors = generate_out_of_distribution_vectors(
            n_params, n_test_vectors, rng_key, vector_type
        )

        # Compute true HVPs and IHVPs
        print("  Computing true HVPs...")
        start_time = time.time()
        true_hvps = jax.vmap(true_hessian_model.compute_hvp)(test_vectors)
        hvp_time = time.time() - start_time

        print("  Computing true IHVPs...")
        start_time = time.time()
        true_ihvps = jax.vmap(true_hessian_model.compute_ihvp)(test_vectors)
        ihvp_time = time.time() - start_time

        results_dict[vector_type]["true_hessian"] = {
            "hvp_time": hvp_time,
            "ihvp_time": ihvp_time,
        }

        # Test each approximation method
        for approx_name, approx_method in approximation_methods.items():
            print(f"  Testing {approx_name.upper()}...")
            results_dict[vector_type][approx_name] = {}

            # Compute approximate HVPs
            start_time = time.time()
            approx_hvps = jax.vmap(approx_method.compute_hvp)(test_vectors)
            approx_hvp_time = time.time() - start_time

            # Compute approximate IHVPs
            start_time = time.time()
            approx_ihvps = jax.vmap(approx_method.compute_ihvp)(test_vectors)
            approx_ihvp_time = time.time() - start_time

            results_dict[vector_type][approx_name]["hvp_time"] = approx_hvp_time
            results_dict[vector_type][approx_name]["ihvp_time"] = approx_ihvp_time

            # Compute HVP metrics
            results_dict[vector_type][approx_name]["hvp_metrics"] = {}
            for metric in hvp_metrics:
                metric_value = metric.compute(true_hvps, approx_hvps, reduction="mean")
                results_dict[vector_type][approx_name]["hvp_metrics"][metric.value] = (
                    float(metric_value)
                )

            # Compute IHVP metrics
            results_dict[vector_type][approx_name]["ihvp_metrics"] = {}
            for metric in ihvp_metrics:
                if metric in [
                    VectorMetric.INNER_PRODUCT_DIFF,
                    VectorMetric.INNER_PRODUCT_RATIO,
                ]:
                    # Use original test vectors as auxiliary vectors for inner product metrics
                    metric_value = metric.compute(
                        true_ihvps, approx_ihvps, x=test_vectors, reduction="mean"
                    )
                else:
                    metric_value = metric.compute(
                        true_ihvps, approx_ihvps, reduction="mean"
                    )
                results_dict[vector_type][approx_name]["ihvp_metrics"][metric.value] = (
                    float(metric_value)
                )

            print(
                f"    HVP Relative Error: {results_dict[vector_type][approx_name]['hvp_metrics']['relative_error']:.6f}"
            )
            print(
                f"    IHVP Relative Error: {results_dict[vector_type][approx_name]['ihvp_metrics']['relative_error']:.6f}"
            )

    return results_dict


#### EXPERIMENT SETUP ####

# Configuration parameters
n_samples = 2000
random_state = 42
num_reps = 10
min_params = 100
max_params = 1000

target_param_sizes = jnp.linspace(
    min_params, max_params, num=num_reps, dtype=int
).tolist()
n_classes_list = [10] * num_reps

# Test settings
n_test_vectors = 20
vector_types = ["random_normal", "random_uniform", "sparse", "structured"]

#### SINGLE LINEAR LAYER EXPERIMENTS ####

dataset_nums = []
for i, (target_params, n_classes) in enumerate(zip(target_param_sizes, n_classes_list)):
    n_features = max(10, int(target_params / n_classes))
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

#### MLP LAYER EXPERIMENTS ####


def choose_hidden_dims(target_params: int):
    if target_params <= 10000:
        return [20]
    elif target_params <= 15000:
        return [30]
    else:
        return [50]


dataset_nums_mlp = []
for target_params, n_classes in zip(target_param_sizes, n_classes_list):
    n_features = max(10, int(target_params / n_classes))
    n_informative = min(max(2, int(0.6 * n_features)), 100)
    hidden_dims = choose_hidden_dims(target_params)

    total_params = n_features * hidden_dims[0]
    for i in range(len(hidden_dims) - 1):
        total_params += hidden_dims[i] * hidden_dims[i + 1]
    total_params += hidden_dims[-1] * n_classes

    dataset_nums_mlp.append(
        {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "n_informative": n_informative,
            "hidden_dims": hidden_dims,
            "total_params": total_params,
        }
    )

dataset_configs_mlp = {}
training_configs_mlp = {}
hidden_dims_mlp = {}

for dataset_num in dataset_nums_mlp:
    n_samples = dataset_num["n_samples"]
    n_features = dataset_num["n_features"]
    n_classes = dataset_num["n_classes"]
    n_informative = dataset_num["n_informative"]
    hidden_dims_list = dataset_num["hidden_dims"]

    dataset_configs_mlp[
        f"random_classification_mlp_{n_samples}_{n_features}_{hidden_dims_list}_{n_classes}"
    ] = RandomClassificationConfig(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        random_state=42,
        train_test_split=1,
    )

    training_configs_mlp[
        f"random_classification_mlp_{n_samples}_{n_features}_{hidden_dims_list}_{n_classes}"
    ] = TrainingConfig(
        epochs=1000,
        lr=0.001,
        optimizer="sgd",
        batch_size=100,
        loss="cross_entropy",
    )

    hidden_dims_mlp[
        f"random_classification_mlp_{n_samples}_{n_features}_{hidden_dims_list}_{n_classes}"
    ] = hidden_dims_list

mlp_collection = (dataset_configs_mlp, training_configs_mlp, hidden_dims_mlp)

#### RUN EXPERIMENTS ####

for collection_name, collection in [
    ("single_linear", single_linear_collection),
    ("mlp", mlp_collection),
]:
    print(f"\n{'=' * 80}")
    print(f"Running {collection_name.upper()} experiments")
    print(f"{'=' * 80}\n")

    dataset_configs, training_configs = collection[:2]
    if len(collection) == 3:
        hidden_dims = collection[2]
    else:
        hidden_dims = {name: [] for name in dataset_configs.keys()}

    all_results = {}

    for dataset_name, dataset_config in dataset_configs.items():
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        model_config = MLPModelConfig(
            loss="cross_entropy", hidden_dim=hidden_dims[dataset_name]
        )

        full_config = Config(
            dataset=dataset_config,
            model=model_config,
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

        hessian_configs = (kfac_config, ekfac_config)

        start_time = time.time()
        results = compare_hvp_ihvp(
            full_config=full_config,
            hessian_approxs_configs=hessian_configs,
            n_test_vectors=n_test_vectors,
            vector_types=vector_types,
        )
        end_time = time.time()

        results["total_time"] = end_time - start_time
        all_results[dataset_name] = results

        print(f"\nCompleted {dataset_name} in {end_time - start_time:.2f} seconds")

    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    folder = f"data/artifacts/results/{collection_name}_hvp_ihvp/{timestamp}/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(f"{folder}/hvp_ihvp_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nSaved results to {folder}/hvp_ihvp_comparison_results.json")

    # Create visualizations
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "figure.figsize": (16, 10),
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.5,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )

    # Plot for each vector type
    for vector_type in vector_types:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            f"HVP and IHVP Comparison: {vector_type.replace('_', ' ').title()} Vectors",
            fontsize=16,
            y=0.995,
        )

        colors = {"kfac": "#1f77b4", "ekfac": "#ff7f0e"}
        markers = {"kfac": "o", "ekfac": "s"}

        # Metrics to plot
        hvp_metrics_to_plot = [
            "relative_error",
            "cosine_similarity",
            "absolute_l2_diff",
        ]
        ihvp_metrics_to_plot = [
            "relative_error",
            "cosine_similarity",
            "inner_product_diff",
        ]

        # Plot HVP metrics
        for idx, metric in enumerate(hvp_metrics_to_plot):
            ax = axes[0, idx]

            for approx_method in ["kfac", "ekfac"]:
                x = []
                y = []
                for dataset_name in all_results.keys():
                    if vector_type in all_results[dataset_name]:
                        # Extract parameter count from dataset name or results
                        param_size = int(dataset_name.split("_")[-3]) * int(
                            dataset_name.split("_")[-1]
                        )
                        metric_value = all_results[dataset_name][vector_type][
                            approx_method
                        ]["hvp_metrics"][metric]
                        x.append(param_size)
                        y.append(metric_value)

                if x:  # Only plot if we have data
                    x, y = zip(*sorted(zip(x, y)))
                    label = approx_method.upper()
                    ax.plot(
                        x,
                        y,
                        marker=markers[approx_method],
                        label=label,
                        color=colors[approx_method],
                        linewidth=2,
                        markersize=6,
                    )

            metric_title = metric.replace("_", " ").title()
            ax.set_title(f"HVP: {metric_title}", fontsize=12, pad=10)
            ax.set_xlabel("Number of Parameters", fontsize=11)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)

            if idx == 0:
                ax.legend(loc="best", fontsize=10)

        # Plot IHVP metrics
        for idx, metric in enumerate(ihvp_metrics_to_plot):
            ax = axes[1, idx]

            for approx_method in ["kfac", "ekfac"]:
                x = []
                y = []
                for dataset_name in all_results.keys():
                    if vector_type in all_results[dataset_name]:
                        param_size = int(dataset_name.split("_")[-3]) * int(
                            dataset_name.split("_")[-1]
                        )
                        metric_value = all_results[dataset_name][vector_type][
                            approx_method
                        ]["ihvp_metrics"][metric]
                        x.append(param_size)
                        y.append(metric_value)

                if x:  # Only plot if we have data
                    x, y = zip(*sorted(zip(x, y)))
                    label = approx_method.upper()
                    ax.plot(
                        x,
                        y,
                        marker=markers[approx_method],
                        label=label,
                        color=colors[approx_method],
                        linewidth=2,
                        markersize=6,
                    )

            metric_title = metric.replace("_", " ").title()
            ax.set_title(f"IHVP: {metric_title}", fontsize=12, pad=10)
            ax.set_xlabel("Number of Parameters", fontsize=11)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)

            if idx == 0:
                ax.legend(loc="best", fontsize=10)

        plt.tight_layout()

        # Save plots
        plt.savefig(
            f"{folder}/hvp_ihvp_comparison_{vector_type}_{timestamp}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            f"{folder}/hvp_ihvp_comparison_{vector_type}_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print(f"Saved plots to {folder}")
