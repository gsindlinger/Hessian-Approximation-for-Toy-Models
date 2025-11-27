import json
import logging
import os
from time import time
from typing import Dict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from chex import PRNGKey
from jax import flatten_util
from jaxtyping import Array, Float

from src.hessian_approximations.kfac.kfac_data_provider import KFACProvider

from ..models.base import ApproximationModel
from ..models.dataclasses.model_context import ModelContext

logger = logging.getLogger(__name__)


def plot_training_curve(train_losses, val_losses, title="Training Curve"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_regression_results(
    inputs, true_values, predicted_values, title="Regression Results"
):
    plt.figure(figsize=(10, 6))
    plt.scatter(inputs.numpy(), true_values.numpy(), label="True Values", alpha=0.6)
    plt.scatter(
        inputs.numpy(),
        predicted_values.numpy(),
        label="Predicted Values",
        alpha=0.6,
    )
    # draw regression line
    sorted_indices = inputs[:, 0].argsort()
    plt.plot(
        inputs[sorted_indices].numpy(),
        predicted_values[sorted_indices].numpy(),
        color="red",
        linewidth=2,
        label="Regression Line",
    )

    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def get_total_jax_memory(obj, seen=None):
    """
    Recursively compute the total memory footprint of all JAX arrays reachable
    from a Python object (including dicts, dataclasses, nested attributes).
    """
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)

    total = 0

    # -----------------------
    # Case 1 — JAX arrays
    # -----------------------
    if isinstance(obj, (jax.Array, jax.numpy.ndarray)):
        try:
            return obj.size * obj.dtype.itemsize
        except:
            return 0

    # -----------------------
    # Case 2 — NumPy arrays
    # -----------------------
    if isinstance(obj, np.ndarray):
        return obj.nbytes

    # -----------------------
    # Case 3 — Primitive types
    # -----------------------
    if isinstance(obj, (int, float, bool, str, bytes, type(None))):
        return 0

    # -----------------------
    # Case 4 — Containers
    # -----------------------
    if isinstance(obj, dict):
        for v in obj.values():
            total += get_total_jax_memory(v, seen)
        return total

    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            total += get_total_jax_memory(v, seen)
        return total

    # -----------------------
    # Case 5 — Objects with __dict__
    # -----------------------
    if hasattr(obj, "__dict__"):
        for key, value in vars(obj).items():
            total += get_total_jax_memory(value, seen)
        return total

    # -----------------------
    # Case 6 — Objects with slots
    # -----------------------
    if hasattr(obj, "__slots__"):
        for slot in obj.__slots__:
            if hasattr(obj, slot):
                total += get_total_jax_memory(getattr(obj, slot), seen)
        return total

    # -----------------------
    # Unknown type → skip
    # -----------------------
    return 0


def get_device_memory_stats(label="") -> str:
    """Get current GPU memory statistics for all JAX devices as a formatted string."""
    lines = []
    lines.append(f"{'=' * 60}")
    lines.append(f"GPU Memory Stats: {label}")
    lines.append(f"{'=' * 60}")

    for device in jax.devices():
        try:
            stats = device.memory_stats()
            lines.append(f"Device: {device}")
            lines.append(f"  Backend: {device.platform}")
            lines.append(
                f"  Bytes in use:      {stats.get('bytes_in_use', 0) / (1024**3):.2f} GB"
            )
            lines.append(
                f"  Peak bytes in use: {stats.get('peak_bytes_in_use', 0) / (1024**3):.2f} GB"
            )
            if "bytes_limit" in stats:
                lines.append(
                    f"  Memory limit:      {stats['bytes_limit'] / (1024**3):.2f} GB"
                )
        except Exception as e:
            lines.append(f"Device: {device} - Could not get memory stats: {e}")
    lines.append(f"{'=' * 60}")

    return "\n".join(lines)


def print_array_device_info(array, name="Array"):
    """Print information about where a JAX array is located."""
    try:
        devices = array.devices()
        print(f"{name} located on: {devices}")
    except Exception as e:
        print(f"Could not determine device for {name}: {e}")


def save_results(results: dict, filename: str):
    """Save results dictionary to a JSON file."""

    dir_name = os.path.dirname(filename)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(filename, "w") as f:
        json.dump(results, f, indent=4)


def plot_metric_results(
    results: dict,
    filename: str,
    metrics: list[str],
):
    # create subplot with relative frobenius, relative spectral, cosine similarity, trace distance
    # for each dataset and each approximation method by parameter size
    # save the plot as pdf and png

    # choose fig size depending on number of metrics
    num_metrics = len(metrics)
    fig_size = (4 * num_metrics, 4)

    plt.figure(figsize=fig_size)

    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "figure.figsize": (12, 4),
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
    fig, axes = plt.subplots(1, num_metrics, figsize=fig_size)

    # Define colors for consistency
    colors = {"kfac": "#1f77b4", "ekfac": "#ff7f0e"}
    markers = {"kfac": "o", "ekfac": "s"}

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for approx_method in ["kfac", "ekfac"]:
            x = []
            y = []
            for dataset_name in results.keys():
                param_size = results[dataset_name][approx_method]["num_params"]
                metric_value = results[dataset_name][approx_method][metric]
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
                    results[dataset_name][approx_method][metric]
                    for dataset_name in results.keys()
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
    folder = os.path.dirname(filename)

    # create folder if it does not exist
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

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


def sample_gradient_from_output_distribution(
    model: ApproximationModel,
    params: Dict,
    x_train: Float[Array, "... n_features"],
    loss_fn,
    rng_key: PRNGKey | None = None,
):
    """Generate a single deterministic test vector."""
    if x_train.ndim == 1:
        x_sample = x_train
    else:
        x_sample = x_train[0]
    pseudo_target = KFACProvider.generate_pseudo_targets(
        model=model,
        params=params,
        inputs=x_sample,
        loss_fn=loss_fn,
        rng_key=rng_key,
    )
    grad = jax.jit(
        jax.grad(lambda p: loss_fn(model.apply(p, x_sample), pseudo_target))
    )(params)
    vec, _ = flatten_util.ravel_pytree(grad)
    return vec


def sample_gradient_from_output_distribution_batched(
    model_data: ModelContext, n_vectors: int = 5, rng_key: PRNGKey | None = None
):
    """Utility: Generate batched test vectors for IHVP tests."""
    loss_fn = model_data.loss
    x_train, _ = model_data.dataset.get_train_data()

    pseudo_targets = KFACProvider.generate_pseudo_targets(
        model=model_data.model,
        params=model_data.params,
        inputs=x_train[:n_vectors],
        loss_fn=loss_fn,
        rng_key=rng_key,
    )

    gradient_vecs = []
    for i in range(n_vectors):
        grad = jax.jit(
            jax.grad(
                lambda p: loss_fn(
                    model_data.model.apply(p, x_train[i]), pseudo_targets[i]
                )
            )
        )(model_data.params)
        gradient_vecs.append(flatten_util.ravel_pytree(grad)[0])

    return jnp.stack(gradient_vecs)
