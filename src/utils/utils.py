import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import jax
import matplotlib.pyplot as plt
import numpy as np


def hash_data(data: Dict[str, Any], length: int = 10) -> str:
    canonical = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),  # canonical JSON
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]


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


def print_device_memory_stats(label=""):
    """Print current GPU memory statistics for all JAX devices."""
    print(f"\n{'=' * 60}")
    print(f"GPU Memory Stats: {label}")
    print(f"{'=' * 60}")

    for device in jax.devices():
        try:
            stats = device.memory_stats()
            print(f"Device: {device}")
            print(f"  Backend: {device.platform}")
            print(
                f"  Bytes in use:      {stats.get('bytes_in_use', 0) / (1024**3):.2f} GB"
            )
            print(
                f"  Peak bytes in use: {stats.get('peak_bytes_in_use', 0) / (1024**3):.2f} GB"
            )
            if "bytes_limit" in stats:
                print(f"  Memory limit:      {stats['bytes_limit'] / (1024**3):.2f} GB")
        except Exception as e:
            print(f"Device: {device} - Could not get memory stats: {e}")
    print(f"{'=' * 60}\n")


def get_peak_bytes_in_use():
    """Get peak bytes in use across all JAX devices."""
    total_peak = 0
    for device in jax.devices():
        try:
            stats = device.memory_stats()
            total_peak += stats.get("peak_bytes_in_use", 0)
        except Exception:
            pass
    return total_peak


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


def plot_metric_results_with_seeds(
    results: dict,
    filename: str,
    metrics: list[str],
):
    """
    Plot metrics with error bars showing standard deviation across seeds.

    Expected results structure:
    results[dataset_name][approx_method][metric] = {
        'mean': float,
        'std': float,
        'values': list
    }
    """
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

    # Handle case where there's only one metric
    if num_metrics == 1:
        axes = [axes]

    # Define colors for consistency
    colors = {
        "kfac": "#1f77b4",
        "ekfac": "#ff7f0e",
        "fim": "#2ca02c",
        "gauss_newton": "#d62728",
    }
    markers = {"kfac": "o", "ekfac": "s", "fim": "^", "gauss_newton": "d"}

    approx_methods = results[next(iter(results))].keys()
    for i, metric in enumerate(metrics):
        ax = axes[i]

        for approx_method in approx_methods:
            x = []
            y_mean = []
            y_std = []

            for dataset_name in results.keys():
                # Get number of parameters (should be the same across seeds)
                param_size = results[dataset_name][approx_method]["num_params"]["mean"]

                # Get mean and std for the metric
                metric_data = results[dataset_name][approx_method][metric]
                mean_value = metric_data["mean"]
                std_value = metric_data["std"]

                x.append(param_size)
                y_mean.append(mean_value)
                y_std.append(std_value)

            # Sort x, y_mean, and y_std based on x
            sorted_data = sorted(zip(x, y_mean, y_std))
            x, y_mean, y_std = zip(*sorted_data)

            label_dict = {
                "kfac": "K-FAC",
                "ekfac": "EK-FAC",
                "fim": "FIM",
                "gauss_newton": "Gauss-Newton",
            }

            label = label_dict.get(approx_method)

            # Plot line with error band
            ax.plot(
                x,
                y_mean,
                marker=markers[approx_method],
                label=label,
                color=colors[approx_method],
                linewidth=2,
                markersize=6,
                markerfacecolor=colors[approx_method],
                markeredgecolor="white",
                markeredgewidth=0.5,
            )

            # Add shaded error region (± 1 std)
            ax.fill_between(
                x,
                [m - s for m, s in zip(y_mean, y_std)],
                [m + s for m, s in zip(y_mean, y_std)],
                color=colors[approx_method],
                alpha=0.2,
            )

            # Optionally, add error bars instead of (or in addition to) shaded regions
            # Uncomment below to use error bars instead
            # ax.errorbar(
            #     x,
            #     y_mean,
            #     yerr=y_std,
            #     marker=markers[approx_method],
            #     label=label,
            #     color=colors[approx_method],
            #     linewidth=2,
            #     markersize=6,
            #     capsize=4,
            #     capthick=1.5,
            #     elinewidth=1.5,
            # )

        # Format metric name for LaTeX
        metric_latex = (
            metric.replace("_", " ")
            .replace("relative", "Relative")
            .replace("frobenius", "Frobenius")
            .replace("cosine", "Cosine")
            .replace("similarity", "Similarity")
            .replace("trace", "Trace")
            .replace("distance", "Distance")
            .replace("inner product diff", "Inner Product Difference")
        )

        ax.set_xlabel(r"Number of Parameters", fontsize=12)
        # ax.set_ylabel(metric_latex, fontsize=12)
        ax.set_title(metric_latex, fontsize=12, pad=10)

        # Improve grid
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Set scientific notation for y-axis if values are very small
        all_values = []
        for dataset_name in results.keys():
            for approx_method in approx_methods:
                all_values.append(results[dataset_name][approx_method][metric]["mean"])

        if max(all_values) < 0.01:
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
        r"Comparison to Ground Truth Hessian (Mean $\pm$ Std across seeds)",
        fontsize=14,
        y=1.02,
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
    plt.close()


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
    colors = {
        "kfac": "#1f77b4",
        "ekfac": "#ff7f0e",
        "fim": "#2ca02c",
        "gauss_newton": "#d62728",
    }
    markers = {"kfac": "o", "ekfac": "s", "fim": "^", "gauss_newton": "d"}

    for i, metric in enumerate(metrics):
        ax = axes[i]

        approx_methods = results[next(iter(results))].keys()

        for approx_method in approx_methods:
            x = []
            y = []
            for dataset_name in results.keys():
                param_size = results[dataset_name][approx_method]["num_params"]
                metric_value = results[dataset_name][approx_method][metric]
                x.append(param_size)
                y.append(metric_value)

            # Sort x and y based on x
            x, y = zip(*sorted(zip(x, y)))

            label_dict = {
                "kfac": "K-FAC",
                "ekfac": "EK-FAC",
                "fim": "FIM",
                "gauss_newton": "Gauss-Newton",
            }

            label = label_dict.get(approx_method)
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
                    for approx_method in ["kfac", "ekfac", "fim", "gauss_newton"]
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

    plt.suptitle(r"Comparison to Ground Truth Hessian", fontsize=14, y=1.02)
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


def write_markdown_report(
    folder: str,
    aggregated_results: Dict[str, Any],
    all_seeds_results: Dict[str, Any],
    timestamp: str,
    title: str,
):
    """
    Create a detailed Markdown report summarizing a Hessian comparison run.

    Args:
        folder (str): Output directory where results were written.
        aggregated_results (dict): Mean/std aggregated metrics.
        all_seeds_results (dict): Raw results per seed.
        plot_prefix (str): Prefix used when generating plot filenames.
        timestamp (str): Timestamp of the run.
    """

    md_path = Path(folder) / f"report_{timestamp}.md"

    with open(md_path, "w") as f:
        f.write(f"# {title}\n")
        f.write(f"**Timestamp:** `{timestamp}`  \n")
        f.write(f"**Output Folder:** `{folder}`\n\n")

        # -----------------------------
        # 1. Summary of experimental parameters
        # -----------------------------
        f.write("## Experiment Summary\n")
        f.write(f"- Number of seeds: **{len(all_seeds_results)}**\n")
        f.write(f"- Seeds used: `{list(all_seeds_results.keys())}`\n\n")

        # -----------------------------
        # 2. Embedded Plots
        # -----------------------------
        f.write("## Plots\n")
        plot_files = [p for p in os.listdir(folder) if p.endswith(".png")]
        if plot_files:
            for plt_name in plot_files:
                f.write(f"### {plt_name}\n")
                f.write(f"![{plt_name}]({plt_name})\n\n")
        else:
            f.write("_No plots found._\n\n")

        # -----------------------------
        # 3. JSON File Links
        # -----------------------------
        f.write("## Output JSON Files\n")
        json_files = [p for p in os.listdir(folder) if p.endswith(".json")]
        for js in json_files:
            f.write(f"- [{js}]({js})\n")
        f.write("\n")

        # -----------------------------
        # 4. Aggregated results tables
        # -----------------------------
        f.write("## Aggregated Metric Results (Mean ± Std)\n")

        for dataset_name, approx_data in aggregated_results.items():
            f.write(f"### Dataset: `{dataset_name}`\n\n")

            for approx_method, metrics in approx_data.items():
                f.write(f"#### Approximation: `{approx_method}`\n\n")
                f.write("| Metric | Mean | Std |\n")
                f.write("|--------|------|-----|\n")

                for metric_name, metric_stats in metrics.items():
                    mean = metric_stats["mean"]
                    std = metric_stats["std"]
                    f.write(f"| {metric_name} | {mean:.6f} | {std:.6f} |\n")

                f.write("\n")

        # -----------------------------
        # 5. Peak memory & runtime per dataset
        # -----------------------------
        f.write("## Runtime and Memory Summary\n")
        first_seed = list(all_seeds_results.keys())[0]

        for dataset_name, dct in all_seeds_results[first_seed].items():
            time_taken = dct.get("time_taken_overall", "N/A")
            peak_mem = dct.get("peak_memory", "N/A")
            f.write(
                f"- `{dataset_name}`: **Runtime:** {time_taken}, **Peak GPU mem:** {peak_mem}\n"
            )

        f.write("\n---\n")
        f.write("Report generated automatically.\n")

    print(f"Markdown report written to: {md_path}")


def write_global_markdown_summary(
    all_collections_results: Dict[str, Dict[str, Any]],
    timestamp: str,
    title: str,
    folder: str,
    overview_metric: str,
):
    summary_root = Path(f"{folder}")
    summary_root.mkdir(parents=True, exist_ok=True)

    md_path = summary_root / f"markdown_report_{timestamp}.md"

    with open(md_path, "w") as f:
        f.write(f"# {title}\n")
        f.write(f"**Timestamp:** `{timestamp}`\n\n")
        f.write("This report summarizes **all experimental runs**.\n\n")

        # ================================================================
        # Iterate through each collection: single_linear, multi_linear, mlp
        # ================================================================
        for collection_name, entry in all_collections_results.items():
            folder = entry["folder"]
            aggregated_results = entry["aggregated"]
            all_seeds_results = entry["all_seeds"]

            f.write("\n---\n")
            f.write(f"## Collection: **{collection_name}**\n")
            f.write(f"Folder: `{folder}`\n\n")

            # ============================================================
            # Compact Overview Table: mean ± std for all approximations
            # ============================================================
            metric_name = overview_metric  # e.g., "l2_norm_difference"

            f.write(
                f"### Overview Table (Mean ± Std for {metric_name} for all methods)\n"
            )
            f.write("| Dataset | #Params | KFAC | EKFAC | FIM | Gauss-Newton |\n")
            f.write("|---------|---------|------|--------|-----|--------------|\n")

            for dataset_name, approx_data in aggregated_results.items():
                # Extract parameter count (same for all methods)
                any_method = next(iter(approx_data.values()))
                num_params = int(any_method["num_params"]["mean"])

                # Collect "mean ± std" for each method
                row_values = []
                for method in ["kfac", "ekfac", "fim", "gauss_newton"]:
                    if method in approx_data and metric_name in approx_data[method]:
                        mean = approx_data[method][metric_name]["mean"]
                        std = approx_data[method][metric_name]["std"]
                        row_values.append(f"{mean:.3f} ± {std:.3f}")
                    else:
                        row_values.append("N/A")

                # Write row
                f.write(
                    f"| `{dataset_name}` | {num_params} | "
                    f"{row_values[0]} | {row_values[1]} | {row_values[2]} | {row_values[3]} |\n"
                )

            f.write("\n")

            # ============================================================
            # Plots (unchanged but inside collapsible box)
            # ============================================================
            plot_files = sorted([p for p in os.listdir(folder) if p.endswith(".png")])
            if plot_files:
                for plt_name in plot_files:
                    f.write(f"#### {plt_name}\n")
                    f.write(f"![{plt_name}](./{collection_name}/{plt_name})\n\n")
            else:
                f.write("_No plots found._\n\n")

            # ============================================================
            # JSON Files (short section)
            # ============================================================
            f.write("### JSON Files\n")
            for js in sorted([p for p in os.listdir(folder) if p.endswith(".json")]):
                f.write(f"- [{js}](./{collection_name}/{js})\n")
            f.write("\n")

            # ============================================================
            # Aggregated results — now collapsible per dataset
            # ============================================================
            f.write(
                "<details>\n<summary><strong>Aggregated Metric Results</strong></summary>\n\n"
            )
            methods_order = ["kfac", "ekfac", "fim", "gauss_newton"]

            for dataset_name, approx_data in aggregated_results.items():
                f.write(f"\n### Dataset: `{dataset_name}`\n\n")

                # Table header
                f.write("| Metric | KFAC | EKFAC | FIM | Gauss–Newton |\n")
                f.write("|--------|------|--------|------|--------------|\n")

                # Collect all metric names
                metric_names = list(next(iter(approx_data.values())).keys())

                for metric in metric_names:
                    row = f"| {metric} "

                    for method in methods_order:
                        if method in approx_data and metric in approx_data[method]:
                            mean = approx_data[method][metric]["mean"]
                            std = approx_data[method][metric]["std"]
                            row += f"| {mean:.3f} ± {std:.3f} "
                        else:
                            row += "| N/A "
                    row += "|\n"
                    f.write(row)

                f.write("\n")

            f.write("</details>\n\n")

            # ============================================================
            # Runtime summary
            # ============================================================
            f.write("### Runtime & Memory Summary\n")
            first_seed = list(all_seeds_results.keys())[0]
            for dataset_name, dct in all_seeds_results[first_seed].items():
                t = dct.get("time_taken_overall", "N/A")
                mem = dct.get("peak_memory", "N/A")
                f.write(
                    f"- `{dataset_name}` → Runtime: **{t}**, Peak GPU mem: **{mem}**\n"
                )
            f.write("\n")

        f.write("\n---\nReport generated automatically.\n")

    print(f"Global Markdown summary written to: {md_path}")
