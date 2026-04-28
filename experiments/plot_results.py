"""
Plot Hessian analysis results from JSON output.

Produces one tall PNG per (model, epoch) pair. Each metric gets its own row
as a bar plot (one bar per approximator, log-scale y-axis). Rows are ordered
by category: Matrix, HVP, IHVP, Round-trip.

Usage:
    python -m experiments.plot_results <results.json> [--output-dir DIR] [--reference exact]
                                                      [--approxs A,B,C]

--approxs is a comma-separated list of approximator names; it both filters
(only listed ones are plotted) and orders (list order = bar order). When
omitted, all approximators present in the JSON are plotted in APPROX_ORDER,
with unknown names appended alphabetically.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# ── Style ────────────────────────────────────────────────────────────────────

RCPARAMS = {
    "font.size": 11,
}

COLORS = {
    "kfac": "#1f77b4",
    "ekfac": "#ff7f0e",
    "gnh": "#d62728",
    "fim": "#2ca02c",
    "block_fim": "#9467bd",
    "block_hessian": "#8c564b",
    "shampoo": "#e377c2",
    "eshampoo": "#7f7f7f",
    "identity": "#17becf",
}

LABELS = {
    "kfac": "K-FAC",
    "ekfac": "EK-FAC",
    "gnh": "GNH",
    "fim": "FIM",
    "block_fim": "Block FIM",
    "block_hessian": "Block Hessian",
    "shampoo": "Shampoo",
    "eshampoo": "E-Shampoo",
    "identity": "Identity",
}

APPROX_ORDER = [
    "gnh",
    "fim",
    "block_hessian",
    "block_fim",
    "shampoo",
    "eshampoo",
    "ekfac",
    "kfac",
]

CATEGORY_DISPLAY = {
    "matrix": "Matrix",
    "hvp": "HVP",
    "ihvp": "IHVP",
    "roundtrip": "Round-trip",
}


def _format_metric_name(metric: str) -> str:
    return (
        metric.replace("_", " ")
        .title()
        .replace("L2", r"$L_2$")
        .replace("Hvp", "HVP")
        .replace("Ihvp", "IHVP")
    )


# ── Data reshaping ───────────────────────────────────────────────────────────


def _build_epoch_data(data: dict, reference: str):
    epoch_data = defaultdict(
        lambda: defaultdict(
            lambda: {
                "matrix": defaultdict(dict),
                "hvp": defaultdict(dict),
                "ihvp": defaultdict(dict),
                "roundtrip": {},
            }
        )
    )
    model_params = {}

    for r in data["results"]:
        name = r["model_name"]
        epoch = r["epoch"]
        model_params[name] = r["num_parameters"]
        ha = r["hessian_analysis"]
        entry = epoch_data[epoch][name]

        for metric, ref_dict in ha["matrix_comparisons"].items():
            if reference in ref_dict:
                entry["matrix"][metric] = ref_dict[reference]

        for metric, ref_dict in ha["hvp_comparisons"].items():
            if reference in ref_dict:
                entry["hvp"][metric.lower()] = ref_dict[reference]

        for metric, ref_dict in ha["ihvp_comparisons"].items():
            if reference in ref_dict:
                entry["ihvp"][metric.lower()] = ref_dict[reference]

        rt = ha.get("ihvp_round_trip_approximation_errors", {})
        if reference in rt:
            entry["roundtrip"] = rt[reference]

    return epoch_data, model_params


# ── Figure builder ───────────────────────────────────────────────────────────


def _build_model_figure(
    model_name,
    num_params,
    epoch,
    model_data,
    approxs,
    matrix_metrics,
    vector_metrics,
    ref_label,
):
    rows = []
    for m in matrix_metrics:
        rows.append(("matrix", m))
    for m in vector_metrics:
        rows.append(("hvp", m))
    for m in vector_metrics:
        rows.append(("ihvp", m))
    rows.append(("roundtrip", "round_trip_error"))

    n_rows = len(rows)
    n_approx = len(approxs)
    x = np.arange(n_approx)

    fig_width = max(10, n_approx * 1.1 + 2)
    row_height = 2.4
    total_height = row_height * n_rows + 1.2

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(fig_width, total_height),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    for ax, (cat_key, metric) in zip(axes, rows):
        if cat_key == "roundtrip":
            raw = {a: model_data["roundtrip"].get(a) for a in approxs}
        else:
            metric_dict = model_data[cat_key].get(metric, {})
            raw = {a: metric_dict.get(a) for a in approxs}

        plot_vals = [
            raw[a] if raw[a] is not None and raw[a] > 0 else np.nan for a in approxs
        ]

        ax.bar(
            x,
            plot_vals,
            color=[COLORS.get(a, f"C{i}") for i, a in enumerate(approxs)],
            alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [LABELS.get(a, a) for a in approxs],
            rotation=35,
            ha="right",
        )

        if any(np.isfinite(v) for v in plot_vals):
            ax.set_yscale("log")

        cat_display = CATEGORY_DISPLAY[cat_key]
        metric_display = _format_metric_name(metric)
        ax.set_title(f"{cat_display} — {metric_display}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{model_name}  ({num_params} params)  \u2014  Epoch {epoch}  (vs {ref_label})",
        fontsize=12,
        fontweight="bold",
    )
    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Plot Hessian analysis results.")
    parser.add_argument("results_json", help="Path to the results JSON file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots (default: same directory as input JSON).",
    )
    parser.add_argument(
        "--reference",
        default="exact",
        help="Reference method for comparisons (default: exact).",
    )
    parser.add_argument(
        "--approxs",
        default=None,
        help="Comma-separated approximators to include, in plot order "
        "(default: all found, ordered by APPROX_ORDER).",
    )
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)

    output_dir = args.output_dir or os.path.dirname(args.results_json)
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update(RCPARAMS)

    epoch_data, model_params = _build_epoch_data(data, args.reference)

    matrix_metrics = data["hessian_config"]["matrix_config"]["metrics"]
    vector_metrics = [
        m.lower() for m in data["hessian_config"]["vector_config"]["metrics"]
    ]

    ref_label = args.reference.replace("_", " ").title()
    ts = "s"

    all_approxs = set()
    for epoch_models in epoch_data.values():
        for model_d in epoch_models.values():
            for cat in ("matrix", "hvp", "ihvp"):
                for metric_dict in model_d[cat].values():
                    all_approxs.update(metric_dict.keys())
            all_approxs.update(model_d["roundtrip"].keys())

    if args.approxs:
        approxs = [a.strip() for a in args.approxs.split(",") if a.strip()]
    else:
        rank = {a: i for i, a in enumerate(APPROX_ORDER)}
        approxs = sorted(all_approxs, key=lambda a: (rank.get(a, len(rank)), a))

    print(f"Generating plots (reference={args.reference})...")

    for epoch in sorted(epoch_data.keys()):
        models = sorted(epoch_data[epoch].keys(), key=lambda m: model_params[m])
        for model in models:
            fig = _build_model_figure(
                model,
                model_params[model],
                epoch,
                epoch_data[epoch][model],
                approxs,
                matrix_metrics,
                vector_metrics,
                ref_label,
            )
            safe_name = model.replace(" ", "_").replace("/", "_")
            out_path = os.path.join(output_dir, f"{safe_name}_epoch{epoch}_{ts}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
            plt.close(fig)
            print(f"  Saved {out_path}")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
