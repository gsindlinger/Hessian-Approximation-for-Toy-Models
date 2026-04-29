"""LDS analysis pipeline — split into two clearly separated steps.

  ``attribute``  Compute a (n_query, n_train) influence matrix for a given
                 (model, epoch, method) and save it as ``.npy`` on disk.
                 This is the *expensive* step that needs the trained model,
                 the Hessian approximator, and per-example gradients.

  ``lds``        Load a saved attribution matrix, retrain the model on K
                 random complements (ELSO), and report the Spearman rank
                 correlation between predicted and observed group effects.
                 Driven entirely by ``LDSConfig`` and the .npy file path.

The two steps are deliberately decoupled so a single ELSO retraining cache
can be reused across damping / approximator / method sweeps.

Examples:

    # 1) Compute attributions for one (model, epoch, method) tuple.
    python -m experiments.lds_analysis attribute \\
        --model experiments/models/.../mlp_h32 \\
        --epoch 100 \\
        --method ekfac \\
        --num-test-examples 50 \\
        --output experiments/attributions/mlp_h32_e100_ekfac.npy

    # 2) Score the saved attributions against ELSO.
    python -m experiments.lds_analysis lds \\
        --model experiments/models/.../mlp_h32 \\
        --epoch 100 \\
        --attribution-scores experiments/attributions/mlp_h32_e100_ekfac.npy \\
        --num-subsets 100 --reps-per-model 1 --subset-fraction 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
from jax.random import PRNGKey

from src.config import (
    HessianApproximationMethod,
    LDSConfig,
    PseudoTargetGenerationStrategy,
    RegularizationStrategy,
)
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.registry import HessianComputerRegistry
from src.hessians.utils.data import ModelContext
from src.utils.influence import compute_influence_matrix, compute_per_example_flat_grads
from src.utils.lds import compute_lds
from src.utils.loss import get_loss
from src.utils.train import (
    load_model_checkpoint,
    load_train_test_for_model,
    resolve_regularization,
)
from src.utils.utils import collector_cache_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attribute step: compute and save a (n_query, n_train) influence matrix
# ---------------------------------------------------------------------------


def attribute(
    model_directory: str,
    method: HessianApproximationMethod,
    output_path: str,
    epoch: Optional[int],
    num_test_examples: int,
    pseudo_target_strategy: PseudoTargetGenerationStrategy,
    pseudo_target_repetitions: int,
    regularization_strategy: RegularizationStrategy,
    regularization_value: float,
    seed: int,
) -> str:
    """Compute the (n_query, n_train) influence matrix and write it to disk.

    Returns:
        The output path the .npy file was written to.
    """
    params, model, model_config, _ = load_model_checkpoint(model_directory, epoch=epoch)
    loss_fn = get_loss(model_config.loss)

    train_dataset, test_dataset = load_train_test_for_model(model_directory)
    n_query = min(num_test_examples, len(test_dataset.inputs))
    query_inputs = test_dataset.inputs[:n_query]
    query_targets = test_dataset.targets[:n_query]

    epoch_str = f"epoch_{epoch}" if epoch is not None else "final"
    logger.info(
        "[ATTRIBUTE] %s (%s) method=%s",
        model_config.get_model_display_name(),
        epoch_str,
        method.value,
    )

    # Collector — shared cache key so EKFAC + the chosen approximator hit the
    # same on-disk activations/gradients.
    collector_dir = collector_cache_dir(
        model_directory=model_directory,
        pseudo_target_strategy=pseudo_target_strategy,
        pseudo_target_repetitions=pseudo_target_repetitions,
        epoch=epoch,
        seed=seed,
    )
    collector = CollectorActivationsGradients(
        model=model,
        params=params,
        loss_fn=loss_fn,
        pseudo_target_repetitions=pseudo_target_repetitions,
        pseudo_target_strategy=pseudo_target_strategy,
    )
    collector_data = collector.collect(
        dataset=train_dataset,
        save_directory=collector_dir,
        try_load=True,
        rng_key=PRNGKey(seed),
    )

    # For MCMC, fit Λ on an independent collector run (different rng key) so
    # the eigenvalue correction isn't overfit on the samples used for Q_A/Q_G.
    # Deterministic strategies (EMPIRICAL_FISHER, ALL_CLASSES) reuse the same
    # data — there is no sampling noise to decorrelate.
    if pseudo_target_strategy == PseudoTargetGenerationStrategy.MCMC:
        collector_dir_corr = collector_cache_dir(
            model_directory=model_directory,
            pseudo_target_strategy=pseudo_target_strategy,
            pseudo_target_repetitions=pseudo_target_repetitions,
            epoch=epoch,
            seed=seed + 1,
        )
        collector_data_corr = collector.collect(
            dataset=train_dataset,
            save_directory=collector_dir_corr,
            try_load=True,
            rng_key=PRNGKey(seed + 1),
        )
    else:
        collector_data_corr = collector_data

    # Damping: derived from EKFAC for AUTO_* strategies, identity for FIXED /
    # PSEUDO_INVERSE.
    if regularization_strategy in (
        RegularizationStrategy.AUTO_MEAN_EIGENVALUE,
        RegularizationStrategy.AUTO_MEAN_EIGENVALUE_CORRECTION,
    ):
        ekfac = EKFACComputer(
            compute_context=collector_data,
            corr_context=collector_data_corr,
        ).build(base_directory=collector_dir)
        damping, pseudo_inverse_factor = resolve_regularization(
            strategy=regularization_strategy,
            factor=regularization_value,
            built_computer=ekfac,
        )
    else:
        damping, pseudo_inverse_factor = resolve_regularization(
            strategy=regularization_strategy,
            factor=regularization_value,
        )
    logger.info(
        "[ATTRIBUTE] damping=%s, pseudo_inverse_factor=%s",
        damping,
        pseudo_inverse_factor,
    )

    # Build the chosen approximator.
    model_ctx = ModelContext.create(
        dataset=train_dataset, model=model, params=params, loss_fn=loss_fn
    )
    compute_ctx = HessianComputerRegistry.get_compute_context(
        approximator=method,
        collector_data=collector_data,
        model_ctx=model_ctx,
    )
    computer = HessianComputerRegistry.get_computer(
        method, compute_ctx, corr_context=collector_data_corr
    )
    if not isinstance(computer, (HessianEstimator, HessianComputer)):
        raise ValueError(f"Method {method.value} has no IHVP support.")
    if isinstance(computer, HessianEstimator):
        computer.build(base_directory=collector_dir)

    # Per-example gradients on the full training set + queries.
    train_grads = compute_per_example_flat_grads(
        model=model,
        params=params,
        inputs=train_dataset.inputs,
        targets=train_dataset.targets,
        loss_fn=loss_fn,
    )
    test_grads = compute_per_example_flat_grads(
        model=model,
        params=params,
        inputs=query_inputs,
        targets=query_targets,
        loss_fn=loss_fn,
    )

    attributions = compute_influence_matrix(
        test_flat_grads=test_grads,
        train_flat_grads=train_grads,
        computer=computer,
        damping=damping,
        pseudo_inverse_factor=pseudo_inverse_factor,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, np.asarray(attributions))
    logger.info(
        "[ATTRIBUTE] saved attribution matrix shape=%s to %s",
        attributions.shape,
        output_path,
    )
    return output_path


# ---------------------------------------------------------------------------
# LDS step: load attributions + retrain on complements + score
# ---------------------------------------------------------------------------


def lds(config: LDSConfig, output_path: Optional[str] = None) -> dict:
    """Run ELSO retraining and score the saved attribution matrix.

    The train/test split is resolved from ``model.json`` metadata inside
    ``compute_lds``; this wrapper only handles JSON-serialisation of the
    result dict.
    """
    result = compute_lds(config)
    result["lds_config"] = asdict(config)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("[LDS] wrote results → %s", output_path)

    s = result["lds_scores"]
    logger.info("=" * 70)
    logger.info(
        "[LDS] mean_LDS=%.4f ± %.4f  (95%% CI [%.4f, %.4f])  valid_queries=%d/%d",
        s["mean_lds"],
        s["std_lds"],
        s["ci_low"],
        s["ci_high"],
        s["num_valid_queries"],
        s["num_valid_queries"] + s["num_undefined_queries"],
    )
    logger.info("=" * 70)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LDS analysis: compute attributions, then score them via ELSO."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ----- attribute -----
    p_attr = sub.add_parser(
        "attribute",
        help="Compute (n_query, n_train) influence scores and save to .npy.",
    )
    p_attr.add_argument("--model", required=True, help="Model directory.")
    p_attr.add_argument(
        "--method",
        required=True,
        choices=[m.value for m in HessianApproximationMethod],
        help="Hessian approximator.",
    )
    p_attr.add_argument(
        "--output", required=True, help="Output .npy path for the attribution matrix."
    )
    p_attr.add_argument("--epoch", type=int, default=None)
    p_attr.add_argument("--num-test-examples", type=int, default=50)
    p_attr.add_argument(
        "--pseudo-target-strategy",
        default=PseudoTargetGenerationStrategy.MCMC.value,
        choices=[s.value for s in PseudoTargetGenerationStrategy],
    )
    p_attr.add_argument("--pseudo-target-repetitions", type=int, default=5)
    p_attr.add_argument(
        "--regularization-strategy",
        default=RegularizationStrategy.AUTO_MEAN_EIGENVALUE.value,
        choices=[s.value for s in RegularizationStrategy],
    )
    p_attr.add_argument("--regularization-value", type=float, default=0.1)
    p_attr.add_argument("--seed", type=int, default=42)

    # ----- lds -----
    p_lds = sub.add_parser(
        "lds",
        help="Score a saved attribution matrix against ELSO retraining.",
    )
    p_lds.add_argument("--model", required=True, help="Model directory.")
    p_lds.add_argument(
        "--attribution-scores",
        required=True,
        help="Path to .npy attribution matrix produced by `attribute`.",
    )
    p_lds.add_argument("--epoch", type=int, default=None)
    p_lds.add_argument("--num-subsets", type=int, default=100)
    p_lds.add_argument("--reps-per-model", type=int, default=1)
    p_lds.add_argument("--subset-fraction", type=float, default=0.5)
    p_lds.add_argument("--num-test-examples", type=int, default=50)
    p_lds.add_argument("--seed", type=int, default=42)
    p_lds.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the rep_mean ELSO cache.",
    )
    p_lds.add_argument(
        "--output",
        default=None,
        help="Optional path to write a JSON summary of the LDS run.",
    )

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()

    if args.command == "attribute":
        attribute(
            model_directory=args.model,
            method=HessianApproximationMethod(args.method),
            output_path=args.output,
            epoch=args.epoch,
            num_test_examples=args.num_test_examples,
            pseudo_target_strategy=PseudoTargetGenerationStrategy(
                args.pseudo_target_strategy
            ),
            pseudo_target_repetitions=args.pseudo_target_repetitions,
            regularization_strategy=RegularizationStrategy(
                args.regularization_strategy
            ),
            regularization_value=args.regularization_value,
            seed=args.seed,
        )
        return

    if args.command == "lds":
        config = LDSConfig(
            model=args.model,
            attribution_scores=args.attribution_scores,
            epoch=args.epoch,
            num_subsets=args.num_subsets,
            reps_per_model=args.reps_per_model,
            subset_fraction=args.subset_fraction,
            num_test_examples=args.num_test_examples,
            lds_seed=args.seed,
            cache_elso=not args.no_cache,
        )
        output_path = args.output
        if output_path is None:
            ts = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(args.model, "lds_results", f"{ts}.json")
        lds(config=config, output_path=output_path)
        return


if __name__ == "__main__":
    main()
