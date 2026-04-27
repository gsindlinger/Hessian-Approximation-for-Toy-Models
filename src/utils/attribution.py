"""Attribution-score generation for the LDS pipeline.

Computes per-(query, train) influence matrices τ(z_q, z_i) for one or more
Hessian approximators and saves them as one ``.npy`` file per method inside
a per-(model, epoch) directory. The LDS analysis loads these directories
directly (memory-mapped) and merges across multiple source dirs by method.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np
from jax.random import PRNGKey

from src.config import HessianEstimatorsConfig, LossType, PseudoTargetGenerationStrategy
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.registry import HessianComputerRegistry
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.utils.data.data import Dataset
from src.utils.influence import compute_influence_matrix, compute_per_example_flat_grads
from src.utils.loss import get_loss
from src.utils.train import load_model_checkpoint
from src.utils.utils import cleanup_memory, collector_cache_dir, resolve_regularization

logger = logging.getLogger(__name__)


def compute_attribution_scores_for_model(
    model_directory: str,
    train_dataset: Dataset,
    query_inputs,
    query_targets,
    hessian_config: HessianEstimatorsConfig,
    seed: int,
    epoch: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Compute influence matrices for every approximator in ``hessian_config``.

    Returns a dict ``{method_name: (n_test, n_train) matrix}``.
    """

    params, model, model_config, _ = load_model_checkpoint(model_directory, epoch=epoch)
    loss_fn = get_loss(model_config.loss)

    if model_config.loss == LossType.MSE:
        train_inputs, query_inputs = Dataset.normalize_data(
            train_dataset.inputs, query_inputs
        )
        train_targets, query_targets = Dataset.normalize_data(
            train_dataset.targets, query_targets
        )
        assert query_inputs is not None and query_targets is not None
        train_dataset = Dataset(train_inputs, train_targets)

    collector_dir_base = collector_cache_dir(
        model_directory=model_directory,
        pseudo_target_strategy=hessian_config.pseudo_target_generation_strategy.value,
        pseudo_target_repetitions=hessian_config.pseudo_target_generation_repetitions,
        epoch=epoch,
    )

    def _make_collector() -> CollectorActivationsGradients:
        return CollectorActivationsGradients(
            model=model,
            params=params,
            loss_fn=loss_fn,
            pseudo_target_repetitions=hessian_config.pseudo_target_generation_repetitions,
            pseudo_target_strategy=hessian_config.pseudo_target_generation_strategy,
        )

    collector_data: DataActivationsGradients = _make_collector().collect(
        dataset=train_dataset,
        save_directory=collector_dir_base,
        try_load=True,
        rng_key=PRNGKey(seed),
    )

    # Corrected Kronecker methods (for example EKFAC) need a second collector
    # context for the eigenvalue correction pass. Deterministic strategies can
    # reuse the same data; MCMC must use an independent draw to avoid fitting
    # the correction on the same samples as the eigenspaces.
    if (
        hessian_config.pseudo_target_generation_strategy
        == PseudoTargetGenerationStrategy.MCMC
    ):
        collector_data_corr: DataActivationsGradients = _make_collector().collect(
            dataset=train_dataset,
            save_directory=f"{collector_dir_base}_corr",
            try_load=True,
            rng_key=PRNGKey(seed + 1),
        )
    else:
        collector_data_corr = collector_data

    ekfac_computer = EKFACComputer(
        compute_context=collector_data, corr_context=collector_data_corr
    ).build(base_directory=collector_dir_base)
    damping, pseudo_inverse_factor = resolve_regularization(
        strategy=hessian_config.regularization_strategy,
        factor=hessian_config.regularization_value,
        built_computer=ekfac_computer,
    )
    logger.info(
        "[ATTR] Regularization: damping=%s, pseudo_inverse_factor=%s",
        damping,
        pseudo_inverse_factor,
    )

    model_ctx = ModelContext.create(
        dataset=train_dataset,
        model=model,
        params=params,
        loss_fn=loss_fn,
    )

    logger.info(
        "[ATTR] Computing per-example gradients (%d train, %d queries).",
        len(train_dataset.inputs),
        len(query_inputs),
    )
    train_flat_grads = compute_per_example_flat_grads(
        model=model,
        params=params,
        inputs=train_dataset.inputs,
        targets=train_dataset.targets,
        loss_fn=loss_fn,
    )
    test_flat_grads = compute_per_example_flat_grads(
        model=model,
        params=params,
        inputs=query_inputs,
        targets=query_targets,
        loss_fn=loss_fn,
    )
    cleanup_memory("gradient_computation")

    attributions: Dict[str, np.ndarray] = {}
    for approx in hessian_config.approximators:
        logger.info("[ATTR] Attribution method: %s", approx.value)

        compute_ctx = HessianComputerRegistry.get_compute_context(
            approximator=approx,
            collector_data=collector_data,
            model_ctx=model_ctx,
        )
        computer = HessianComputerRegistry.get_computer(
            approx, compute_ctx, corr_context=collector_data_corr
        )

        if not isinstance(computer, (HessianEstimator, HessianComputer)):
            logger.warning("Skipping %s: no IHVP support.", approx.value)
            continue

        if isinstance(computer, HessianEstimator):
            computer.build(base_directory=collector_dir_base)

        matrix = compute_influence_matrix(
            test_flat_grads=test_flat_grads,
            train_flat_grads=train_flat_grads,
            estimator=computer,
            damping=damping,
            pseudo_inverse_factor=pseudo_inverse_factor,
        )
        attributions[approx.value] = np.asarray(matrix)
        cleanup_memory(f"attribution_{approx.value}")

    return attributions


def save_attribution_scores(
    attributions: Dict[str, np.ndarray],
    output_dir: str,
) -> None:
    """Save attribution matrices as one ``.npy`` per method under ``output_dir``.

    Each matrix lands at ``{output_dir}/{method}.npy`` and can be loaded via
    ``src.utils.lds.load_attribution_scores``. Per-method files keep memory
    bounded for large models (no zip handle holding every matrix at once) and
    let the loader memory-map each matrix independently.
    """
    os.makedirs(output_dir, exist_ok=True)
    for method, matrix in attributions.items():
        np.save(os.path.join(output_dir, f"{method}.npy"), matrix)
    logger.info(
        "[ATTR] Saved %d matrices to %s (methods: %s).",
        len(attributions),
        output_dir,
        sorted(attributions.keys()),
    )
