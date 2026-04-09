from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.layer_matrix import LayerMatrix, LayerVector
from src.hessians.utils.data import (
    DataActivationsGradients,
    ModelContext,
)
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric

logger = logging.getLogger(__name__)


@dataclass
class HessianEstimator(ABC):
    """
    Base class for Hessian approximators.  Every subclass overrides `_build`
    to return a fully materialized `LayerMatrix`; the `estimate_*` public API
    is a thin wrapper over that matrix.
    """

    compute_context: ModelContext | DataActivationsGradients

    is_built: bool = False
    layer_matrix: Optional[LayerMatrix] = None
    layer_matrix_directory: Optional[str] = None

    # ---- build / load ----
    def build(
        self, base_directory: Optional[str] = None, try_load: bool = True
    ) -> "HessianEstimator":
        """
        Scaffolding to check if a built `LayerMatrix` already exists on disk, and if not, build it and save it.
        """
        if self.is_built:
            return self

        directory_path: Optional[str] = None
        if base_directory is not None:
            if self.layer_matrix_directory is None:
                directory_name = (
                    type(self).__name__.lower().replace("computer", "_layer_matrix")
                )
            else:
                directory_name = self.layer_matrix_directory
            directory_path = f"{base_directory}/{directory_name}"

        if (
            directory_path is not None
            and try_load
            and LayerMatrix.exists(directory_path)
        ):
            self.layer_matrix = LayerMatrix.load(directory_path)
            self.is_built = True
            logger.info(f"Loaded LayerMatrix from directory: {directory_path}")
        else:
            self.layer_matrix = self._build(compute_context=self.compute_context)
            if directory_path is not None and self.layer_matrix is not None:
                self.layer_matrix.save(directory=directory_path)
                logger.info(f"Saved LayerMatrix to directory: {directory_path}")
            self.is_built = True
        return self

    @abstractmethod
    def _build(
        self,
        compute_context: ModelContext | DataActivationsGradients,
    ) -> LayerMatrix:
        """Produce the `LayerMatrix` this estimator approximates."""

    # ---- estimate_* public API ----
    def _require_built(self, op: str) -> LayerMatrix:
        if not self.is_built or self.layer_matrix is None:
            raise RuntimeError(
                f"HessianEstimator not built. Please call `.build()` "
                f"before {op}."
            )
        return self.layer_matrix

    def estimate_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "n_params n_params"]:
        """Compute the full `(n_params, n_params)` Hessian approximation."""
        M = self._require_built("estimating the Hessian")
        d = 0.0 if damping is None else damping
        return M.damped(d).to_dense()

    def compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """Compare the estimated matrix against a reference matrix."""
        return metric.compute(comparison_matrix, self.estimate_hessian(damping))

    def estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute the Hessian-vector product `(H + dI) @ v` for each row of `vectors`."""
        M = self._require_built("estimating the Hessian-vector product")
        d = 0.0 if damping is None else damping
        lvec = LayerVector.from_flat(
            jnp.asarray(vectors),
            shapes=M.layer_shapes,
            param_groups=M.param_groups,
        )
        return (M.damped(d) @ lvec).to_flat()

    def estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
        pseudo_inverse_factor: Optional[float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute the inverse-Hessian-vector product `(H + dI)^{-1} @ v`."""
        if pseudo_inverse_factor is not None and damping is not None:
            raise ValueError(
                "Cannot use both damping and pseudo-inverse factor simultaneously."
            )
        M = self._require_built("estimating the inverse-Hessian-vector product")
        d = 0.0 if damping is None else damping
        p = 0.0 if pseudo_inverse_factor is None else pseudo_inverse_factor
        lvec = LayerVector.from_flat(
            jnp.asarray(vectors),
            shapes=M.layer_shapes,
            param_groups=M.param_groups,
        )
        return (
            M.inverse(damping=d, pseudo_inverse_factor=p) @ lvec
        ).to_flat()
