from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

from jaxtyping import Array, Float

from src.hessians.computer.computer import CollectorBasedHessianEstimator
from src.hessians.utils.data import DataActivationsGradients, ETKFACData
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric

logger = logging.getLogger(__name__)


@dataclass
class ETKFACComputer(CollectorBasedHessianEstimator):
    """
    Kronecker-Factored Approximate Curvature (KFAC) and Eigenvalue-Corrected KFAC (EKFAC) Hessian approximation.
    """

    precomputed_data: ETKFACData = field(default_factory=ETKFACData)

    @staticmethod
    def _build(
        compute_context: Tuple[DataActivationsGradients, DataActivationsGradients],
    ) -> ETKFACData:
        """Method to build the required TK-FAC components to compute the Hessian approximation."""
        raise NotImplementedError("ETKFAC build method not implemented yet.")

    def _estimate_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute full Hessian approximation.
        """
        raise NotImplementedError("ETKFAC Hessian estimation not implemented yet.")

    def _estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Hessian-vector product."""

        assert self.precomputed_data is not None, (
            "ETKFAC data not computed. Please build the computer first."
        )
        raise NotImplementedError("ETKFAC HVP estimation not implemented yet.")

    def _compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> float:
        """
        Compare the ETKFAC Hessian approximation to a given comparison matrix.
        Reuses the KFAC comparison implementation.
        """
        assert self.precomputed_data is not None, (
            "ETKFAC data not computed. Please build the computer first."
        )
        raise NotImplementedError("ETKFAC Hessian comparison not implemented yet.")

    def _estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute inverse Hessian-vector product.
        """
        assert self.precomputed_data is not None, (
            "ETKFAC data not computed. Please build the computer first."
        )
        raise NotImplementedError("ETKFAC IHVP estimation not implemented yet.")
