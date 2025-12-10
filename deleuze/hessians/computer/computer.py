from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from jaxtyping import Array, Float

from deleuze.metrics.full_matrix_metrics import FullMatrixMetric


@dataclass
class HessianComputer(ABC):
    compute_context: Any

    @abstractmethod
    def compute_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "n_params n_params"]:
        """Compute Hessian approximation."""
        pass

    @abstractmethod
    def compare_hessians(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """Compare Hessian approximation with another Hessian matrix."""
        pass

    @abstractmethod
    def compute_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Hessian-vector product."""
        pass

    @abstractmethod
    def compute_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Inverse Hessian-vector product."""
        pass
