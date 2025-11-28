from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from jaxtyping import Array, Float

from ..config.config import Config
from ..models.dataclasses.model_context import ModelContext
from ..models.train import train_or_load


@dataclass
class HessianApproximation(ABC):
    """Abstract base class for Hessian approximations."""

    full_config: Config
    model_context: ModelContext = field(init=False)

    def __post_init__(self):
        self.model_context = train_or_load(self.full_config)

    @abstractmethod
    def compute_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "num_params num_params"]:
        """Compute Hessian approximation."""
        pass

    @abstractmethod
    def compute_hvp(
        self,
        vectors: Float[Array, "*batch_size num_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size num_params"]:
        """Compute Hessian-vector product."""
        pass

    @abstractmethod
    def compute_ihvp(
        self,
        vectors: Float[Array, "*batch_size num_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size num_params"]:
        """Compute Inverse Hessian-vector product."""
        pass
