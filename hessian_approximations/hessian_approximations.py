from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import jax.numpy as jnp

from config.config import Config
from models.dataclasses.model_data import ModelData
from models.train import train_or_load


@dataclass
class HessianApproximation(ABC):
    """Abstract base class for Hessian approximations."""

    full_config: Config
    model_data: ModelData = field(init=False)

    def __post_init__(self):
        self.model_data = train_or_load(self.full_config)

    @abstractmethod
    def compute_hessian(self) -> jnp.ndarray:
        """Compute Hessian approximation."""
        pass

    @abstractmethod
    def compute_hvp(self, vector: jnp.ndarray) -> jnp.ndarray:
        """Compute Hessian-vector product."""
        pass

    @abstractmethod
    def compute_ihvp(
        self,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Inverse Hessian-vector product."""
        pass
