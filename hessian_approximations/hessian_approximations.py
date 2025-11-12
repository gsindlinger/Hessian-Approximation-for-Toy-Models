from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp

from config.config import Config


@dataclass
class HessianApproximation(ABC):
    """Abstract base class for Hessian approximations."""

    full_config: Config

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
