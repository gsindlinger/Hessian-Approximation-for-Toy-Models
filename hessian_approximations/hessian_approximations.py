from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
from pyparsing import Callable

from models.base import ApproximationModel


class HessianApproximation(ABC):
    """Abstract base class for Hessian approximations."""

    @abstractmethod
    def compute_hessian(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
    ) -> jnp.ndarray:
        """Compute Hessian approximation."""
        pass

    @abstractmethod
    def compute_hvp(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Hessian-vector product."""
        pass

    @abstractmethod
    def compute_ihvp(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Inverse Hessian-vector product."""
        pass
