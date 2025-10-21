from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
from pyparsing import Callable

from config.config import HessianApproximationConfig
from models.base import ApproximationModel


def create_hessian(config: HessianApproximationConfig) -> HessianApproximation:
    from hessian_approximations.fim.fisher_information import FisherInformation
    from hessian_approximations.gauss_newton.gauss_newton import GaussNewton
    from hessian_approximations.hessian.exact_hessian_regression import (
        HessianExactRegression,
    )
    from hessian_approximations.hessian.hessian import Hessian

    """Create Hessian approximation from config."""
    hessian_map = {
        "hessian": Hessian,
        "exact-hessian-regression": HessianExactRegression,
        "fim": FisherInformation,
        "gauss-newton": GaussNewton,
    }

    hessian_cls = hessian_map.get(config.name)
    if hessian_cls is None:
        raise ValueError(f"Unknown Hessian approximation method: {config.name}")

    model_kwargs = vars(config).copy()
    model_kwargs.pop("name", None)
    return hessian_cls(**model_kwargs)


def create_hessian_by_name(name: str) -> HessianApproximation:
    from hessian_approximations.fim.fisher_information import FisherInformation
    from hessian_approximations.gauss_newton.gauss_newton import GaussNewton
    from hessian_approximations.hessian.exact_hessian_regression import (
        HessianExactRegression,
    )
    from hessian_approximations.hessian.hessian import Hessian

    hessian_map = {
        "hessian": Hessian,
        "exact-hessian-regression": HessianExactRegression,
        "fim": FisherInformation,
        "gauss-newton": GaussNewton,
    }

    hessian_cls = hessian_map.get(name)
    if hessian_cls is None:
        raise ValueError(f"Unknown Hessian approximation method: {name}")

    return hessian_cls()


def hessian_approximation(
    method: HessianApproximation,
    model: ApproximationModel,
    parameters: Any,
    test_data: jnp.ndarray,  # Input for the Hessian
    test_targets: jnp.ndarray,  # Target for the Hessian
    loss: Callable,
):
    return method.compute_hessian(model, parameters, test_data, test_targets, loss)


def hessian_vector_product(
    method: HessianApproximation,
    model: ApproximationModel,
    parameters: Any,
    test_data: jnp.ndarray,
    test_targets: jnp.ndarray,
    loss: Callable,
    vector: jnp.ndarray,
):
    return method.compute_hvp(model, parameters, test_data, test_targets, loss, vector)


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
