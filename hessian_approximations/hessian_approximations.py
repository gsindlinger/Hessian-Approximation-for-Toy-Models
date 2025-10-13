from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from pyparsing import Callable
from torch.nn import MSELoss, CrossEntropyLoss
from config.config import Config
from models.models import ApproximationModel
import jax.numpy as jnp


def create_hessian(config: Config) -> HessianApproximation:
    from hessian_approximations.exact_hessian_regression import HessianExactRegression
    from hessian_approximations.fisher_information import FisherInformation
    from hessian_approximations.gauss_newton import GaussNewton
    from hessian_approximations.hessian import Hessian

    """Create Hessian approximation from config."""
    hessian_map = {
        "hessian": Hessian,
        "exact-hessian-regression": HessianExactRegression,
        "fim": FisherInformation,
        "gauss-newton": GaussNewton,
    }

    hessian_cls = hessian_map.get(config.hessian_approximation.name)
    if hessian_cls is None:
        raise ValueError(
            f"Unknown Hessian approximation method: {config.hessian_approximation.name}"
        )

    return hessian_cls()


def create_hessian_by_name(name: str) -> HessianApproximation:
    from hessian_approximations.exact_hessian_regression import HessianExactRegression
    from hessian_approximations.fisher_information import FisherInformation
    from hessian_approximations.gauss_newton import GaussNewton
    from hessian_approximations.hessian import Hessian

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
