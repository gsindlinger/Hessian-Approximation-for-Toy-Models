from typing import Any, Callable

from jax import numpy as jnp

from config.config import HessianApproximationConfig, HessianName
from hessian_approximations.fim.fisher_information import FisherInformation
from hessian_approximations.gauss_newton.gauss_newton import GaussNewton
from hessian_approximations.hessian.exact_hessian_regression import (
    HessianExactRegression,
)
from hessian_approximations.hessian.hessian import Hessian
from hessian_approximations.hessian_approximations import HessianApproximation
from hessian_approximations.kfac.kfac import KFAC
from models.base import ApproximationModel


def create_hessian(config: HessianApproximationConfig) -> HessianApproximation:
    hessian_map = {
        HessianName.HESSIAN: Hessian,
        HessianName.EXACT_HESSIAN_REGRESSION: HessianExactRegression,
        HessianName.FIM: FisherInformation,
        HessianName.GAUSS_NEWTON: GaussNewton,
        HessianName.KFAC: KFAC,
    }

    hessian_cls = hessian_map.get(config.name)
    if hessian_cls is None:
        raise ValueError(f"Unknown Hessian approximation method: {config.name}")

    model_kwargs = vars(config).copy()
    model_kwargs.pop("name", None)
    return hessian_cls(**model_kwargs)


def create_hessian_by_name(name: str | HessianName) -> HessianApproximation:
    if isinstance(name, str):
        try:
            name = HessianName(name)
        except ValueError:
            raise ValueError(
                f"Invalid Hessian name: {name}. Must be one of {[n.value for n in HessianName]}"
            )
    config = HessianApproximationConfig(name=name)
    return create_hessian(config)


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


def inverse_hessian_vector_product(
    method: HessianApproximation,
    model: ApproximationModel,
    parameters: Any,
    test_data: jnp.ndarray,
    test_targets: jnp.ndarray,
    loss: Callable,
    vector: jnp.ndarray,
):
    return method.compute_ihvp(model, parameters, test_data, test_targets, loss, vector)
