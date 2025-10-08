from __future__ import annotations
from abc import ABC, abstractmethod
from torch.nn import MSELoss, CrossEntropyLoss
import torch
from models.models import ApproximationModel


def create_hessian(config) -> HessianApproximation:
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


def hessian_approximation(
    method: HessianApproximation,
    model: ApproximationModel,
    test_data: torch.Tensor,  # Input for the Hessian
    test_targets: torch.Tensor,  # Target for the Hessian
    loss: MSELoss | CrossEntropyLoss,
):
    return method.compute(model, test_data, test_targets, loss)


class HessianApproximation(ABC):
    @abstractmethod
    def compute(
        self,
        model: ApproximationModel,
        test_data: torch.Tensor,
        test_targets: torch.Tensor,
        loss: MSELoss | CrossEntropyLoss,
    ):
        raise NotImplementedError("Subclasses should implement this method.")
