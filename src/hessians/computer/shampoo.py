from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from src.hessians.computer.eshampoo import EShampooComputer
from src.hessians.computer.kfac import KFACComputer
from src.hessians.utils.data import DataActivationsGradients, EKFACData


@dataclass
class ShampooComputer(KFACComputer):
    """
    Shampoo Hessian approximation.
    Uses Shampoo covariances (from EShampooComputer) but simple eigenvalue products (like KFAC).
    """

    @staticmethod
    def _build(
        compute_context: Tuple[DataActivationsGradients, DataActivationsGradients],
    ) -> EKFACData:
        return EShampooComputer._build(compute_context)