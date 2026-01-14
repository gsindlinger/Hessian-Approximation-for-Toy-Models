from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from src.hessians.computer.etkfac import ETKFACComputer
from src.hessians.computer.kfac import KFACComputer
from src.hessians.utils.data import DataActivationsGradients, EKFACData


@dataclass
class TKFACComputer(KFACComputer):
    """
    Trace-normalized KFAC (TKFAC) Hessian approximation.
    Uses TKFAC covariances (from ETKFACComputer) but simple eigenvalue products (like KFAC).
    """

    @staticmethod
    def _build(
        compute_context: Tuple[DataActivationsGradients, DataActivationsGradients],
    ) -> EKFACData:
        return ETKFACComputer._build(compute_context)