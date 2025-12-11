from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from src.hessians.approximator.approximator import ApproximatorBase
from src.hessians.collector import CollectorActivationsGradients
from src.utils.data.data import Dataset
from src.utils.models.approximation_model import ApproximationModel


@dataclass
class FIMApproximator(ApproximatorBase):
    """
    Builder for FIM Hessian approximation.
    """

    data: None = None  # FIM does not require additional data storage

    def _build(
        self,
        model: ApproximationModel,
        params: Dict,
        dataset: Dataset,
        loss_fn: Callable,
    ):
        activations, gradients = CollectorActivationsGradients(model, params).collect(
            dataset.inputs, dataset.targets, loss_fn
        )

    def save_build(self, save_path: str):
        """
        Save the built Hessian approximation to the specified path.
        """
        pass
