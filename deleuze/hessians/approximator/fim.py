from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from deleuze.hessians.approximator.approximator import ApproximatorBase
from deleuze.hessians.collector import CollectorActivationsGradients
from deleuze.models.approximation_model import ApproximationModel
from deleuze.utils.data.data import Dataset


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
