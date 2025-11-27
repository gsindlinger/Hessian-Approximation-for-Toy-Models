from dataclasses import dataclass
from typing import Any, Callable, Dict

from ...data.data import AbstractDataset
from ...models.base import ApproximationModel


@dataclass
class ModelContext:
    """Container for the model, dataset, parameters, and loss function."""

    model: ApproximationModel
    dataset: AbstractDataset
    params: Dict[str, Any]
    loss: Callable
