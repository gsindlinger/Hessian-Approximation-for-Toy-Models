from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import flax.struct as struct
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float

from src.utils.data.data import Dataset
from src.utils.models.approximation_model import ApproximationModel


class ApproximationData(ABC):
    """
    Base class for Hessian approximation data.
    """

    pass


@struct.dataclass
class ModelContext:
    """
    Data class to hold model context information for Hessian computations.
    """

    inputs: Float[Array, "..."]
    params_flat: Float[Array, "..."]
    unravel_fn: Callable = struct.field(pytree_node=False)
    model_apply_fn: Callable = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)
    targets: Optional[Float[Array, "..."]] = None

    @classmethod
    def create(
        cls,
        model: ApproximationModel,
        params: Dict,
        dataset: Dataset,
        loss_fn: Callable,
        requires_targets: bool = True,
    ) -> "ModelContext":
        """
        Create a ModelContext instance from model, parameters, dataset, and loss function.
        """
        params_flat, unravel_fn = ravel_pytree(params)
        if requires_targets:
            targets = dataset.targets
        else:
            targets = None

        return cls(
            inputs=dataset.inputs,
            params_flat=params_flat,
            unravel_fn=unravel_fn,
            model_apply_fn=model.apply,
            loss_fn=loss_fn,
            targets=targets,
        )


@dataclass
class EKFACData(ApproximationData):
    """
    Data class to hold K-FAC related data.
    """

    activation_eigenvectors: Dict[str, Float[Array, "I I"]] = field(
        default_factory=dict
    )
    gradient_eigenvectors: Dict[str, Float[Array, "O O"]] = field(default_factory=dict)
    activation_eigenvalues: Dict[str, Float[Array, "I"]] = field(default_factory=dict)
    gradient_eigenvalues: Dict[str, Float[Array, "O"]] = field(default_factory=dict)
    eigenvalue_corrections: Dict[str, Float[Array, "I O"]] = field(default_factory=dict)
    layer_names: List[str] = field(default_factory=list)

    mean_eigenvalues: Dict[str, Float] = field(
        default_factory=dict
    )  # mean eigenvalues per layer
    mean_eigenvalues_aggregated: Float = field(
        default=0.0
    )  # mean eigenvalue over all layers

    mean_corrections: Dict[str, Float] = field(
        default_factory=dict
    )  # mean eigenvalue corrections per layer
    mean_corrections_aggregated: Float = field(
        default=0.0
    )  # mean eigenvalue correction over all layers
