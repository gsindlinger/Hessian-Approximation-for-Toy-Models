from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    get_args,
    get_origin,
)

import flax.struct as struct
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float

from src.config import PseudoTargetGenerationStrategy
from src.utils.data.data import Dataset
from src.utils.models.approximation_model import ApproximationModel

logger = logging.getLogger(__name__)


class ApproximationData(ABC):
    """
    Base class for Hessian approximation data.
    """

    @classmethod
    def name(cls) -> str:
        """Returns the directory name for saving/loading data."""
        return cls.__name__.lower()

    def save(self, directory: str) -> None:
        """Saves the components data to the specified directory."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, np.ndarray] = {}

        assert is_dataclass(self), "Data must be a dataclass instance."

        for field_t in fields(self):
            value = getattr(self, field_t.name)
            field_type = field_t.type

            # -------------------------
            # Dict[str, array]
            # -------------------------
            if get_origin(field_type) is dict:
                key_type, _ = get_args(field_type)
                if key_type is str:
                    for k, arr in value.items():
                        arrays[f"{field_t.name}/{k}"] = np.asarray(arr)
                    continue

            # -------------------------
            # List[str]
            # -------------------------
            if self._is_string_sequence(field_type):
                arrays[field_t.name] = np.array(value, dtype="U")
                continue

            # -------------------------
            # Scalar / array
            # -------------------------
            arrays[field_t.name] = np.asarray(value)

        np.savez_compressed(save_dir / "data.npz", **arrays)  # type: ignore

    @staticmethod
    def exists(directory: str | None) -> bool:
        """Checks if the components data exists in the specified directory."""
        if directory is None:
            return False
        load_dir = Path(directory)
        return (load_dir / "data.npz").exists()

    @classmethod
    def load(cls: Type[ApproximationData], directory: str) -> ApproximationData:
        """Loads the components data and corresponding config from the specified directory."""
        load_dir = Path(directory)

        # Load arrays with allow_pickle=True to handle string arrays
        loaded = np.load(load_dir / "data.npz", allow_pickle=True)

        if not is_dataclass(cls):
            raise TypeError("The subclass must annotate `data` with a dataclass type")

        data_kwargs = {}

        for field_t in fields(cls):
            field_type = field_t.type

            # Dict[str, array]
            if get_origin(field_type) is dict:
                key_type, _ = get_args(field_type)
                if key_type is str:
                    prefix = f"{field_t.name}/"
                    d = {}
                    for k in loaded.files:
                        if k.startswith(prefix):
                            d[k[len(prefix) :]] = jnp.array(loaded[k])
                    data_kwargs[field_t.name] = d
                    continue

            if cls._is_string_sequence(field_type):
                lst = list(map(str, loaded[field_t.name]))
                data_kwargs[field_t.name] = lst
                continue

            # Scalar / array
            if field_t.name in loaded:
                arr = loaded[field_t.name]
                if arr.shape == ():
                    data_kwargs[field_t.name] = arr.item()
                else:
                    data_kwargs[field_t.name] = jnp.array(arr)

        data_obj = cls(**data_kwargs)
        return data_obj

    @staticmethod
    def _is_string_sequence(t):
        return (
            get_origin(t) in (list, List, tuple, Tuple)
            and get_args(t)
            and get_args(t)[0] is str
        )


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
    model: Optional[ApproximationModel] = struct.field(pytree_node=False, default=None)

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
            model=model,
        )


def layer_shapes_from_model_context(
    ctx: ModelContext,
) -> Dict[str, Tuple[int, int]]:
    """
    Walk the Flax param tree and return `{layer_name: (I_aug, O)}` matching
    the KFAC layer convention (kernel + optional bias as a single
    bias-augmented block).

    Uses `ctx.model.get_layer_names()` for the canonical layer ordering and
    `ctx.unravel_fn(ctx.params_flat)` to read shapes.  Asserts that the
    layer-aggregated total matches `params_flat.size`, so callers can rely
    on each layer's parameters being contiguous in `params_flat` (which is
    the standard Flax tree-flatten ordering for sequential models).
    """
    if ctx.model is None:
        raise ValueError(
            "ModelContext.model is required to derive layer shapes."
        )
    params = ctx.unravel_fn(ctx.params_flat)
    if "params" in params:
        params_root = params["params"]
    else:
        params_root = params
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(params_root)
    layer_names = ctx.model.get_layer_names()

    shapes: Dict[str, Tuple[int, int]] = {}
    for layer in layer_names:
        layer_leaves = [
            leaf
            for path, leaf in leaves_with_paths
            if "/".join(k.key for k in path[:-1]) == layer
        ]
        if not layer_leaves:
            raise ValueError(
                f"No leaves found for layer '{layer}' in Flax param tree."
            )
        # Expected: one kernel (I, O) and optional bias (O,).
        kernel_candidates = [l for l in layer_leaves if l.ndim == 2]
        if not kernel_candidates:
            raise ValueError(
                f"Layer '{layer}' has no 2-D kernel leaf — cannot derive (I, O)."
            )
        kernel = kernel_candidates[0]
        I, O = kernel.shape
        has_bias = any(l.ndim == 1 for l in layer_leaves)
        shapes[layer] = ((I + 1) if has_bias else I, O)

    total = sum(I * O for (I, O) in shapes.values())
    if total != ctx.params_flat.size:
        raise ValueError(
            f"Sum of layer sizes ({total}) does not match params_flat.size "
            f"({ctx.params_flat.size}).  Either the model has parameters "
            f"outside named layers, or layers are not contiguous in the "
            f"flattened param vector."
        )
    return shapes


@dataclass
class EKFACData(ApproximationData):
    """
    Data class to hold EK-FAC related data.
    """

    activation_eigenvectors: Dict[str, Float[Array, "I I"]] = field(
        default_factory=dict
    )
    gradient_eigenvectors: Dict[str, Float[Array, "O O"]] = field(default_factory=dict)
    activation_eigenvalues: Dict[str, Float[Array, "I"]] = field(default_factory=dict)
    gradient_eigenvalues: Dict[str, Float[Array, "O"]] = field(default_factory=dict)
    eigenvalue_corrections: Dict[str, Float[Array, "I O"]] = field(default_factory=dict)

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

    @classmethod
    def name(cls) -> str:
        return "ekfac_data"


@dataclass
class ETKFACData(ApproximationData):
    """
    Data class to hold (E)TK-FAC related data.
    """

    @classmethod
    def name(cls) -> str:
        return "ekfac_data"


@dataclass
class FIMData(ApproximationData):
    """
    Data class to hold FIM related data.
    """

    per_sample_grads: Float[Array, "N n_params | N K n_params"] = field(
        default_factory=lambda: jnp.array([])
    )
    probabilities: Optional[Float[Array, "N K"]] = field(default=None)

    @classmethod
    def name(cls) -> str:
        return "fim_data"


@dataclass
class DataActivationsGradients(ApproximationData):
    """
    Data structure to hold activations and gradients per layer.

    Attributes:
    activations: Per-layer activations, shape (N, I_l) per layer
    gradients: Per-layer gradients, shape (K, N, O_l) per layer (K depending on pseudo-target strategy)
    probabilities: Predicted probabilities p(c|x), shape (N, K), optional (only need when ALL_CLASS strategy is used for pseudo-targets)
    layer_names: Ordered list of layer names
    """

    activations: Dict[str, Float[Array, "N I"]] = field(
        default_factory=dict
    )  # Only used for Block FIM
    gradients: Dict[str, Float[Array, "K N O"]] = field(default_factory=dict)
    layer_names: List[str] = field(default_factory=list)
    probabilities: Optional[Float[Array, "N K"]] = field(default=None)
    pseudo_target_strategy: PseudoTargetGenerationStrategy = field(
        default=PseudoTargetGenerationStrategy.EMPIRICAL_FISHER
    )

    @classmethod
    def name(cls) -> str:
        return "activations_gradients"


@dataclass
class BlockHessianData(ApproximationData):
    """
    Data class to hold Block Hessian related data.

    `blocks[i]` is the `(start, end)` flat-index range for the i-th
    layer (kernel + optional bias merged).  `layer_names[i]` is the
    matching layer name.  `layer_shapes[name]` is the per-layer
    `(I_aug, O)` shape used for `LayerVector` reshapes.
    """

    blocks: List[Tuple[int, int]] = field(default_factory=list)
    layer_names: List[str] = field(default_factory=list)
    layer_shapes: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    n_params: int = 0
    block_mask: Float[Array, "n_params n_params"] = field(
        default_factory=lambda: jnp.array([[]])
    )

    @classmethod
    def name(cls) -> str:
        return "block_hessian_data"
