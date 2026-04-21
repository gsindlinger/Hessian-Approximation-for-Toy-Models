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

    Walks leaves in the order `tree_flatten_with_path` yields them — which
    matches `ravel_pytree`'s flat layout — and enforces:

    - every name in `model.get_layer_names()` has at least one leaf,
    - each layer's leaves are *contiguous* in the flat vector,
    - the declared layer order matches the flat-layout order.

    Callers slicing the flat vector by `layer_shapes` (`LayerVector.from_flat`,
    `LayerMatrix.from_dense`) can then rely on offsets accumulating correctly.
    """
    if ctx.model is None:
        raise ValueError(
            "ModelContext.model is required to derive layer shapes."
        )
    params = ctx.unravel_fn(ctx.params_flat)
    params_root = params["params"] if "params" in params else params
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(params_root)
    layer_names = list(ctx.model.get_layer_names())

    # Walk leaves in flat-layout order, record (start, end) runs per layer.
    layer_set = set(layer_names)
    per_layer_runs: Dict[str, List[Tuple[int, int]]] = {n: [] for n in layer_names}
    per_layer_leaves: Dict[str, List] = {n: [] for n in layer_names}
    offset = 0
    for path, leaf in leaves_with_paths:
        layer = "/".join(k.key for k in path[:-1])
        if layer in layer_set:
            per_layer_runs[layer].append((offset, offset + leaf.size))
            per_layer_leaves[layer].append(leaf)
        offset += int(leaf.size)

    # Contiguity: each layer's leaves must form one unbroken run.
    for name in layer_names:
        runs = per_layer_runs[name]
        if not runs:
            raise ValueError(
                f"No leaves found for layer '{name}' in Flax param tree."
            )
        for (_, prev_end), (next_start, _) in zip(runs, runs[1:]):
            if prev_end != next_start:
                raise ValueError(
                    f"Layer '{name}' leaves are not contiguous in the flat "
                    f"param vector (gap between offsets {prev_end} and {next_start}). "
                    f"Flax sorts param keys alphabetically; another layer's leaf "
                    f"is interleaved with this one."
                )

    # Declared order must match flat-layout order.
    first_offsets = [per_layer_runs[n][0][0] for n in layer_names]
    if first_offsets != sorted(first_offsets):
        layout_order = sorted(layer_names, key=lambda n: per_layer_runs[n][0][0])
        raise ValueError(
            f"model.get_layer_names() returns {layer_names} but the Flax flat "
            f"layout order is {layout_order}. Declare layers in the order Flax "
            f"places them (alphabetical at each level of the param tree)."
        )

    shapes: Dict[str, Tuple[int, int]] = {}
    for name in layer_names:
        layer_leaves = per_layer_leaves[name]
        kernel_candidates = [l for l in layer_leaves if l.ndim == 2]
        if not kernel_candidates:
            raise ValueError(
                f"Layer '{name}' has no 2-D kernel leaf — cannot derive (I, O)."
            )
        kernel = kernel_candidates[0]
        I, O = kernel.shape
        has_bias = any(l.ndim == 1 for l in layer_leaves)
        shapes[name] = ((I + 1) if has_bias else I, O)

    total = sum(I * O for (I, O) in shapes.values())
    if total != ctx.params_flat.size:
        raise ValueError(
            f"Sum of layer sizes ({total}) does not match params_flat.size "
            f"({ctx.params_flat.size}). The model has parameters outside "
            f"named layers."
        )
    return shapes


@dataclass
class DataActivationsGradients(ApproximationData):
    """
    Single collector run: per-layer activations and per-sample/per-k output gradients.

    Attributes:
        activations: Per-layer input activations, shape (N, I_l).
        gradients: Per-layer output gradients, shape (N, O_l, k). k is the
            number of pseudo-target draws (k=1 for empirical_fisher,
            k=num_classes for all_classes, k=repetitions for mcmc).
        probs: Per-sample, per-k weights, shape (N, k). Conventions:
            - empirical_fisher: ones (N, 1)
            - all_classes: softmax(logits), rows sum to 1
            - mcmc: ones (N, k)  # FLAGGED: normalization convention deferred
        layer_names: Ordered list of layer names.
    """

    activations: Dict[str, Float[Array, "N I"]] = field(default_factory=dict)
    gradients: Dict[str, Float[Array, "N O k"]] = field(default_factory=dict)
    probs: Float[Array, "N k"] = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32)
    )
    layer_names: List[str] = field(default_factory=list)

    @classmethod
    def name(cls) -> str:
        return "activations_gradients"


