from dataclasses import asdict, replace
from hashlib import sha256
from typing import Any, Callable, Dict, List, MutableMapping, Tuple

import jax
import numpy as np

from src.config import LossType, ModelConfig
from src.hessians.layer_matrix import LayerMatrix
from src.hessians.utils.data import ModelContext
from src.utils.data.data import Dataset
from src.utils.loss import get_loss
from src.utils.models.approximation_model import ApproximationModel
from src.utils.models.registry import ModelRegistry
from src.utils.optimizers import optimizer
from src.utils.train import train_model

TrainedModelEntry = Tuple[ApproximationModel, Dict, Callable]


def _freeze_hidden_dim(hidden_dim: Any) -> Any:
    if hidden_dim is None:
        return None
    if isinstance(hidden_dim, (list, tuple)):
        return tuple(_freeze_hidden_dim(item) for item in hidden_dim)
    return hidden_dim


def _array_digest(array: Any) -> str:
    np_array = np.asarray(array)
    return sha256(np_array.tobytes()).hexdigest()


def training_cache_key(
    config: ModelConfig,
    dataset: Dataset,
    *,
    seed: int,
    shuffle: bool,
) -> Tuple[Any, ...]:
    """Build a cache key that is stable across equivalent test fixtures."""
    training_cfg = asdict(config.training)
    return (
        config.architecture.value,
        _freeze_hidden_dim(config.hidden_dim),
        config.activation.value if config.activation is not None else None,
        config.loss.value,
        tuple(sorted(training_cfg.items())),
        np.asarray(dataset.inputs).shape,
        str(np.asarray(dataset.inputs).dtype),
        _array_digest(dataset.inputs),
        np.asarray(dataset.targets).shape,
        str(np.asarray(dataset.targets).dtype),
        _array_digest(dataset.targets),
        seed,
        shuffle,
    )


def train_model_for_dataset(
    config: ModelConfig,
    dataset: Dataset,
    *,
    seed: int,
    shuffle: bool = False,
) -> Tuple[ApproximationModel, Dict, Callable]:
    """Train a model against a dataset without mutating the original config fixture."""
    model_config = replace(
        config,
        input_dim=dataset.input_dim(),
        output_dim=dataset.output_dim(),
    )
    loss_fn = get_loss(model_config.loss)

    model, params, _ = train_model(
        model_config=model_config,
        dataloader=dataset.get_dataloader(
            batch_size=model_config.training.batch_size,
            seed=seed,
            shuffle=shuffle,
        ),
        loss_fn=loss_fn,
        optimizer=optimizer(
            model_config.training.optimizer,
            lr=model_config.training.learning_rate,
        ),
        epochs=model_config.training.epochs,
    )

    return model, params, loss_fn


def cached_train_model_for_dataset(
    config: ModelConfig,
    dataset: Dataset,
    registry: MutableMapping[Tuple[Any, ...], TrainedModelEntry],
    *,
    seed: int,
    shuffle: bool = False,
) -> TrainedModelEntry:
    """Train once per equivalent config/dataset pair and reuse the result."""
    cache_key = training_cache_key(
        config,
        dataset,
        seed=seed,
        shuffle=shuffle,
    )
    if cache_key not in registry:
        registry[cache_key] = train_model_for_dataset(
            config,
            dataset,
            seed=seed,
            shuffle=shuffle,
        )
    return registry[cache_key]


def create_model_context(
    dataset: Dataset,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
) -> ModelContext:
    model, params, loss_fn = model_params_loss
    return ModelContext.create(
        dataset=dataset,
        model=model,
        params=params,
        loss_fn=loss_fn,
    )


def per_layer_true_offsets(ctx: ModelContext) -> Dict[str, Tuple[int, int]]:
    """Per-layer `(start, end)` offsets in the flat parameter vector, derived
    from the tree-flatten order of `ctx.unravel_fn(ctx.params_flat)` — i.e.
    the TRUE layout `ravel_pytree` / `jax.hessian` use, independent of
    anything the pipeline computed.
    """
    params = ctx.unravel_fn(ctx.params_flat)
    params_root = params["params"] if "params" in params else params
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(params_root)
    runs: Dict[str, List[Tuple[int, int]]] = {}
    offset = 0
    for path, leaf in leaves_with_paths:
        layer = "/".join(k.key for k in path[:-1])
        runs.setdefault(layer, []).append((offset, offset + int(leaf.size)))
        offset += int(leaf.size)
    return {layer: (r[0][0], r[-1][1]) for layer, r in runs.items()}


def verify_architecture_layer_alignment(
    model_config: ModelConfig,
    *,
    n_samples: int = 8,
    seed: int = 0,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    """Drop-in check for any architecture: does `HessianComputer`'s per-layer
    slicing match Flax's true flat layout?

    Synthesizes `n_samples` random inputs/targets sized to `model_config`'s
    `input_dim`/`output_dim`, initializes params (no training — alignment is
    about param-vector layout, not convergence), runs `HessianComputer`, and
    delegates to `assert_layer_alignment`.

    Targets are integers in `[0, output_dim)` for cross-entropy, Gaussians
    for MSE.
    """
    from src.hessians.computer.hessian import HessianComputer

    assert model_config.input_dim > 0 and model_config.output_dim > 0, (
        "verify_architecture_layer_alignment requires model_config.input_dim "
        "and model_config.output_dim to be set."
    )

    model = ModelRegistry.get_model(model_config)
    inputs_key, targets_key, init_key = jax.random.split(
        jax.random.PRNGKey(seed), 3
    )
    inputs = jax.random.normal(inputs_key, (n_samples, model_config.input_dim))
    if model_config.loss == LossType.CROSS_ENTROPY:
        targets = jax.random.randint(
            targets_key, (n_samples,), minval=0, maxval=model_config.output_dim
        )
    else:
        targets = jax.random.normal(
            targets_key, (n_samples, model_config.output_dim)
        )

    params = model.init(init_key, inputs[:1])
    ctx = ModelContext.create(
        model=model,
        params=params,
        dataset=Dataset(inputs=inputs, targets=targets),
        loss_fn=get_loss(model_config.loss),
    )
    hessian = HessianComputer(compute_context=ctx).build()
    assert hessian.layer_matrix is not None
    assert_layer_alignment(ctx, hessian.layer_matrix, atol=atol, rtol=rtol)


def assert_layer_alignment(
    ctx: ModelContext,
    lmat: LayerMatrix,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    """For each layer declared by `ctx.model.get_layer_names()`, verify that
    the pipeline's block `lmat.blocks[(gi, gj)]` equals the slice of the
    exact JAX Hessian at the TRUE Flax offsets for `(gi, gj)`.

    Only meaningful for pipelines that materialize the exact Hessian (e.g.
    `HessianComputer`). Loops over all declared layer groups and every
    populated `(gi, gj)` pair, so it catches per-architecture ordering /
    slicing mismatches that full-flat-vector HVP tests cannot detect.
    """
    from src.hessians.computer.hessian import HessianComputer

    H = HessianComputer._compute_hessian(ctx, 0.0)
    true_offsets = per_layer_true_offsets(ctx)

    layer_names = list(ctx.model.get_layer_names())
    for gi in layer_names:
        for gj in layer_names:
            if (gi, gj) not in lmat.blocks:
                continue
            si, ei = true_offsets[gi]
            sj, ej = true_offsets[gj]
            truth = H[si:ei, sj:ej]
            ours = lmat.blocks[(gi, gj)].to_dense()
            np.testing.assert_allclose(
                np.asarray(ours),
                np.asarray(truth),
                atol=atol,
                rtol=rtol,
                err_msg=f"block ({gi}, {gj}) misaligned with Flax's true flat layout",
            )
