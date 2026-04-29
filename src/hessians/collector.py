import json
import logging
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from src.config import PseudoTargetGenerationStrategy
from src.hessians.utils.data import DataActivationsGradients
from src.hessians.utils.pseudo_targets import generate_pseudo_targets
from src.utils.data.data import Dataset
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.models.approximation_model import ApproximationModel

logger = logging.getLogger(__name__)


class CacheMismatch(ValueError):
    """Raised by `CollectorActivationsGradients.load` when a cached manifest's
    `cache_inputs` disagree with the inputs requested for the current run."""


@dataclass
class CollectorActivationsGradients:
    """
    Run a model with hooks to collect per-layer activations and output gradients.

    Supports three pseudo-target generation strategies:
    - EMPIRICAL_FISHER: single pass with ground-truth labels (k = 1)
    - MCMC: multiple passes with sampled pseudo-targets (k = repetitions)
    - ALL_CLASSES: one pass per class, weighted by softmax(logits) (k = num_classes)

    Collection ordering: [all samples for k=0, all samples for k=1, ...] in
    shape (k*N, ...); teardown reshapes to (N, O, k).

    Output (DataActivationsGradients):
    - activations: (N, I) — identical across k, take first N
    - gradients:   (N, O, k)
    - probs:       (N, k) — softmax(logits) for ALL_CLASSES, ones otherwise
    """

    model: ApproximationModel
    params: Dict
    loss_fn: Callable

    pseudo_target_strategy: PseudoTargetGenerationStrategy = (
        PseudoTargetGenerationStrategy.MCMC
    )
    pseudo_target_repetitions: Optional[int] = 1

    MANIFEST_FILENAME: str = "manifest.json"

    def __post_init__(self):
        # Memmap-backed accumulators written to `_memmap_dir` during the
        # backward hook.  Shape per layer: (k*N, I) for activations,
        # (k*N, O) for gradients, lazily allocated on first hook call since
        # layer shapes are only known once the forward runs.  `_teardown`
        # returns views (no copy) that slice/reshape to the final layout.
        self._act_memmap: Dict[str, np.memmap] = {}
        self._grad_memmap: Dict[str, np.memmap] = {}
        self._offsets: Dict[str, int] = {}
        self._memmap_dir: Optional[Path] = None
        self._total_n: int = 0

        if (
            self.pseudo_target_strategy
            == PseudoTargetGenerationStrategy.EMPIRICAL_FISHER
        ):
            if self.pseudo_target_repetitions != 1:
                logger.warning(
                    f"EMPIRICAL_FISHER strategy requires repetitions=1, "
                    f"but got {self.pseudo_target_repetitions}. Setting to 1."
                )
                self.pseudo_target_repetitions = 1

        if self.pseudo_target_strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            if (
                self.pseudo_target_repetitions != self.model.output_dim
                and self.pseudo_target_repetitions != -1
            ):
                logger.warning(
                    "ALL_CLASSES strategy uses num_classes as k, "
                    "ignoring repetitions parameter."
                )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def collect(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        save_directory: Optional[str] = None,
        try_load: bool = False,
        rng_key: Optional[Array] = None,
        cache_inputs: Optional[Dict[str, Any]] = None,
    ) -> DataActivationsGradients:
        """
        Run the model with hooks to collect activations and gradients.

        `cache_inputs` is the set of upstream knobs that determine the
        collected data (dataset name, test_size, seed, ...). Stashed in
        the manifest at save time and validated against the manifest on
        load time — a mismatch refuses to reuse the cache and recomputes,
        rather than silently loading stale activations.
        """
        if try_load and save_directory is not None:
            try:
                logger.info(f"Trying to load collected data from: {save_directory}")
                return self.load(save_directory, expected_inputs=cache_inputs)
            except CacheMismatch as e:
                logger.warning(
                    f"Cache at {save_directory} stale — recomputing. {e}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load collected data from {save_directory}: {e}. "
                    f"Proceeding to collect data."
                )

        collection_dataset, metadata = self._generate_pseudo_targets(dataset, rng_key)

        # Memmaps are the primary storage: if no save directory was given
        # we spill to a tmp dir so device memory never has to hold the
        # full collection.
        if save_directory is not None:
            self._memmap_dir = Path(save_directory)
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="hessian_collector_"))
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        self._total_n = metadata["k"] * metadata["n_samples"]

        self._run_collection_loop(
            collection_dataset.inputs,
            collection_dataset.targets,
            batch_size,
            metadata,
        )

        collected_data = self._teardown(metadata)

        if save_directory is not None:
            logger.info(f"Saving collected data to: {save_directory}")
            self.save(save_directory, collected_data, metadata, cache_inputs=cache_inputs)

        return collected_data

    # ------------------------------------------------------------------
    # Hook target: called by layer_wrapper_vjp during the backward pass.
    # ------------------------------------------------------------------

    def backward_collector_fn(
        self, name: str, a: Float[Array, "N I"], g: Float[Array, "N O"]
    ) -> None:
        a_np = np.asarray(a)
        g_np = np.asarray(g)
        if name not in self._act_memmap:
            assert self._memmap_dir is not None
            I = a_np.shape[-1]
            O = g_np.shape[-1] if g_np.ndim > 1 else 1
            self._act_memmap[name] = np.lib.format.open_memmap(
                self._memmap_dir / f"activations_{name}.npy",
                mode="w+", dtype=a_np.dtype, shape=(self._total_n, I),
            )
            self._grad_memmap[name] = np.lib.format.open_memmap(
                self._memmap_dir / f"gradients_{name}.npy",
                mode="w+", dtype=g_np.dtype, shape=(self._total_n, O),
            )
            self._offsets[name] = 0

        B = a_np.shape[0]
        off = self._offsets[name]
        self._act_memmap[name][off:off + B] = a_np
        self._grad_memmap[name][off:off + B] = g_np.reshape(B, -1)
        self._offsets[name] = off + B

    # ------------------------------------------------------------------
    # Pseudo-target generation
    # ------------------------------------------------------------------

    def _generate_pseudo_targets(
        self, dataset: Dataset, rng_key: Optional[Array]
    ) -> Tuple[Dataset, Dict[str, Any]]:
        """
        Return (expanded_dataset, metadata).  All strategies produce
        collection order [all samples for k=0, all samples for k=1, ...]
        so `_teardown` can reshape (k*N, ...) -> (k, N, ...) uniformly.
        """
        inputs = dataset.inputs
        n_samples = inputs.shape[0]
        strategy = self.pseudo_target_strategy

        if strategy == PseudoTargetGenerationStrategy.EMPIRICAL_FISHER:
            logger.info("[PSEUDO-TARGETS] EMPIRICAL_FISHER (ground-truth labels)")
            metadata = {"strategy": strategy, "n_samples": n_samples, "k": 1}
            return dataset, metadata

        if strategy == PseudoTargetGenerationStrategy.MCMC:
            if rng_key is None:
                raise ValueError(
                    "rng_key is required for MCMC pseudo-target generation"
                )
            k = self.pseudo_target_repetitions
            logger.info(f"[PSEUDO-TARGETS] MCMC with {k} repetitions")

            expanded_inputs_list = []
            expanded_targets_list = []
            for rep in range(k):
                rep_key = jax.random.fold_in(rng_key, rep)
                sampled_targets = generate_pseudo_targets(
                    model=self.model,
                    params=self.params,
                    inputs=inputs,
                    loss_fn=self.loss_fn,
                    rng_key=rep_key,
                    repetitions=1,
                )
                expanded_inputs_list.append(inputs)
                expanded_targets_list.append(sampled_targets)

            expanded_inputs = jnp.concatenate(expanded_inputs_list, axis=0)
            expanded_targets = jnp.concatenate(expanded_targets_list, axis=0)

            metadata = {
                "strategy": strategy,
                "n_samples": n_samples,
                "k": k,
            }
            return Dataset(expanded_inputs, expanded_targets), metadata

        if strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            if not hasattr(self.model, "output_dim"):
                raise ValueError(
                    "ALL_CLASSES strategy requires model to have 'output_dim' attribute"
                )
            k = self.model.output_dim
            logger.info(f"[PSEUDO-TARGETS] ALL_CLASSES with {k} classes")

            logits = jax.vmap(lambda x: self.model.apply(self.params, x))(inputs)
            assert isinstance(logits, jnp.ndarray), (
                "Model predictions must be a jnp.ndarray"
            )
            probabilities = jax.nn.softmax(logits, axis=-1)  # (N, K)

            expanded_inputs = jnp.tile(inputs, (k, 1))  # (K*N, features)
            expanded_targets = jnp.repeat(jnp.arange(k), n_samples)  # (K*N,)

            metadata = {
                "strategy": strategy,
                "n_samples": n_samples,
                "k": k,
                "probabilities": probabilities,
            }
            return Dataset(expanded_inputs, expanded_targets), metadata

        raise ValueError(f"Unknown pseudo-target strategy: {strategy}")

    # ------------------------------------------------------------------
    # Forward/backward loop over batches
    # ------------------------------------------------------------------

    def _run_collection_loop(
        self,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        batch_size: Optional[int],
        metadata: Dict[str, Any],
    ) -> None:
        def loss_fn_for_grad(p, inputs, targets):
            predictions = self.model.apply(
                p, inputs, self, method=self.model.collector_apply
            )
            return self.loss_fn(predictions, targets, reduction="sum")

        dataloader_batch_size = (
            JAXDataLoader.get_batch_size() if batch_size is None else batch_size
        )
        dataloader = JAXDataLoader(
            inputs=inputs,
            targets=targets,
            shuffle=False,
            batch_size=dataloader_batch_size,
        )

        strategy_name = metadata["strategy"].value
        logger.info(
            f"[COLLECTION] {strategy_name.upper()}: "
            f"N={metadata['n_samples']}, k={metadata['k']}"
        )

        for batch_data, batch_targets in dataloader:
            _ = jax.value_and_grad(loss_fn_for_grad)(
                self.params, batch_data, batch_targets
            )

    # ------------------------------------------------------------------
    # Teardown: concatenate batches and reshape to (N, O, k)
    # ------------------------------------------------------------------

    def _teardown(self, metadata: Dict[str, Any]) -> DataActivationsGradients:
        strategy = metadata["strategy"]
        if isinstance(strategy, str):
            strategy = PseudoTargetGenerationStrategy(strategy)

        n_samples = metadata["n_samples"]
        k = metadata["k"]

        # All k reps share the same inputs (same activations), so take the
        # first-rep slice for activations.  Gradients are reshaped in place
        # to (N, O, k) as a non-contiguous view over the memmap — downstream
        # chunked consumers materialize only the slices they touch.
        activations_final: Dict[str, Array] = {}
        gradients_final: Dict[str, Array] = {}
        for layer_name, act_mm in self._act_memmap.items():
            grad_mm = self._grad_memmap[layer_name]
            O_dim = grad_mm.shape[-1]
            activations_final[layer_name] = act_mm[:n_samples]
            gradients_final[layer_name] = grad_mm.reshape(k, n_samples, O_dim).transpose(1, 2, 0)

        # Unified probs: (N, k).  ALL_CLASSES uses softmax; EF/MCMC use ones.
        # FLAGGED: MCMC normalization convention deferred.
        if strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            probs = metadata["probabilities"]
        else:
            probs = jnp.ones((n_samples, k), dtype=jnp.float32)

        logger.info(
            f"[TEARDOWN] {strategy.value.upper()}: "
            f"activations (N={n_samples}, ...), gradients (N={n_samples}, ..., k={k})"
        )

        return DataActivationsGradients(
            activations=activations_final,
            gradients=gradients_final,
            probs=probs,
            layer_names=self.model.get_layer_names(),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        directory: str,
        data: DataActivationsGradients,
        metadata: Dict[str, Any],
        cache_inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Per-layer activations/gradients are already on disk as .npy
        # memmaps in `directory`; only the small sidecar metadata + probs
        # need writing here.
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "probs.npy", np.asarray(data.probs))
        manifest = {
            "layer_names": list(data.layer_names),
            "n_samples": int(metadata["n_samples"]),
            "k": int(metadata["k"]),
            "pseudo_target_strategy": self.pseudo_target_strategy.value,
            "pseudo_target_repetitions": int(self.pseudo_target_repetitions),
            "cache_inputs": cache_inputs or {},
        }
        with open(directory / self.MANIFEST_FILENAME, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Saved collected data manifest to: {directory}")

    @staticmethod
    def load(
        directory: str,
        expected_inputs: Optional[Dict[str, Any]] = None,
    ) -> DataActivationsGradients:
        directory = Path(directory)
        manifest_path = directory / CollectorActivationsGradients.MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ValueError(f"File not found: {manifest_path}")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if expected_inputs is not None:
            cached_inputs = manifest.get("cache_inputs")
            if cached_inputs is None:
                raise CacheMismatch(
                    f"Manifest at {directory} predates cache validation "
                    f"(no cache_inputs key). Rebuild the cache."
                )
            if cached_inputs != expected_inputs:
                diff_keys = sorted(
                    k for k in set(cached_inputs) | set(expected_inputs)
                    if cached_inputs.get(k) != expected_inputs.get(k)
                )
                raise CacheMismatch(
                    f"Mismatch on {diff_keys} — cached={cached_inputs}, "
                    f"requested={expected_inputs}."
                )

        layer_names: List[str] = list(manifest["layer_names"])
        n_samples = int(manifest["n_samples"])
        k = int(manifest["k"])

        activations: Dict[str, Array] = {}
        gradients: Dict[str, Array] = {}
        for name in layer_names:
            act_mm = np.load(directory / f"activations_{name}.npy", mmap_mode="r")
            grad_mm = np.load(directory / f"gradients_{name}.npy", mmap_mode="r")
            O_dim = grad_mm.shape[-1]
            activations[name] = act_mm[:n_samples]
            gradients[name] = grad_mm.reshape(k, n_samples, O_dim).transpose(1, 2, 0)

        probs = jnp.array(np.load(directory / "probs.npy"))
        return DataActivationsGradients(
            activations=activations,
            gradients=gradients,
            probs=probs,
            layer_names=layer_names,
        )


# Backwards-compatible alias for model files that type-hint the hook target.
CollectorBase = CollectorActivationsGradients


# ---------------------------------------------------------------------------
# Hooks for collecting activations and gradients of specific layers
# ---------------------------------------------------------------------------


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def layer_wrapper_vjp(
    pure_apply_fn: Callable,
    params: Dict,
    x: jnp.ndarray,
    name: str,
    collector: CollectorActivationsGradients,
):
    """Custom VJP wrapper for capturing activations and gradients."""
    return pure_apply_fn(params, x)


def layer_wrapper_fwd(
    pure_apply_fn: Callable,
    params: Dict,
    x: jnp.ndarray,
    name: str,
    collector: CollectorActivationsGradients,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Dict]]:
    output = pure_apply_fn(params, x)
    residuals = (x, params)
    return output, residuals


def layer_wrapper_bwd(
    pure_apply_fn: Callable,
    name: str,
    collector: CollectorActivationsGradients,
    residuals: Tuple[jnp.ndarray, Dict],
    g: jnp.ndarray,
) -> Tuple[Dict, jnp.ndarray]:
    a, params = residuals
    collector.backward_collector_fn(name, a, g)
    _primals, vjp_fn = jax.vjp(pure_apply_fn, params, a)
    param_grads, input_grads = vjp_fn(g)
    return (param_grads, input_grads)


layer_wrapper_vjp.defvjp(layer_wrapper_fwd, layer_wrapper_bwd)
