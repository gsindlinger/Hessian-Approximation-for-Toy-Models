import logging
import os
from dataclasses import dataclass, field
from functools import partial
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

    FILENAME: str = "collected_activations_gradients.npz"

    def __post_init__(self):
        # Per-call accumulators for the backward hook: each list element is
        # one hook invocation's chunk, shape matches `backward_collector_fn`'s
        # signature.  `_teardown` concatenates + reshapes the lists into the
        # final (N, O, k) layout.
        self.activations: Dict[str, List[Float[Array, "N I"]]] = {}
        self.gradients: Dict[str, List[Float[Array, "N O"]]] = {}

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
    ) -> DataActivationsGradients:
        """
        Run the model with hooks to collect activations and gradients.
        """
        if try_load and save_directory is not None:
            try:
                logger.info(f"Trying to load collected data from: {save_directory}")
                return self.load(save_directory)
            except Exception as e:
                logger.warning(
                    f"Failed to load collected data from {save_directory}: {e}. "
                    f"Proceeding to collect data."
                )

        collection_dataset, metadata = self._generate_pseudo_targets(dataset, rng_key)

        self._run_collection_loop(
            collection_dataset.inputs,
            collection_dataset.targets,
            batch_size,
            metadata,
        )

        collected_data = self._teardown(metadata)

        if save_directory is not None:
            logger.info(f"Saving collected data to: {save_directory}")
            self.save(save_directory, collected_data)

        return collected_data

    # ------------------------------------------------------------------
    # Hook target: called by layer_wrapper_vjp during the backward pass.
    # ------------------------------------------------------------------

    def backward_collector_fn(
        self, name: str, a: Float[Array, "N I"], g: Float[Array, "N O"]
    ) -> None:
        self.activations.setdefault(name, []).append(a)
        self.gradients.setdefault(name, []).append(g)

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

        activations_concat = {
            layer: jnp.concatenate(v, axis=0) for layer, v in self.activations.items()
        }
        gradients_concat = {
            layer: jnp.concatenate(v, axis=0) for layer, v in self.gradients.items()
        }

        activations_final: Dict[str, Array] = {}
        gradients_final: Dict[str, Array] = {}
        for layer_name in activations_concat.keys():
            act = activations_concat[layer_name]  # (k*N, I)
            grad = gradients_concat[layer_name]  # (k*N, O)

            activations_final[layer_name] = act[:n_samples]

            O_dim = grad.shape[-1] if grad.ndim > 1 else 1
            # (k*N, O) -> (k, N, O) -> (N, O, k)
            gradients_final[layer_name] = grad.reshape(k, n_samples, O_dim).transpose(
                1, 2, 0
            )

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

    def save(self, directory: str, data: DataActivationsGradients) -> None:
        if os.path.isfile(directory):
            logger.warning(
                f"Provided save_path {directory} is a file. Using its directory instead."
            )
            directory = os.path.dirname(directory)
        os.makedirs(directory, exist_ok=True)

        file_path = os.path.join(directory, self.FILENAME)
        data_dict = {
            **{f"activations_{k}": np.array(v) for k, v in data.activations.items()},
            **{f"gradients_{k}": np.array(v) for k, v in data.gradients.items()},
            "layer_names": np.array(",".join(data.layer_names), dtype="U"),
            "probs": np.array(data.probs),
            "pseudo_target_strategy": np.array(
                self.pseudo_target_strategy.value, dtype="U"
            ),
            "pseudo_target_repetitions": np.array(self.pseudo_target_repetitions),
        }
        np.savez_compressed(file=file_path, allow_pickle=False, **data_dict)
        logger.info(f"Saved collected data to: {file_path}")

    @staticmethod
    def load(directory: str) -> DataActivationsGradients:
        load_path = f"{directory}/{CollectorActivationsGradients.FILENAME}"
        try:
            loaded = np.load(load_path)
            assert isinstance(loaded, np.lib.npyio.NpzFile)
        except FileNotFoundError:
            raise ValueError(f"File not found: {load_path}")

        layer_names = loaded["layer_names"].item().split(",")
        activations = {name: jnp.array(loaded[f"activations_{name}"]) for name in layer_names}
        gradients = {name: jnp.array(loaded[f"gradients_{name}"]) for name in layer_names}
        probs = jnp.array(loaded["probs"])
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
