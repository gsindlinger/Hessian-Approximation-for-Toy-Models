import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

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

T = TypeVar("T")


@dataclass
class CollectorBase(ABC, Generic[T]):
    """
    Base class for collecting activations and gradients via hooks.

    Subclasses must implement:
    - backward_collector_fn: Called during backward pass
    - teardown: Convert collected data to final format
    - save/load: Persistence

    Data Ordering Convention:
    All strategies use unified ordering: [all samples for k=0, all samples for k=1, ...]
    - Inputs:  [x0,x1,...,xN-1, x0,x1,...,xN-1, ...] (k repetitions)
    - Targets: [t0,t1,...,tN-1, t0,t1,...,tN-1, ...] (targets vary by strategy)

    This allows teardown to use a single reshape: (k*N, ...) -> (k, N, ...)
    """

    model: ApproximationModel
    params: Dict
    loss_fn: Callable

    # Pseudo-target generation config
    pseudo_target_strategy: PseudoTargetGenerationStrategy = (
        PseudoTargetGenerationStrategy.MCMC
    )
    pseudo_target_repetitions: int = 5

    def collect(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        save_directory: Optional[str] = None,
        try_load: bool = False,
        rng_key: Optional[Array] = None,
        **kwargs,
    ) -> T:
        """
        Run the model with hooks to collect activations and gradients.

        Args:
            dataset: Dataset with inputs and targets
            batch_size: Batch size for collection.
            save_directory: Optional path to save collected data.
            try_load: Whether to try loading cached data.
            rng_key: Random key for pseudo-target generation (required for MCMC)
            **kwargs: Additional arguments for specific collectors.

        Returns:
            Collected data from teardown().
        """
        # Try loading previously collected data if specified
        if try_load and save_directory is not None:
            try:
                logger.info(
                    f"Trying to load previously collected data from: {save_directory}"
                )
                return self.load(save_directory)
            except Exception as e:
                logger.warning(
                    f"Failed to load collected data from {save_directory}: {e}. "
                    f"Proceeding to collect data."
                )

        # Generate pseudo-targets based on strategy and get expanded dataset with correct amount of repetitions
        collection_dataset, metadata = self._generate_pseudo_targets(
            dataset, rng_key, **kwargs
        )

        # Run collection loop
        self._run_collection_loop(
            collection_dataset.inputs, collection_dataset.targets, batch_size, metadata
        )

        # Get collected data
        collected_data = self.teardown(metadata)

        if save_directory is not None:
            logger.info(f"Saving collected data to: {save_directory}")
            self.save(save_directory, collected_data)

        return collected_data

    def _generate_pseudo_targets(
        self, dataset: Dataset, rng_key: Optional[Array] = None, **kwargs
    ) -> Tuple[Dataset, Dict[str, Any]]:
        """
        Generate pseudo-targets based on the configured strategy.

        All strategies produce unified data ordering:
        - Inputs:  [x0,x1,...,xN-1, x0,x1,...,xN-1, ...] (k repetitions)
        - Targets: [t0,t1,...,tN-1, t0,t1,...,tN-1, ...] (targets vary by strategy)

        Returns:
            (expanded_dataset, metadata)
        """
        inputs = dataset.inputs
        n_samples = inputs.shape[0]

        if (
            self.pseudo_target_strategy
            == PseudoTargetGenerationStrategy.EMPIRICAL_FISHER
        ):
            # Use ground truth labels directly (k=1, no expansion needed)
            logger.info("[PSEUDO-TARGETS] Using EMPIRICAL_FISHER (ground truth labels)")
            metadata = {
                "strategy": PseudoTargetGenerationStrategy.EMPIRICAL_FISHER,
                "n_samples": n_samples,
                "k": 1,
                "total_samples": n_samples,
            }
            return dataset, metadata

        elif self.pseudo_target_strategy == PseudoTargetGenerationStrategy.MCMC:
            # Sample pseudo-targets multiple times
            if rng_key is None:
                raise ValueError(
                    "rng_key is required for MCMC pseudo-target generation"
                )

            k = self.pseudo_target_repetitions
            logger.info(f"[PSEUDO-TARGETS] Using MCMC with {k} repetitions")

            # Generate k different pseudo-target sets
            # Ordering: [all samples for rep 0, all samples for rep 1, ...]
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

            # Stack: (k*N, ...)
            expanded_inputs = jnp.concatenate(expanded_inputs_list, axis=0)
            expanded_targets = jnp.concatenate(expanded_targets_list, axis=0)

            metadata = {
                "strategy": PseudoTargetGenerationStrategy.MCMC,
                "n_samples": n_samples,
                "k": k,
                "total_samples": n_samples * k,
            }

            return Dataset(expanded_inputs, expanded_targets), metadata

        elif self.pseudo_target_strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            # Generate targets for all classes
            if not hasattr(self.model, "output_dim"):
                raise ValueError(
                    "ALL_CLASSES strategy requires model to have 'output_dim' attribute"
                )

            k = self.model.output_dim
            logger.info(f"[PSEUDO-TARGETS] Using ALL_CLASSES with {k} classes")

            # Compute probabilities for weighting later
            logits = jax.vmap(lambda x: self.model.apply(self.params, x))(inputs)
            assert isinstance(logits, jnp.ndarray), (
                "Model predictions must be a jnp.ndarray"
            )
            probabilities = jax.nn.softmax(logits, axis=-1)  # (N, K)

            # Use unified ordering: [all samples for class 0, all samples for class 1, ...]
            # Inputs: tile to repeat all samples k times
            expanded_inputs = jnp.tile(inputs, (k, 1))  # (K*N, features)

            # Targets: [0,0,...,0 (N times), 1,1,...,1 (N times), ..., K-1,K-1,...,K-1 (N times)]
            expanded_targets = jnp.repeat(jnp.arange(k), n_samples)  # (K*N,)

            metadata = {
                "strategy": PseudoTargetGenerationStrategy.ALL_CLASSES,
                "n_samples": n_samples,
                "k": k,
                "total_samples": n_samples * k,
                "probabilities": probabilities,  # (N, K) for FIM weighting
            }

            return Dataset(expanded_inputs, expanded_targets), metadata

        else:
            raise ValueError(
                f"Unknown pseudo-target strategy: {self.pseudo_target_strategy}"
            )

    def _run_collection_loop(
        self,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        batch_size: Optional[int],
        metadata: Dict[str, Any],
    ):
        """
        Run the collection loop over batches.

        Args:
            inputs: Prepared input data
            targets: Prepared target data
            batch_size: Batch size for DataLoader
            metadata: Additional metadata from preparation step
        """

        def loss_fn_for_grad(p, inputs, targets):
            predictions = self.model.apply(
                p,
                inputs,
                self,
                method=self.model.collector_apply,
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

        logger.info(f"Start collecting data for Collector: {self.__class__.__name__}")

        self._log_collection_start(metadata)

        for batch_data, batch_targets in dataloader:
            _ = jax.value_and_grad(loss_fn_for_grad)(
                self.params, batch_data, batch_targets
            )

        logger.info(
            f"Finished collecting data for Collector: {self.__class__.__name__}"
        )

    def _log_collection_start(self, metadata: Dict[str, Any]):
        """Log collection start with strategy-specific info."""
        strategy = metadata["strategy"]
        n_samples = metadata["n_samples"]
        k = metadata["k"]
        total_samples = metadata["total_samples"]

        strategy_name = strategy.value if hasattr(strategy, "value") else str(strategy)

        logger.info(
            f"[COLLECTION] Strategy: {strategy_name.upper()}, "
            f"Original samples: {n_samples}, "
            f"Repetitions/Classes (k): {k}, "
            f"Total samples to collect: {total_samples}"
        )

    @abstractmethod
    def backward_collector_fn(self, name: str, a: jnp.ndarray, g: jnp.ndarray):
        """Function to be called during backward pass to collect information."""
        pass

    @abstractmethod
    def teardown(self, metadata: Dict[str, Any]) -> T:
        """
        Function to be called after all data has been collected.

        Args:
            metadata: Metadata from pseudo-target generation

        Returns:
            Collected data in final format
        """
        pass

    @abstractmethod
    def save(self, save_path: str, data: Any):
        """Save collected data to the specified path."""
        pass

    @staticmethod
    @abstractmethod
    def load(load_path: str) -> T:
        """Load collected data from the specified path."""
        pass


@dataclass
class CollectorActivationsGradients(CollectorBase[DataActivationsGradients]):
    """
    Unified collector for storing layer-wise activations and output gradients.

    This collector supports all pseudo-target generation strategies:
    - EMPIRICAL_FISHER: Single pass with ground truth labels (k = 1)
    - MCMC: Multiple repetitions with sampled targets (k = repetitions)
    - ALL_CLASSES: One pass for each class (k = num_classes, only for classification)

    Data is collected with unified ordering: [all samples for k=0, all samples for k=1, ...]

    Output shapes:
    - activations: (N, I) - same activations for all k (first N samples)
    - gradients: (k, N, O) - reshaped from collected (k*N, O)
    - probabilities: (N, K) for ALL_CLASSES, None otherwise

    We store gradients of the log-likelihood:
        ∂ log p(y | x) / ∂z
    """

    activations: Dict[str, List[Float[Array, "N I"]]] = field(default_factory=dict)
    gradients: Dict[str, List[Float[Array, "N O"]]] = field(default_factory=dict)

    FILENAME: str = "collected_activations_gradients.npz"

    def __post_init__(self):
        """Validate configuration."""
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
            if self.pseudo_target_repetitions != self.model.output_dim:
                logger.warning(
                    "ALL_CLASSES strategy uses num_classes as k, "
                    "ignoring repetitions parameter."
                )

    def backward_collector_fn(
        self, name: str, a: Float[Array, "N I"], g: Float[Array, "N O"]
    ):
        """Store per-layer activations and log-likelihood gradients."""
        if name not in self.activations:
            self.activations[name] = [a]
            self.gradients[name] = [g]
        else:
            self.activations[name].append(a)
            self.gradients[name].append(g)

    def teardown(self, metadata: Dict[str, Any]) -> DataActivationsGradients:
        """
        Concatenate and reshape collected data appropriately.

        All strategies use unified data ordering:
        - Collected data: [all samples for k=0, all samples for k=1, ...]
        - Shape: (k*N, ...)

        Returns:
        - activations: (N, I) - first N samples (identical across all k)
        - gradients: (k, N, O) - reshaped from (k*N, O)
        - probabilities: (N, K) for ALL_CLASSES, None otherwise

        Args:
            metadata: Metadata from pseudo-target generation containing strategy info
        """
        strategy = metadata["strategy"]
        if isinstance(strategy, str):
            strategy = PseudoTargetGenerationStrategy(strategy)

        n_samples = metadata["n_samples"]
        k = metadata["k"]

        # Concatenate all batches: (k*N, ...)
        activations_concat = {
            layer: jnp.concatenate(v, axis=0) for layer, v in self.activations.items()
        }
        gradients_concat = {
            layer: jnp.concatenate(v, axis=0) for layer, v in self.gradients.items()
        }

        activations_final = {}
        gradients_reshaped = {}

        for layer_name in activations_concat.keys():
            act = activations_concat[layer_name]  # (k*N, I)
            grad = gradients_concat[layer_name]  # (k*N, O)

            O_dim = grad.shape[-1] if grad.ndim > 1 else 1

            # Activations: take first N (identical across all k repetitions)
            activations_final[layer_name] = act[:n_samples]

            # Gradients: reshape from (k*N, O) to (k, N, O)
            if grad.ndim > 1:
                gradients_reshaped[layer_name] = grad.reshape(k, n_samples, O_dim)
            else:
                gradients_reshaped[layer_name] = grad.reshape(k, n_samples)

        logger.info(
            f"[TEARDOWN] {strategy.value.upper()}: "
            f"activations shape (N={n_samples}, ...), "
            f"gradients shape (k={k}, N={n_samples}, ...)"
        )

        # Store probabilities if using ALL_CLASSES
        probabilities = None
        if (
            strategy == PseudoTargetGenerationStrategy.ALL_CLASSES
            and "probabilities" in metadata
        ):
            probabilities = metadata["probabilities"]  # (N, K)

        return DataActivationsGradients(
            activations=activations_final,
            gradients=gradients_reshaped,
            layer_names=self.model.get_layer_names(),
            probabilities=probabilities,
            pseudo_target_strategy=strategy,
        )

    def save(self, directory: str, data: DataActivationsGradients):
        """Save collected activations and gradients with metadata."""
        if os.path.isfile(directory):
            logger.warning(
                f"Provided save_path {directory} is a file. Using its directory instead."
            )
            directory = os.path.dirname(directory)

        if not os.path.exists(directory):
            logger.info(f"Directory {directory} does not exist. Creating it.")
            os.makedirs(directory)

        file_path = os.path.join(directory, self.FILENAME)

        activations, gradients, layer_names = (
            data.activations,
            data.gradients,
            data.layer_names,
        )

        # Build data dict with strategy metadata
        data_dict = {
            **{f"activations_{k}": np.array(v) for k, v in activations.items()},
            **{f"gradients_{k}": np.array(v) for k, v in gradients.items()},
            "layer_names": np.array(",".join(layer_names), dtype="U"),
            "pseudo_target_strategy": np.array(
                self.pseudo_target_strategy.value, dtype="U"
            ),
            "pseudo_target_repetitions": np.array(self.pseudo_target_repetitions),
        }

        # Add probabilities if present (for ALL_CLASSES strategy)
        if data.probabilities is not None:
            data_dict["probabilities"] = np.array(data.probabilities)

        np.savez_compressed(file=file_path, allow_pickle=False, **data_dict)
        logger.info(f"Saved collected data to: {file_path}")

    @staticmethod
    def load(directory: str) -> DataActivationsGradients:
        """Load collected activations and gradients with metadata."""
        load_path = f"{directory}/{CollectorActivationsGradients.FILENAME}"
        try:
            loaded = np.load(load_path)
            assert isinstance(loaded, np.lib.npyio.NpzFile)
        except FileNotFoundError:
            raise ValueError(f"File not found: {load_path}")

        activations = {}
        gradients = {}
        layer_names = loaded["layer_names"].item().split(",")

        # Load strategy metadata if present
        strategy_str = loaded.get("pseudo_target_strategy", None)
        if strategy_str is not None:
            strategy_str = strategy_str.item()
            logger.info(f"Loaded data collected with strategy: {strategy_str}")

        # Load activations and gradients
        for k in loaded.files:
            if k.startswith("activations_"):
                layer_name = k[len("activations_") :]
                activations[layer_name] = jnp.array(loaded[k])
            elif k.startswith("gradients_"):
                layer_name = k[len("gradients_") :]
                gradients[layer_name] = jnp.array(loaded[k])

        # Convert strategy string back to enum if present
        pseudo_target_strategy = PseudoTargetGenerationStrategy.MCMC  # default
        if strategy_str is not None:
            pseudo_target_strategy = PseudoTargetGenerationStrategy(strategy_str)

        # Load probabilities if present (for ALL_CLASSES strategy)
        probabilities = None
        if "probabilities" in loaded.files:
            probabilities = jnp.array(loaded["probabilities"])

        return DataActivationsGradients(
            activations=activations,
            gradients=gradients,
            layer_names=layer_names,
            probabilities=probabilities,
            pseudo_target_strategy=pseudo_target_strategy,
        )


### Hooks for collecting activations and gradients of specific layers
@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def layer_wrapper_vjp(
    pure_apply_fn: Callable,
    params: Dict,
    x: jnp.ndarray,
    name: str,
    collector: CollectorBase,
):
    """Custom VJP wrapper for capturing activations and gradients."""
    return pure_apply_fn(params, x)


def layer_wrapper_fwd(
    pure_apply_fn: Callable,
    params: Dict,
    x: jnp.ndarray,
    name: str,
    collector: CollectorBase,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Dict]]:
    """Forward pass: compute output and save residuals for backward pass."""
    output = pure_apply_fn(params, x)
    residuals = (x, params)
    return output, residuals


def layer_wrapper_bwd(
    pure_apply_fn: Callable,
    name: str,
    collector: CollectorBase,
    residuals: Tuple[jnp.ndarray, Dict],
    g: jnp.ndarray,
) -> Tuple[Dict, jnp.ndarray]:
    """Backward pass: capture data and compute gradients."""
    a, params = residuals
    collector.backward_collector_fn(name, a, g)
    _primals, vjp_fn = jax.vjp(pure_apply_fn, params, a)
    param_grads, input_grads = vjp_fn(g)
    return (param_grads, input_grads)


layer_wrapper_vjp.defvjp(layer_wrapper_fwd, layer_wrapper_bwd)
