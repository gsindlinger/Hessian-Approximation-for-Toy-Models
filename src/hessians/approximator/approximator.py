from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Optional,
    Tuple,
    Type,
    get_args,
    get_origin,
)

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from src.config import Config
from src.hessians.utils.data import ApproximationData, EKFACData
from src.utils.data.jax_dataloader import JAXDataLoader

logger = logging.getLogger(__name__)


@dataclass
class ApproximatorBase(ABC):
    """Base class providing the structure to compute and save/load components for the Hessian approximation."""

    collected_data_path: str
    """Path to previously collected data"""

    def build(
        self, config: Config, save_directory: Optional[str]
    ) -> Tuple[ApproximationData, Config]:
        """Build the Hessian approximation by computing the relevant components. Optionally saves the components and config to the specified directory."""

        logger.info(f"Start computing components for: {self.__class__.__name__}")
        data = self._build()
        logger.info(f"Finished computing components for: {self.__class__.__name__}")

        if save_directory is not None:
            self.save_data(directory=save_directory, data=data, config=config)
            logger.info(f"Saved Hessian approximation to directory: {save_directory}")

        return data, config

    @abstractmethod
    def _build(self) -> ApproximationData:
        """Method which computes the relevant components for the Hessian approximation.
        Each subclass must implement this method to populate self.data accordingly."""
        pass

    def save_data(
        self, directory: str, data: ApproximationData, config: Config
    ) -> None:
        """Saves the components data and corresponding config to the specified directory."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(save_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=4)

        arrays = {}

        assert is_dataclass(data), "Data must be a dataclass instance."
        for field_t in fields(data):
            value = getattr(data, field_t.name)
            field_type = field_t.type

            # Dict[str, array]
            if get_origin(field_type) is dict:
                key_type, _ = get_args(field_type)
                if key_type is str:
                    for k, arr in value.items():
                        arrays[f"{field_t.name}/{k}"] = np.asarray(arr)
                    continue

            arrays[field_t.name] = np.asarray(value)

        np.savez_compressed(save_dir / "data.npz", **arrays)

    @classmethod
    def get_data_type(cls) -> Type[ApproximationData]:
        """Returns a mapping from subclass names to their corresponding data class types."""
        mapping = {
            "EKFACApproximator": EKFACData,
        }

        data_type = mapping.get(cls.__name__)
        if data_type is None:
            raise ValueError(f"No data type mapping found for {cls.__name__}")
        return data_type

    @classmethod
    def load_data(
        cls: Type[ApproximatorBase], directory: str
    ) -> Tuple[ApproximationData, Config]:
        """Loads the components data and corresponding config from the specified directory."""
        load_dir = Path(directory)

        # Load config
        with open(load_dir / "config.json") as f:
            config_dict = json.load(f)
        config = Config(**config_dict)

        # Load arrays
        loaded = np.load(load_dir / "data.npz")

        # Determine the data class from the subclass annotation
        data_t = cls.get_data_type()

        if not is_dataclass(data_t):
            raise TypeError("The subclass must annotate `data` with a dataclass type")

        data_kwargs = {}

        for field_t in fields(data_t):
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

            # Scalar / array
            if field_t.name in loaded:
                arr = loaded[field_t.name]
                if arr.shape == ():
                    data_kwargs[field_t.name] = arr.item()
                else:
                    data_kwargs[field_t.name] = jnp.array(arr)

        data_obj = data_t(**data_kwargs)
        return data_obj, config

    @staticmethod
    def compute_eigenvectors_and_eigenvalues(
        activations_covariances: Dict[str, jnp.ndarray],
        gradients_covariances: Dict[str, jnp.ndarray],
    ) -> Tuple[
        Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
    ]:
        """Compute eigenvectors of the covariance matrices A and G for each layer."""

        activation_eigvals = {}
        activation_eigvecs = {}
        gradient_eigvals = {}
        gradient_eigvecs = {}

        for layer_name in activations_covariances.keys():
            (
                activation_eigvals_layer,
                gradient_eigvals_layer,
                activation_eigvecs_layer,
                gradient_eigvecs_layer,
            ) = ApproximatorBase.compute_layer_eigenvectors(
                activations_covariances[layer_name],
                gradients_covariances[layer_name],
            )

            activation_eigvals[layer_name] = activation_eigvals_layer
            activation_eigvecs[layer_name] = activation_eigvecs_layer
            gradient_eigvals[layer_name] = gradient_eigvals_layer
            gradient_eigvecs[layer_name] = gradient_eigvecs_layer

        return (activation_eigvecs, gradient_eigvecs), (
            activation_eigvals,
            gradient_eigvals,
        )

    @staticmethod
    @jax.jit
    def compute_layer_eigenvectors(
        A: Float[Array, "I I"], G: Float[Array, "O O"]
    ) -> Tuple[
        Float[Array, "I"], Float[Array, "O"], Float[Array, "I I"], Float[Array, "O O"]
    ]:
        """
        Compute eigenvectors of covariance matrices A and G for a single layer.

        Returns:
            Tuple containing:
            - Eigenvalues of A (Float[Array, "I"])
            - Eigenvalues of G (Float[Array, "O"])
            - Eigenvectors of A (Float[Array, "I I"])
            - Eigenvectors of G (Float[Array, "O O"])
        """
        # Ensure numerical stability by using float64 for eigen decomposition
        jax.config.update("jax_enable_x64", True)
        A = A.astype(jnp.float64)
        G = G.astype(jnp.float64)

        eigenvals_A: Float[Array, "I"]
        eigvecs_A: Float[Array, "I I"]
        eigenvals_G: Float[Array, "O"]
        eigvecs_G: Float[Array, "O O"]

        eigenvals_A, eigvecs_A = jnp.linalg.eigh(A)
        eigenvals_G, eigvecs_G = jnp.linalg.eigh(G)
        jax.config.update("jax_enable_x64", False)

        return (
            eigenvals_A.astype(jnp.float32),
            eigenvals_G.astype(jnp.float32),
            eigvecs_A.astype(jnp.float32),
            eigvecs_G.astype(jnp.float32),
        )

    @staticmethod
    def batched_collector_processing(
        layer_keys: Iterable[str],
        num_samples: int,
        compute_fn: Callable,
        data: Dict[str, Dict[str, Float[Array, "..."]]],
        static_data: Optional[Dict[str, Dict[str, Float[Array, "..."]]]] = None,
        normalize: bool = True,
    ) -> Dict[str, Float[Array, "..."]]:
        """Process collected data (e.g. activations, gradients) with optional additional
        information (e.g. precomputed eigenvectors) in batches to avoid memory issues.

        Since the method should be rather generic, the data is provided as dict with layer names as keys.
        The compute_fn method must accept keyword arguments matching the keys of the data dict.
        """
        batch_size = JAXDataLoader.get_batch_size()
        num_batches = (num_samples + batch_size - 1) // batch_size
        accumulator = {}

        # Loop over batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            # Loop over layer names (extract data for item in data dict)
            for layer_name in layer_keys:
                # Extract batch data for this layer
                data_batch = {
                    key: data[key][layer_name][start_idx:end_idx] for key in data
                }
                # Extract static data for this layer if provided
                static_layer_data = (
                    {
                        static_data_key: static_data[static_data_key][layer_name]
                        for static_data_key in static_data
                    }
                    if static_data is not None
                    else {}
                )

                # Compute result for this batch
                batch_result = compute_fn(**data_batch, **static_layer_data)

                # Accumulate results
                if batch_idx == 0:
                    accumulator[layer_name] = batch_result
                else:
                    accumulator[layer_name] += batch_result

        # Normalize by number of samples if requested
        if normalize:
            for layer_name in layer_keys:
                accumulator[layer_name] /= num_samples

        return accumulator

    @staticmethod
    def compute_eigenvalue_corrections(
        activations: Dict[str, Float[Array, "N I"]],
        gradients: Dict[str, Float[Array, "N O"]],
        activation_eigenvectors: Dict[str, Float[Array, "I I"]],
        gradient_eigenvectors: Dict[str, Float[Array, "O O"]],
    ) -> Dict[str, Float[Array, "I O"]]:
        """
        Compute eigenvalue corrections for each layer.
        """
        return ApproximatorBase.batched_collector_processing(
            layer_keys=activations.keys(),
            num_samples=list(activations.values())[0].shape[0],
            compute_fn=ApproximatorBase._compute_eigenvalue_correction_batch,
            data={
                "activations": activations,
                "gradients": gradients,
            },
            static_data={
                "Q_A": activation_eigenvectors,
                "Q_G": gradient_eigenvectors,
            },
        )

    @staticmethod
    @jax.jit
    def _compute_eigenvalue_correction_batch(
        Q_A: Float[Array, "I I"],
        Q_G: Float[Array, "O O"],
        activations: Float[Array, "N I"],
        gradients: Float[Array, "N O"],
    ) -> Float[Array, "I O"]:
        r"""
        Compute eigenvalue correction for a given layer.

        For each sample n, we compute:
            (Q_G \otimes Q_A)^T vec(a_n g_n^T) = (Q_G \otimes Q_A)^T (g_n \otimes a_n)

        Using the Kronecker product property (A \otimes B)^T = A^T \otimes B^T and
        the mixed-product property, this simplifies to:
            (Q_G^T g_n) \otimes (Q_A^T a_n) = vec((Q_A^T a_n) (Q_G^T g_n)^T)

        where:
        - Q_A, Q_G are the eigenvector matrices of the activation and gradient covariances
        - a_n, g_n are the activation and pre-activation gradient vectors for sample n
        - \otimes denotes the Kronecker product

        Implementation steps:
        1. Transform activations to eigenbasis: a_tilde_n = Q_A^T @ a_n
        2. Transform gradients to eigenbasis: g_tilde_n = Q_G^T @ g_n
        3. Compute outer product / Kronecker product: a_tilde_n \otimes g_tilde_n
        4. Square and sum across samples (averaging is later done by the caller)

        Note:
        The paper of Grosse et al. (2023) misses the transpose of the eigenvector
        basis (Q_A \otimes Q_G) in Equation (20). Refer to George et al. (2018) for the
        correct formulation.

        In this JAX implementation, we have to swap the Kronecker order to
        (Q_G \otimes Q_A) because JAX layers store weights with shape [input_dim, output_dim],
        unlike the [output_dim, input_dim] convention used in the original paper.

        Furthermore, we don't store the vectorized version of the outer product, but rather the
        matrix form. Note, that the resulting matrix represents the row-major flattening
        of the original EKFAC formulation due to the JAX weight layout convention.
        """
        g_tilde = jnp.einsum("op, np -> no", Q_G.T, gradients)
        a_tilde = jnp.einsum("ij, nj -> ni", Q_A.T, activations)
        outer = jnp.einsum("ni, no -> nio", a_tilde, g_tilde)
        return (outer**2).sum(axis=0)
