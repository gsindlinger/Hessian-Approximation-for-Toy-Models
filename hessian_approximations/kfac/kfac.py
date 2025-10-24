from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag
from typing_extensions import override

from config.config import KFACConfig
from hessian_approximations.hessian_approximations import HessianApproximation
from hessian_approximations.kfac.activation_gradient_collector import (
    ActivationGradientCollector,
)
from hessian_approximations.kfac.layer_components import LayerComponents
from models.loss import get_loss_name
from models.train import ApproximationModel


class KFAC(HessianApproximation):
    """
    Eigenvalue-corrected Kronecker-Factored Approximate Curvature (EKFAC).

    Provides a structured approximation to the Fisher Information Matrix
    using Kronecker-factored covariance matrices with optional eigenvalue corrections.
    """

    def __init__(
        self,
        model_name: str = "model",
        config: KFACConfig | None = None,
    ):
        super().__init__()
        self.config = config or KFACConfig()
        self.storage = KFACStorage(model_name)
        self.collector = ActivationGradientCollector()

        # Core components
        self.covariances = LayerComponents()
        self.eigenvectors = LayerComponents()
        self.eigenvalue_corrections: Dict[str, jnp.ndarray] = {}

    def get_sample_size(self) -> int:
        """Get number of samples used in the collected data."""
        if not self.collector.captured_data:
            raise ValueError("No captured data. Run a forward-backward pass first.")

        first_layer = next(iter(self.collector.captured_data.values()))
        activations, _ = first_layer
        return activations.shape[0]

    @override
    def compute_hessian(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
    ) -> jnp.ndarray:
        """
        Compute full Hessian approximation.

        Not practical for large models but useful for testing and comparison
        with the true Hessian.
        """
        if not self.covariances:
            self.generate_ekfac_components(
                model, params, training_data, training_targets, loss_fn
            )

        hessian_layers = self._compute_layer_hessians(params)
        # Combine layer Hessians into full block-diagonal Hessian
        full_hessian = block_diag(*[jnp.asarray(H) for H in hessian_layers.values()])
        return jnp.array(full_hessian)

    @override
    def compute_hvp(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Hessian-vector product."""
        raise NotImplementedError("HVP computation not implemented yet for EKFAC")

    @override
    def compute_ihvp(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jax.Array,
        training_targets: jax.Array,
        loss_fn: Callable[..., Any],
        vector: jax.Array,
    ) -> jax.Array:
        """Compute inverse Hessian-vector product."""
        raise NotImplementedError("IHVP computation not implemented yet for EKFAC")

    def generate_ekfac_components(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
    ) -> None:
        """
        Compute all EKFAC components in sequence.

        Steps:
        1. Compute covariances from activations and gradients
        2. Compute eigenvectors of covariance matrices
        3. Optionally compute eigenvalue corrections

        If config.reload_data is False, attempts to load from disk first.
        """
        if not self.config.reload_data:
            try:
                self._load_from_disk(model)
                return
            except FileNotFoundError:
                print("No saved EKFAC components found. Computing from scratch.")

        self._compute_from_scratch(
            model, params, training_data, training_targets, loss_fn
        )

    def _load_from_disk(self, model: ApproximationModel) -> None:
        """Load all (E)KFAC components from disk."""
        self.load_covariances_from_disk()
        if self.config.use_eigenvalue_correction:
            self.load_eigenvectors_from_disk()
            self.load_eigenvalue_corrections_from_disk()

    def _compute_from_scratch(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
    ) -> None:
        """Compute all (E)KFAC components from scratch."""
        self.compute_covariances(
            model, params, training_data, training_targets, loss_fn
        )
        if self.config.use_eigenvalue_correction:
            self.compute_eigenvectors()
            self.compute_eigenvalue_corrections(
                model, params, training_data, training_targets, loss_fn
            )

    def compute_covariances(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        targets: jnp.ndarray,
        loss_fn: Callable,
    ):
        """
        Compute A and G covariance matrices for each layer.
        Performs a forward-backward pass to collect activations and gradients,
        then computes their covariance matrices.
        """

        self._process_multiple_batches_collector(
            model,
            params,
            training_data,
            targets,
            loss_fn,
            compute_method="covariance",
        )
        self.storage.save_covariances(self.covariances)

    def compute_eigenvectors(self) -> None:
        """Compute eigenvectors of the covariance matrices A and G for each layer."""
        if not self.covariances:
            raise ValueError(
                "Covariances not computed yet. Run compute_covariances first."
            )

        activation_eigvecs = {}
        gradient_eigvecs = {}

        for layer_name in self.covariances.activations.keys():
            A = self.covariances.activations[layer_name]
            G = self.covariances.gradients[layer_name]

            # Ensure numerical stability by using float64 for eigen decomposition
            A = A.astype(jnp.float64)
            G = G.astype(jnp.float64)

            _, eigvecs_A = jnp.linalg.eigh(A)
            _, eigvecs_G = jnp.linalg.eigh(G)

            activation_eigvecs[layer_name] = eigvecs_A.astype(jnp.float32)
            gradient_eigvecs[layer_name] = eigvecs_G.astype(jnp.float32)

        self.eigenvectors = LayerComponents(activation_eigvecs, gradient_eigvecs)
        self.storage.save_eigenvectors(self.eigenvectors)

    def compute_eigenvalue_corrections(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
    ) -> None:
        """
        Compute eigenvalue corrections for each layer.

        Projects activations and gradients onto eigenbases and computes
        the empirical eigenvalue corrections as outer products.
        """
        if not self.covariances:
            raise ValueError(
                "Covariances not computed yet. Run compute_covariances first."
            )
        if not self.eigenvectors:
            raise ValueError(
                "Eigenvectors not computed yet. Run compute_eigenvectors_and_eigenvalues first."
            )

        self._process_multiple_batches_collector(
            model,
            params,
            training_data,
            training_targets,
            loss_fn,
            compute_method="eigenvalue_correction",
        )
        self.storage.save_eigenvalue_corrections(self.eigenvalue_corrections)

    def covariance(self, input: jnp.ndarray) -> jnp.ndarray:
        """
        Compute covariance matrix for given input.

        For each layer:
        - A = (1/N) * a^T @ a
        - G = (1/N) * g^T @ g
        """
        return input.mT @ input

    def eigenvalue_correction(
        self,
        layer_name: str,
        activations: jnp.ndarray,
        gradients: jnp.ndarray,
    ) -> None:
        """Compute eigenvalue correction for a given layer."""
        Q_A = self.eigenvectors.activations[layer_name]
        Q_G = self.eigenvectors.gradients[layer_name]

        g_tilde = jnp.einsum("oi, ni -> no", Q_G.T, gradients)  # [N, O]
        a_tilde = jnp.einsum("ij, nj -> ni", Q_A.T, activations)  # [N, I]

        # Compute outer product and average
        outer = jnp.einsum("no, ni -> noi", g_tilde, a_tilde)  # [N, O, I]
        correction = (outer**2).sum(
            axis=0
        )  # [O, I] (averaging is done by the calling method)

        self._accumulate_data(self.eigenvalue_corrections, layer_name, correction)

    def _process_multiple_batches_collector(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        targets: jnp.ndarray,
        loss_fn: Callable,
        compute_method: Literal["covariance", "eigenvalue_correction"],
    ):
        """
        Process data in batches to collect activations and gradients and apply some function to it.
        So far it is used to compute covariances of the activations and preactivation gradients, as well as computing eigenvalue corrections.
        """

        def loss_fn_for_grad(p, training_data, targets):
            predictions = model.apply(
                p, training_data, self.collector, method=model.kfac_apply
            )
            # Use sum reduction to avoid prematurely averaging gradients
            return loss_fn(predictions, targets, reduction="sum")

        # Optionally generate pseudo-targets for true Fisher computation
        if self.config.use_pseudo_targets:
            targets = self.generate_pseudo_targets(
                model, params, training_data, loss_fn
            )

        # Process all data in batches
        n_samples = training_data.shape[0]
        effective_batch_size = (
            self.config.batch_size if self.config.batch_size is not None else n_samples
        )

        for start in range(0, n_samples, effective_batch_size):
            end = min(start + effective_batch_size, n_samples)
            batch_data = training_data[start:end]
            batch_targets = targets[start:end]
            _ = jax.value_and_grad(loss_fn_for_grad)(params, batch_data, batch_targets)
            self._process_single_batch_collector(
                end - start,
                compute_method,
                model.use_bias,
            )

        if compute_method == "covariance":
            # Average covariances over all samples
            for layer_name in self.covariances.activations.keys():
                self.covariances.activations[layer_name] /= n_samples
                self.covariances.gradients[layer_name] /= n_samples
        if compute_method == "eigenvalue_correction":
            # Average eigenvalue corrections over all samples and add damping
            for layer_name in self.eigenvalue_corrections.keys():
                self.eigenvalue_corrections[layer_name] = (
                    self.eigenvalue_corrections[layer_name] / n_samples
                ) + jnp.ones_like(
                    self.eigenvalue_corrections[layer_name]
                ) * self.config.damping_lambda

    def _process_single_batch_collector(
        self,
        batch_size,
        compute_method: Literal["covariance", "eigenvalue_correction"],
        use_bias: bool = False,
    ):
        """
        Process a batch of activation and gradient data. Compute and accumulate their covariances / eigenvalue corrections.
        """
        for layer_name, (a, g) in self.collector.captured_data.items():
            if use_bias:
                batch_size = a.shape[0]
                a = jnp.concatenate([a, jnp.ones((batch_size, 1))], axis=1)

            if compute_method == "covariance":
                # Compute running covariance / eigenvalue correction by accumulation
                self._accumulate_covariances(layer_name, a, g)

            elif compute_method == "eigenvalue_correction":
                self._accumulate_eigenvalue_corrections(layer_name, a, g)
            else:
                raise ValueError(f"Unknown compute method: {compute_method}")

    def _accumulate_data(
        self, store: Dict, key: str, batch_covariance: jnp.ndarray
    ) -> None:
        if key in store:
            store[key] += batch_covariance
        else:
            store[key] = batch_covariance

    def _accumulate_covariances(
        self,
        layer_name: str,
        activations: jnp.ndarray,
        gradients: jnp.ndarray,
    ):
        """Accumulate covariance matrices for a given layer."""
        self._accumulate_data(
            self.covariances.activations,
            layer_name,
            self.covariance(activations),
        )
        self._accumulate_data(
            self.covariances.gradients,
            layer_name,
            self.covariance(gradients),
        )

    def _accumulate_eigenvalue_corrections(
        self,
        layer_name: str,
        activations: jnp.ndarray,
        gradients: jnp.ndarray,
    ):
        r"""
        Compute eigenvalue correction for a given layer.

        For each sample n, we compute:
        (Q_A \otimes Q_G)^T vec(g_n * a_n^T)

        Using the Kronecker product property (A \otimes B)^T = A^T \otimes B^T and the
        mixed-product property, this simplifies to:
            (Q_A^T a_n) \otimes (Q_G^T g_n)

        where:
        - Q_A, Q_G are the eigenvector matrices of activation and gradient covariances
        - a_n, g_n are the activation and gradient vectors for sample n
        - \otimes denotes the Kronecker product

        Implementation steps:
        1. Transform activations to eigenbasis: a_tilde_n = Q_A^T @ a_n
        2. Transform gradients to eigenbasis: g_tilde_n = Q_G^T @ g_n
        3. Compute outer product / Kronecker product: a_tilde_n \otimes g_tilde_n
        4. Square and sum across samples (averaging is later done by caller after summing over all batches)

        """
        Q_A = self.eigenvectors.activations[layer_name]
        Q_G = self.eigenvectors.gradients[layer_name]

        # Project activations and gradients onto eigenbases
        g_tilde = jnp.einsum("oi, ni -> no", Q_G.T, gradients)  # [N, O]
        a_tilde = jnp.einsum("ij, nj -> ni", Q_A.T, activations)  # [N, I]

        # Compute outer product and average
        outer = jnp.einsum("ni, no -> nio", a_tilde, g_tilde)  # [N, I, O]
        correction = (outer**2).sum(
            axis=0
        )  # [I, O] (averaging is done by the calling method)

        self._accumulate_data(self.eigenvalue_corrections, layer_name, correction)

    def _compute_layer_hessians(self, params: Any) -> Dict[str, jnp.ndarray]:
        """Compute Hessian approximation for each layer."""
        hessian_layers = {}

        for layer_name in params["params"].keys():
            if self.config.use_eigenvalue_correction:
                hessian_layers[layer_name] = (
                    self._compute_layer_hessian_with_correction(layer_name)
                )
            else:
                hessian_layers[layer_name] = self._compute_layer_hessian_kfac_only(
                    layer_name
                )

        return hessian_layers

    def _compute_layer_hessian_with_correction(self, layer_name: str) -> jnp.ndarray:
        """Compute layer Hessian with eigenvalue corrections (EKFAC)."""
        A_eigvecs = self.eigenvectors.activations[layer_name]
        G_eigvecs = self.eigenvectors.gradients[layer_name]
        corrections = self.eigenvalue_corrections[layer_name]

        Q_kron = jnp.kron(A_eigvecs, G_eigvecs)  # [I*O, I*O]
        H_layer = Q_kron @ jnp.diag(corrections.flatten()) @ Q_kron.T

        return H_layer

    def _compute_layer_hessian_kfac_only(self, layer_name: str) -> jnp.ndarray:
        """Compute layer Hessian without eigenvalue corrections (KFAC only)."""
        A = self.covariances.activations[layer_name]
        G = self.covariances.gradients[layer_name]
        return jnp.kron(A, G)

    def generate_pseudo_targets(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        loss_fn: Callable,
        rng_key: jax.Array | None = None,
    ) -> jnp.ndarray:
        """
        Generate pseudo-targets based on the model's output distribution.

        This is used to compute the true Fisher Information Matrix rather than
        the empirical Fisher (which would use true labels).

        Args:
            model: The model to use for predictions
            params: Model parameters
            training_data: Input data
            loss_fn: Loss function to determine target type
            rng_key: Random key for sampling (defaults to PRNGKey(0))

        Returns:
            Pseudo-targets sampled from the model's output distribution
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        loss_name = get_loss_name(loss_fn)
        if loss_name == "cross_entropy":
            return self._generate_classification_pseudo_targets(
                model, params, training_data, rng_key
            )
        elif loss_name == "mse":
            return self._generate_regression_pseudo_targets(
                model, params, training_data, rng_key
            )
        else:
            raise ValueError(f"Unsupported loss function for EKFAC: {loss_name}")

    def _generate_classification_pseudo_targets(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        rng_key: jax.Array,
    ) -> jnp.ndarray:
        """Generate pseudo-targets by sampling from softmax probabilities."""
        logits = model.apply(params, training_data)
        if not isinstance(logits, jnp.ndarray):
            raise ValueError(
                "Model predictions must be a jnp.ndarray for classification."
            )
        probs = jax.nn.softmax(logits, axis=-1)
        return jax.random.categorical(rng_key, jnp.log(probs), axis=-1)

    def _generate_regression_pseudo_targets(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        rng_key: jax.Array,
    ) -> jnp.ndarray:
        """Generate pseudo-targets by adding Gaussian noise to predictions."""
        preds = model.apply(params, training_data)

        if not isinstance(preds, jnp.ndarray):
            raise ValueError("Model predictions must be a jnp.ndarray for regression.")
        noise = self.config.pseudo_target_noise_std * jax.random.normal(
            rng_key, preds.shape
        )
        return preds + noise

    def load_covariances_from_disk(self) -> None:
        """Load previously computed covariances from disk."""
        self.covariances = self.storage.load_covariances()

    def load_eigenvectors_from_disk(self) -> None:
        """Load previously computed eigenvectors from disk."""
        self.eigenvectors = self.storage.load_eigenvectors()

    def load_eigenvalue_corrections_from_disk(self) -> None:
        """Load previously computed eigenvalue corrections from disk."""
        self.eigenvalue_corrections = self.storage.load_eigenvalue_corrections()


class KFACStorage:
    """Handles all disk I/O operations for (E)KFAC components using NumPy's npz format."""

    def __init__(self, model_name: str, base_path: str | Path = "data"):
        self.base_path = Path(base_path) / model_name
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, filename: str) -> Path:
        return self.base_path / filename

    def _save_layer_components(
        self, filename: str, components: LayerComponents
    ) -> None:
        path = self._get_path(filename)
        save_dict = {
            f"{prefix}_{name}": np.asarray(arr)
            for prefix, group in (
                ("activations", components.activations),
                ("gradients", components.gradients),
            )
            for name, arr in group.items()
        }
        np.savez_compressed(path, **save_dict)  # type: ignore

    def _load_layer_components(self, filename: str) -> LayerComponents:
        path = self._get_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"No file found at {path}")
        data = np.load(path, allow_pickle=False)
        activations, gradients = {}, {}
        for key in data.files:
            prefix, name = key.split("_", 1)
            (activations if prefix == "activations" else gradients)[name] = jnp.array(
                data[key]
            )
        return LayerComponents(activations, gradients)

    def save_covariances(self, covariances: LayerComponents) -> None:
        self._save_layer_components("covariances.npz", covariances)

    def load_covariances(self) -> LayerComponents:
        return self._load_layer_components("covariances.npz")

    def save_eigenvectors(self, eigenvectors: LayerComponents) -> None:
        self._save_layer_components("eigenvectors.npz", eigenvectors)

    def load_eigenvectors(self) -> LayerComponents:
        return self._load_layer_components("eigenvectors.npz")

    def save_eigenvalue_corrections(self, corrections: Dict[str, jnp.ndarray]) -> None:
        path = self._get_path("eigenvalue_corrections.npz")
        np.savez_compressed(
            path,
            **{name: np.asarray(arr) for name, arr in corrections.items()},  # type: ignore
        )

    def load_eigenvalue_corrections(self) -> Dict[str, jnp.ndarray]:
        path = self._get_path("eigenvalue_corrections.npz")
        if not path.exists():
            raise FileNotFoundError(f"No eigenvalue correction file found at {path}")
        data = np.load(path, allow_pickle=False)
        return {name: jnp.array(data[name]) for name in data.files}
