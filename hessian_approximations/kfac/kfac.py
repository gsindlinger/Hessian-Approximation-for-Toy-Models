from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
from scipy.linalg import block_diag
from typing_extensions import override

from config.config import KFACConfig
from hessian_approximations.hessian_approximations import HessianApproximation
from hessian_approximations.kfac.activation_gradient_collector import (
    ActivationGradientCollector,
    LayerComponents,
)
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
        self.covariances = LayerComponents({}, {})
        self.eigenvectors = LayerComponents({}, {})
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
        """Load all EKFAC components from disk."""
        self.load_covariances_from_disk(model)
        self.eigenvectors = self.storage.load_eigenvectors()
        if self.config.use_eigenvalue_correction:
            self.compute_eigenvalue_corrections()

    def _compute_from_scratch(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
    ) -> None:
        """Compute all EKFAC components from scratch."""
        self.compute_covariances(
            model, params, training_data, training_targets, loss_fn
        )
        self.compute_eigenvectors_and_eigenvalues()
        if self.config.use_eigenvalue_correction:
            self.compute_eigenvalue_corrections()

    def compute_covariances(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        targets: jnp.ndarray,
        loss_fn: Callable,
    ) -> LayerComponents:
        """
        Compute A and G covariance matrices for each layer.

        Performs a forward-backward pass to collect activations and gradients,
        then computes their covariance matrices.
        """

        def loss_fn_for_grad(p):
            predictions = model.apply(
                p, training_data, self.collector, method=model.kfac_apply
            )
            # Use sum reduction to avoid prematurely averaging gradients
            return loss_fn(predictions, targets, reduction="sum")

        _ = jax.value_and_grad(loss_fn_for_grad)(params)
        self.collector.save_to_disk(model)

        self.covariances = self._compute_covariances_from_collected_data(
            use_bias=model.use_bias
        )
        return self.covariances

    def _compute_covariances_from_collected_data(
        self, use_bias: bool = False
    ) -> LayerComponents:
        """
        Convert raw captured (activations, gradients) to A and G covariance matrices.

        For each layer:
        - A = (1/N) * a^T @ a  (optionally with bias term)
        - G = (1/N) * g^T @ g
        """
        if not self.collector.captured_data:
            raise ValueError("No captured data. Run a forward-backward pass first.")

        A_matrices = {}
        G_matrices = {}
        sample_size = self.get_sample_size()

        for layer_name, (a, g) in self.collector.captured_data.items():
            if use_bias:
                batch_size = a.shape[0]
                a = jnp.concatenate([a, jnp.ones((batch_size, 1))], axis=1)

            # Compute covariance matrices as expectations
            A_matrices[layer_name] = (a.mT @ a) / sample_size
            G_matrices[layer_name] = (g.mT @ g) / sample_size

        return LayerComponents(A_matrices, G_matrices)

    def compute_eigenvectors_and_eigenvalues(self) -> None:
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

            _, eigvecs_A = jnp.linalg.eigh(A)
            _, eigvecs_G = jnp.linalg.eigh(G)

            activation_eigvecs[layer_name] = eigvecs_A
            gradient_eigvecs[layer_name] = eigvecs_G

        self.eigenvectors = LayerComponents(activation_eigvecs, gradient_eigvecs)
        self.storage.save_eigenvectors(self.eigenvectors)

    def compute_eigenvalue_corrections(self) -> None:
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

        self.eigenvalue_corrections = {}

        for layer_name, (a, g) in self.collector.captured_data.items():
            Q_A = self.eigenvectors.activations[layer_name]
            Q_G = self.eigenvectors.gradients[layer_name]

            # Project onto eigenbases
            g_tilde = jnp.einsum("oi, ni -> no", Q_G.T, g)  # [N, O]
            a_tilde = jnp.einsum("ij, nj -> ni", Q_A.T, a)  # [N, I]

            # Compute outer product and average
            outer = jnp.einsum("no, ni -> noi", g_tilde, a_tilde)  # [N, O, I]
            correction = (outer**2).mean(axis=0) + self.config.damping_lambda  # [O, I]

            self.eigenvalue_corrections[layer_name] = correction

        self.storage.save_eigenvalue_corrections(self.eigenvalue_corrections)

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

    def load_covariances_from_disk(self, model: ApproximationModel) -> LayerComponents:
        """Load previously computed covariances from disk."""
        self.collector.load_from_disk(model)
        self.covariances = self._compute_covariances_from_collected_data(
            use_bias=model.use_bias
        )
        return self.covariances


class KFACStorage:
    """Handles all disk I/O operations for EKFAC components."""

    def __init__(self, model_name: str, base_path: str = "data"):
        self.model_name = model_name
        self.base_path = Path(base_path) / model_name
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, filename: str) -> Path:
        return self.base_path / filename

    def save_eigenvectors(self, eigenvectors: LayerComponents) -> None:
        """Save eigenvectors to disk."""
        path = self._get_path("eigenvectors_eigenvalues.pkl")
        with open(path, "wb") as f:
            pickle.dump((eigenvectors.activations, eigenvectors.gradients), f)

    def load_eigenvectors(self) -> LayerComponents:
        """Load eigenvectors from disk."""
        path = self._get_path("eigenvectors_eigenvalues.pkl")
        if not path.exists():
            raise FileNotFoundError(f"No eigenvector file found at {path}")

        with open(path, "rb") as f:
            activations, gradients = pickle.load(f)
        return LayerComponents(activations, gradients)

    def save_eigenvalue_corrections(self, corrections: Dict[str, jnp.ndarray]) -> None:
        """Save eigenvalue corrections to disk."""
        path = self._get_path("eigenvalue_corrections.pkl")
        with open(path, "wb") as f:
            pickle.dump(corrections, f)

    def load_eigenvalue_corrections(self) -> Dict[str, jnp.ndarray]:
        """Load eigenvalue corrections from disk."""
        path = self._get_path("eigenvalue_corrections.pkl")
        if not path.exists():
            raise FileNotFoundError(f"No eigenvalue correction file found at {path}")

        with open(path, "rb") as f:
            corrections = pickle.load(f)
        return corrections
