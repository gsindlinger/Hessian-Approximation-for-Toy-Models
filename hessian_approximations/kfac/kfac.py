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
        dataset_name: str = "dataset",
        config: KFACConfig | None = None,
    ):
        super().__init__()
        self.config = config or KFACConfig()
        self.storage = KFACStorage(model_name=model_name, dataset_name=dataset_name)
        self.collector = ActivationGradientCollector()

        # Core components
        self.covariances = LayerComponents()
        self.eigenvalues = LayerComponents()
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
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute inverse Hessian-vector product using EKFAC approximation.
        """

        # Ensure EKFAC components are computed
        self.generate_ekfac_components(
            model, params, training_data, training_targets, loss_fn
        )

        ihvp_pieces = []
        offset = 0

        for layer_name in params["params"].keys():
            Lambda = self.eigenvalue_corrections[layer_name]  # shape [I, O]

            input_dim, output_dim = Lambda.shape  # Lambda shape is [I, O]
            size = input_dim * output_dim

            # Extract the corresponding part of the vector
            v_flat = vector[..., offset : offset + size]

            # Reshape last two dimensions to [I, O] matching JAX weights shape convention (also Lambda)
            v_layer = v_flat.reshape(v_flat.shape[:-1] + (input_dim, output_dim))

            if self.config.use_eigenvalue_correction:
                ihvp_piece = self._compute_ihvp_layer(
                    v_layer,
                    self.eigenvectors.activations[layer_name],
                    self.eigenvectors.gradients[layer_name],
                    Lambda,
                    self.config.damping_lambda,
                )
            else:
                Lambda_kfac = self.compute_eigenvalue_correction_kfac(layer_name)
                ihvp_piece = self._compute_ihvp_layer(
                    v_layer,
                    self.eigenvectors.activations[layer_name],
                    self.eigenvectors.gradients[layer_name],
                    Lambda_kfac,
                    self.config.damping_lambda,
                )

            ihvp_pieces.append(ihvp_piece)
            offset += size

        # Concatenate all layer IHVPs
        return jnp.concatenate(ihvp_pieces, axis=-1)

    def compute_eigenvalue_correction_kfac(self, layer_name: str) -> jnp.ndarray:
        """Compute eigenvalue correction for KFAC (without EKFAC correction)."""
        A_eigvals = self.eigenvalues.activations[layer_name]
        G_eigvals = self.eigenvalues.gradients[layer_name]
        Lambda_kfac = jnp.outer(A_eigvals, G_eigvals)  # shape [I, O]
        return Lambda_kfac

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
        # check if kfac components are computed, if so reuse
        if self.storage.check_storage() and not self.config.reload_data:
            print("Loading EKFAC components from disk.")
            self._load_from_disk(model)
        else:
            print("Computing EKFAC components from scratch.")
            self._compute_from_scratch(
                model, params, training_data, training_targets, loss_fn
            )

    def _load_from_disk(self, model: ApproximationModel) -> None:
        """Load all (E)KFAC components from disk."""
        self.load_covariances_from_disk()
        self.load_eigenvectors_from_disk()
        self.load_eigenvalues_from_disk()
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

        activation_eigvals = {}
        activation_eigvecs = {}
        gradient_eigvals = {}
        gradient_eigvecs = {}

        for layer_name in self.covariances.activations.keys():
            A = self.covariances.activations[layer_name]
            G = self.covariances.gradients[layer_name]

            # Ensure numerical stability by using float64 for eigen decomposition
            A = A.astype(jnp.float64)
            G = G.astype(jnp.float64)

            eigenvals_A, eigvecs_A = jnp.linalg.eigh(A)
            eigenvals_G, eigvecs_G = jnp.linalg.eigh(G)

            activation_eigvecs[layer_name] = eigvecs_A.astype(jnp.float32)
            gradient_eigvecs[layer_name] = eigvecs_G.astype(jnp.float32)

            activation_eigvals[layer_name] = eigenvals_A.astype(jnp.float32)
            gradient_eigvals[layer_name] = eigenvals_G.astype(jnp.float32)

        self.eigenvectors = LayerComponents(activation_eigvecs, gradient_eigvecs)
        self.eigenvalues = LayerComponents(activation_eigvals, gradient_eigvals)
        self.storage.save_eigenvectors(self.eigenvectors)
        self.storage.save_eigenvalues(self.eigenvalues)

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
            self.config.collector_batch_size
            if self.config.collector_batch_size is not None
            else n_samples
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
                )

    def _process_single_batch_collector(
        self,
        batch_size: int,
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
            (Q_G \otimes Q_A)^T vec(a_n g_n^T)

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
        The EKFAC paper of Grosse et al. (2023) misses the transpose of the eigenvector
        basis (Q_A \otimes Q_G) in Equation (20). Refer to George et al. (2018) for the
        correct formulation.

        In this JAX implementation, we additionally swap the Kronecker order to
        (Q_G \otimes Q_A) because JAX layers store weights with shape [input_dim, output_dim],
        unlike the [output_dim, input_dim] convention used in the original paper.
        This ensures consistency between parameter vectorization, Kronecker factors,
        and the true Hessian structure under JAX’s row-major layout.
        """
        Q_A = self.eigenvectors.activations[layer_name]
        Q_G = self.eigenvectors.gradients[layer_name]
        # Project activations and gradients onto eigenbases
        g_tilde = jnp.einsum("op, np -> no", Q_G.T, gradients)  # [N, O]
        a_tilde = jnp.einsum("ij, nj -> ni", Q_A.T, activations)  # [N, I]

        # Compute outer product and average
        outer = jnp.einsum("ni, no -> nio", a_tilde, g_tilde)  # [N, I, O]
        correction = (outer**2).sum(
            axis=0
        )  # [I, O] (averaging is done by the calling method)

        self._accumulate_data(self.eigenvalue_corrections, layer_name, correction)

    def _compute_layer_hessians(self, params: Any) -> Dict[str, jnp.ndarray]:
        """
        Compute Hessian approximation for each layer.

        Depending on configuration, this computes either:
        - (E)KFAC Hessians with eigenvalue corrections, or
        - Standard KFAC Hessians using Kronecker-factored covariances.

        Note:
        In this JAX implementation, layer weights follow the [input_dim, output_dim]
        convention (unlike the [output_dim, input_dim] layout assumed in the KFAC and
        EKFAC papers). This requires using the Kronecker order (Q_G ⊗ Q_A) to ensure
        that vec(W) and the resulting Hessian block align correctly under JAX's
        row-major parameter flattening.
        """
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

    @staticmethod
    @jax.jit
    def _compute_ihvp_layer(
        v_layer: jnp.ndarray,
        Q_A: jnp.ndarray,  # shape [I, I]
        Q_S: jnp.ndarray,  # shape [O, O]
        Lambda: jnp.ndarray,  # shape [I, O]
        damping: float = 0.1,
    ) -> jnp.ndarray:
        """
        Compute EKFAC inverse Hessian–vector product for a single layer.

        The computation is performed in the Kronecker-factored eigenbasis:
            (H + λI)^{-1} v ≈ Q_A (Ṽ / (Λ + λ)) Q_S^T
        where Ṽ = Q_A^T v Q_S.

        Shapes follow the JAX convention W ∈ R^{[I, O]}, so the correct Kronecker
        factorization order is (Q_G ⊗ Q_A). This ensures that vec(W) and the
        corresponding Hessian block align under row-major flattening.

        Damping λ is scaled by the mean of Λ for numerical stability.
        """

        # Transform to eigenbasis
        V_tilde = Q_A.T @ v_layer @ Q_S  # works with leading dims via broadcasting

        # Scale by eigenvalue corrections + damping
        denom = Lambda + damping * jnp.mean(Lambda)  # shape [I, O]
        scaled = V_tilde / denom

        # Transform back to original basis
        ihvp_mat = Q_A @ scaled @ Q_S.T

        return ihvp_mat.reshape(v_layer.shape[:-2] + (-1,))

    def _compute_layer_hessian_with_correction(self, layer_name: str) -> jnp.ndarray:
        """
        Compute layer Hessian with eigenvalue corrections (EKFAC).

        Uses the Kronecker structure:
            H ≈ (Q_G ⊗ Q_A) diag(Λ + λ) (Q_G ⊗ Q_A)^T

        where Λ contains per-(i, o) eigenvalue corrections. For JAX's
        [input_dim, output_dim] weight layout, the Kronecker order (G ⊗ A)
        matches the vectorization order of vec(W) under row-major flattening.

        Damping λ is scaled by the mean correction for stability.
        """
        A_eigvecs = self.eigenvectors.activations[layer_name]  # shape [I, I]
        G_eigvecs = self.eigenvectors.gradients[layer_name]  # shape [O, O]
        corrections = self.eigenvalue_corrections[layer_name]  # shape [I, O]

        Q_kron = jnp.kron(G_eigvecs, A_eigvecs)  # [O*I, O*I]
        H_layer = (
            Q_kron
            @ jnp.diag(
                corrections.flatten()
                + self.config.damping_lambda * jnp.mean(corrections)
            )
            @ Q_kron.T
        )

        return H_layer  # shape [O*I, O*I]

    def _compute_layer_hessian_kfac_only(self, layer_name: str) -> jnp.ndarray:
        """
        Compute layer Hessian without eigenvalue corrections (standard KFAC).

        Constructs the Hessian using the Kronecker-factored eigen-decomposition,
        but in contrast to EKFAC using the eigenvalues of the covariances and not the
        empirical eigenvalue corrections.

        We are using the eigen-decomposition to allow for damping,
        i.e., being able to compare KFAC and EKFAC fairly.

            H ≈ (Q_G ⊗ Q_A) diag(Λ_G ⊗ Λ_A + λ) (Q_G ⊗ Q_A)^T

        Here (Q_G ⊗ Q_A) reflects the correct Kronecker order for JAX's
        [input_dim, output_dim] parameter layout, ensuring alignment with
        vec(W) under row-major flattening. Damping λ is applied to improve
        numerical stability.
        """
        eigenvectors_A = self.eigenvectors.activations[layer_name]
        eigenvectors_G = self.eigenvectors.gradients[layer_name]
        eigenvalues_A = self.eigenvalues.activations[layer_name]
        eigenvalues_G = self.eigenvalues.gradients[layer_name]

        Q = jnp.kron(eigenvectors_G, eigenvectors_A)
        Lambda = jnp.outer(eigenvalues_G, eigenvalues_A)
        Lambda = Lambda + self.config.damping_lambda * jnp.mean(Lambda)
        return Q @ jnp.diag(Lambda.flatten()) @ Q.T

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

    def load_eigenvalues_from_disk(self) -> None:
        """Load previously computed eigenvalues from disk."""
        self.eigenvalues = self.storage.load_eigenvalues()


class KFACStorage:
    """Handles all disk I/O operations for (E)KFAC components using NumPy's npz format."""

    def __init__(
        self,
        model_name: str = "model",
        dataset_name: str = "dataset",
        base_path: str | Path = "data",
    ):
        if dataset_name is None:
            self.base_path = Path(base_path) / model_name
        else:
            self.base_path = Path(base_path) / model_name / dataset_name
        self.base_path.mkdir(parents=True, exist_ok=True)

    def check_storage(self) -> bool:
        """Check if EKFAC components are already stored on disk."""
        cov_path = self._get_path("covariances.npz")
        eigvec_path = self._get_path("eigenvectors.npz")
        eigval_corr_path = self._get_path("eigenvalue_corrections.npz")
        return cov_path.exists() and eigvec_path.exists() and eigval_corr_path.exists()

    def delete_storage(self) -> None:
        """Delete all stored EKFAC component files from disk including parent folder."""
        for filename in [
            "covariances.npz",
            "eigenvectors.npz",
            "eigenvalues.npz",
            "eigenvalue_corrections.npz",
        ]:
            path = self._get_path(filename)
            if path.exists():
                path.unlink()
        try:
            self.base_path.rmdir()
        except OSError:
            pass  # Directory not empty

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

    def save_eigenvalues(self, eigenvalues: LayerComponents) -> None:
        self._save_layer_components("eigenvalues.npz", eigenvalues)

    def load_eigenvalues(self) -> LayerComponents:
        return self._load_layer_components("eigenvalues.npz")

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
