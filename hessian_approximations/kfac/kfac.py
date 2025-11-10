from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from scipy.linalg import block_diag
from typing_extensions import override

from config.config import KFACConfig
from hessian_approximations.hessian_approximations import HessianApproximation
from hessian_approximations.kfac.activation_gradient_collector import (
    ActivationGradientCollector,
)
from hessian_approximations.kfac.layer_components import LayerComponents
from hessian_approximations.kfac.storage import KFACStorage
from models.loss import get_loss_name
from models.train import ApproximationModel


class KFAC(HessianApproximation):
    """
    Kronecker-Factored Approximate Curvature (KFAC) and Eigenvalue-Corrected KFAC (EKFAC) Hessian approximation.
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
        self.eigenvalue_corrections: Dict[str, Float[Array, "d_in d_out"]] = {}

        # Store the means for reuse in damping
        self.mean_eigenvalues: Dict[str, Float[Array, ""]] = {}
        self.mean_eigenvalue_corrections: Dict[str, Float[Array, ""]] = {}
        self.overall_mean_eigenvalue: Float[Array, ""] = jnp.array(0.0)
        self.overall_mean_eigenvalue_correction: Float[Array, ""] = jnp.array(0.0)

    def damping(self) -> Float[Array, ""]:
        """Get damping value from config.

        Returns:
        Float[Array, ""]: Damping value.
        """
        if self.config.run_config.damping_mode == "mean_eigenvalue":
            return self.config.run_config.damping_lambda * self.overall_mean_eigenvalue
        elif self.config.run_config.damping_mode == "mean_corrections":
            return (
                self.config.run_config.damping_lambda
                * self.overall_mean_eigenvalue_correction
            )
        else:
            raise ValueError(
                f"Unknown damping mode: {self.config.run_config.damping_mode}"
            )

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
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        training_targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
        loss_fn: Callable,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute full Hessian approximation.

        Not practical for large models but useful for testing and comparison
        with the true Hessian.
        """
        return self._compute_hessian_or_inverse_hessian(
            model,
            params,
            training_data,
            training_targets,
            loss_fn,
            method="normal",
        )

    def compute_inverse_hessian(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        training_targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
        loss_fn: Callable,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute inverse Hessian approximation.
        """
        return self._compute_hessian_or_inverse_hessian(
            model,
            params,
            training_data,
            training_targets,
            loss_fn,
            method="inverse",
        )

    def _compute_hessian_or_inverse_hessian(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        training_targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
        loss_fn: Callable,
        method: Literal["normal", "inverse"],
    ) -> Float[Array, "n_params n_params"]:
        """
        Unified helper method to compute either the full Hessian or its inverse.
        """

        self.get_ekfac_components(
            model, params, training_data, training_targets, loss_fn
        )
        hessian_layers = {}
        for layer_name in params["params"].keys():
            if self.config.run_config.use_eigenvalue_correction:
                hessian_layers[layer_name] = (
                    self._compute_layer_hessian_with_correction(layer_name, method)
                )
            else:
                hessian_layers[layer_name] = self._compute_layer_hessian_kfac_only(
                    layer_name, method
                )
        # Combine layer Hessians into full block-diagonal Hessian
        full_hessian = block_diag(*[jnp.asarray(H) for H in hessian_layers.values()])
        return jnp.array(full_hessian)

    @override
    def compute_hvp(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        training_targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
        loss_fn: Callable,
        vector: Float[Array, "*batch_size n_params"],
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Hessian-vector product."""
        return self._compute_ihvp_or_hvp(
            model,
            params,
            training_data,
            training_targets,
            loss_fn,
            vector,
            method="hvp",
        )

    @override
    def compute_ihvp(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        training_targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
        loss_fn: Callable,
        vector: Float[Array, "*batch_size n_params"],
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute inverse Hessian-vector product using EKFAC approximation.
        """
        return self._compute_ihvp_or_hvp(
            model,
            params,
            training_data,
            training_targets,
            loss_fn,
            vector,
            method="ihvp",
        )

    def _compute_ihvp_or_hvp(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        training_targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
        loss_fn: Callable,
        vector: Float[Array, "*batch_size n_params"],
        method: Literal["ihvp", "hvp"],
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute inverse Hessian-vector product or Hessian-vector product.

        Note, that the vector to be multiplied is reshaped in row-major order to
        match the JAX weight layout which is reflected by
        the eigenvalue corrections shape [input_dim, output_dim].
        """

        # Ensure EKFAC components are computed
        self.get_ekfac_components(
            model, params, training_data, training_targets, loss_fn
        )

        vp_pieces = []
        offset = 0

        for layer_name in params["params"].keys():
            Lambda: Float[Array, "I O"] = self.eigenvalue_corrections[layer_name]
            input_dim, output_dim = Lambda.shape
            size = input_dim * output_dim

            # Extract the corresponding part of the vector
            v_flat: Float[Array, "*batch_size I*O"] = vector[
                ..., offset : offset + size
            ]

            # Reshape last two dimensions to [I, O] matching JAX weights shape convention (also Lambda)
            v_layer: Float[Array, "*batch_size I O"] = v_flat.reshape(
                v_flat.shape[:-1] + (input_dim, output_dim)
            )

            # If running KFAC-only, we use the eigenvalues of the covariances and not the corrections
            if not self.config.run_config.use_eigenvalue_correction:
                Lambda = self._compute_eigenvalue_lambda_kfac(layer_name)

            vp_piece = self._compute_ihvp_or_hvp_layer(
                v_layer,
                self.eigenvectors.activations[layer_name],
                self.eigenvectors.gradients[layer_name],
                Lambda,
                self.damping(),
                method=method,
            )
            vp_pieces.append(vp_piece.reshape(v_flat.shape))
            offset += size

        # Concatenate all layer HVPS
        return jnp.concatenate(vp_pieces, axis=-1)

    def get_ekfac_components(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        training_targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
        loss_fn: Callable,
    ) -> None:
        """
        Compute all EKFAC components in sequence.

        Steps:
        1. Compute covariances from activations and gradients
        2. Compute eigenvectors of covariance matrices
        3. Compute eigenvalue corrections
        4. Compute mean eigenvalues and corrections for damping

        If components are already stored on disk and recalculation is not forced,
        they will be reused.
        """
        # check if kfac components are computed, if so reuse
        if self.storage.check_storage() and not (
            self.config.run_config.recalc_kfac_components
            or self.config.build_config.recalc_ekfac_components
        ):
            print("Loading EKFAC components from disk.")
            self._load_from_disk()
        else:
            print("Computing EKFAC components from scratch.")
            self._compute_from_scratch(
                model, params, training_data, training_targets, loss_fn
            )

    def _load_from_disk(self) -> None:
        """Load all (E)KFAC components from disk."""
        self.covariances = self.storage.load_covariances()
        self.eigenvectors = self.storage.load_eigenvectors()
        self.eigenvalues = self.storage.load_eigenvalues()
        self.eigenvalue_corrections = self.storage.load_eigenvalue_corrections()
        (
            self.mean_eigenvalues,
            self.mean_eigenvalue_corrections,
            self.overall_mean_eigenvalue,
            self.overall_mean_eigenvalue_correction,
        ) = self.storage.load_mean_eigenvalues_and_corrections()

    def _compute_from_scratch(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        training_targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
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
        self.compute_mean_eigenvalues_and_corrections()

    def compute_covariances(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
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
            A: Float[Array, "I I"] = self.covariances.activations[layer_name]
            G: Float[Array, "O O"] = self.covariances.gradients[layer_name]

            # Ensure numerical stability by using float64 for eigen decomposition
            A = A.astype(jnp.float64)
            G = G.astype(jnp.float64)

            eigenvals_A: Float[Array, "I"]
            eigvecs_A: Float[Array, "I I"]
            eigenvals_G: Float[Array, "O"]
            eigvecs_G: Float[Array, "O O"]

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
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        training_targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
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

    def compute_mean_eigenvalues_and_corrections(self) -> None:
        """Compute mean eigenvalues and eigenvalue corrections for damping."""
        self.mean_eigenvalues = {}
        self.mean_eigenvalue_corrections = {}

        overall_mean_eigenvalues = 0.0
        overall_mean_eigenvalue_corrections = 0.0

        for layer_name in self.eigenvalues.activations.keys():
            mean_eigenvalue: Float[Array, ""] = jnp.mean(
                self._compute_eigenvalue_lambda_kfac(layer_name)
            )
            self.mean_eigenvalues[layer_name] = mean_eigenvalue
            overall_mean_eigenvalues += mean_eigenvalue

        for layer_name in self.eigenvalue_corrections.keys():
            mean_correction: Float[Array, ""] = jnp.mean(
                self.eigenvalue_corrections[layer_name]
            )
            self.mean_eigenvalue_corrections[layer_name] = mean_correction
            overall_mean_eigenvalue_corrections += mean_correction

        # Divide overall sums by number of layers
        n_layers = len(self.eigenvalues.activations)
        self.overall_mean_eigenvalue = jnp.array(
            overall_mean_eigenvalues / n_layers if n_layers > 0 else 0.0
        )
        self.overall_mean_eigenvalue_correction = jnp.array(
            overall_mean_eigenvalue_corrections / n_layers if n_layers > 0 else 0.0
        )

        self.storage.save_mean_eigenvalues_and_corrections(
            self.mean_eigenvalues,
            self.mean_eigenvalue_corrections,
            self.overall_mean_eigenvalue,
            self.overall_mean_eigenvalue_correction,
        )

    def covariance(
        self, input: Float[Array, "n_samples features"]
    ) -> Float[Array, "features features"]:
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
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
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
        # Ensure different RNG keys for covariance and eigenvalue correction computations
        if self.config.build_config.use_pseudo_targets:
            if compute_method == "covariance":
                prng_key = jax.random.PRNGKey(42)
            else:
                prng_key = jax.random.PRNGKey(43)
            targets = self.generate_pseudo_targets(
                model, params, training_data, loss_fn, rng_key=prng_key
            )

        # Process all data in batches
        n_samples = training_data.shape[0]
        effective_batch_size = (
            self.config.build_config.collector_batch_size
            if self.config.build_config.collector_batch_size is not None
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
            # Average eigenvalue corrections over all samples
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
        self, store: Dict, key: str, batch_covariance: Float[Array, "..."]
    ) -> None:
        if key in store:
            store[key] += batch_covariance
        else:
            store[key] = batch_covariance

    def _accumulate_covariances(
        self,
        layer_name: str,
        activations: Float[Array, "n_batch d_in"],
        gradients: Float[Array, "n_batch d_out"],
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
        activations: Float[Array, "n_batch d_in"],
        gradients: Float[Array, "n_batch d_out"],
    ):
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
        Q_A: Float[Array, "d_in d_in"] = self.eigenvectors.activations[layer_name]
        Q_G: Float[Array, "d_out d_out"] = self.eigenvectors.gradients[layer_name]

        # Project activations and gradients onto eigenbases
        g_tilde: Float[Array, "n_batch d_out"] = jnp.einsum(
            "op, np -> no", Q_G.T, gradients
        )
        a_tilde: Float[Array, "n_batch d_in"] = jnp.einsum(
            "ij, nj -> ni", Q_A.T, activations
        )

        # Compute outer product and average
        outer: Float[Array, "n_batch d_in d_out"] = jnp.einsum(
            "ni, no -> nio", a_tilde, g_tilde
        )
        correction: Float[Array, "d_in d_out"] = (outer**2).sum(
            axis=0
        )  # (averaging is done by the calling method)

        self._accumulate_data(self.eigenvalue_corrections, layer_name, correction)

    def _compute_layer_hessians(self, params: Dict) -> Dict[str, jnp.ndarray]:
        """
        Compute Kronecker-factored Hessian approximations per layer.

        Depending on configuration, this computes either:
        - (E)KFAC Hessians with eigenvalue corrections, or
        - Standard KFAC Hessians using Kronecker-factored covariances.

        Note:
        PyTorch assumes weights shaped [d_out, d_in] with vec(∇W) = a ⊗ ∇s
        (column-major). In contrast, JAX uses [d_in, d_out], which yields
        vec(∇W') = ∇s ⊗ a due to the forward pass formulation y = xW'
        instead of y = Wx (as in PyTorch).

        Because JAX flattens arrays in row-major (C-style) order, the effective
        vectorization swaps again, giving vec_row(∇W') = a ⊗ ∇s. This matches
        the ordering used when comparing with the true Hessian or constructing
        Kronecker-factored curvature blocks.
        """
        hessian_layers = {}

        for layer_name in params["params"].keys():
            if self.config.run_config.use_eigenvalue_correction:
                hessian_layers[layer_name] = (
                    self._compute_layer_hessian_with_correction(layer_name)
                )
            else:
                hessian_layers[layer_name] = self._compute_layer_hessian_kfac_only(
                    layer_name
                )

        return hessian_layers

    @staticmethod
    @partial(jax.jit, static_argnames=["method"])
    def _compute_ihvp_or_hvp_layer(
        v_layer: Float[Array, "*batch_size I O"],
        Q_A: Float[Array, "I I"],
        Q_S: Float[Array, "O O"],
        Lambda: Float[Array, "I O"],
        damping: Float[Array, ""] = jnp.array(0.0),
        method: Literal["ihvp", "hvp"] = "ihvp",
    ) -> Float[Array, "*batch_size I O"]:
        """
        Compute the EKFAC-based (inverse) Hessian-vector product for a single layer.

        Depending on the selected method, this function computes either:
            - the inverse Hessian-vector product (IHVP):
                    (H + λI)⁻¹ v ≈ Q_A (Ṽ / (Λ + λ)) Q_S^T
            - or the Hessian-vector product (HVP):
                    (H + λI) v ≈ Q_A (Ṽ * (Λ + λ)) Q_S^T
        where Ṽ = Q_A^T unvectorized_v Q_S.

        Shapes follow the JAX convention with W ∈ R^{[I, O]},
        i.e., the vector is provided as unvectorized matrix in row-major order.
        """

        # Transform to eigenbasis
        V_tilde: Float[Array, "*batch_size I O"] = Q_A.T @ v_layer @ Q_S

        # Apply eigenvalue corrections + damping
        Lambda_damped: Float[Array, "I O"] = Lambda + damping

        if method == "ihvp":
            scaled: Float[Array, "*batch_size I O"] = V_tilde / Lambda_damped
        else:
            scaled: Float[Array, "*batch_size I O"] = V_tilde * Lambda_damped

        # Transform back to original basis
        vector_product: Float[Array, "*batch_size I O"] = Q_A @ scaled @ Q_S.T

        return vector_product

    def _compute_layer_hessian_with_correction(
        self, layer_name: str, method: Literal["inverse", "normal"] = "normal"
    ) -> Float[Array, "I*O I*O"]:
        """
        Compute layer Hessian with eigenvalue corrections (EKFAC).

        Note: The original KFAC formulation assumes weights shaped [d_out, d_in]
        with vec(∇W) = a ⊗ ∇s (column-major). In contrast, JAX uses [d_in, d_out], which yields
        vec(∇W') = ∇s ⊗ a due to the forward pass formulation y = xW'
        instead of y = Wx (as in PyTorch).

        Because JAX flattens arrays in row-major (C-style) order, the effective
        vectorization swaps again, giving vec_row(∇W') = a ⊗ ∇s. This matches
        the ordering used when comparing with the true Hessian or constructing
        Kronecker-factored curvature blocks.

        Since we store the eigenvalue corrections in the shape [input_dim, output_dim],
        we can directly use them here by flattening in JAX-default row-major order without needing to transpose.
        """
        A_eigvecs: Float[Array, "I I"] = self.eigenvectors.activations[layer_name]
        G_eigvecs: Float[Array, "O O"] = self.eigenvectors.gradients[layer_name]
        corrections: Float[Array, "I O"] = self.eigenvalue_corrections[layer_name]

        Q_kron: Float[Array, "I*O, I*O"] = jnp.kron(A_eigvecs, G_eigvecs)

        if method == "inverse":
            damped_corrections = 1.0 / (corrections + self.damping())
        else:
            damped_corrections = corrections + self.damping()

        H_layer: Float[Array, "I*O, I*O"] = (
            Q_kron @ jnp.diag(damped_corrections.flatten()) @ Q_kron.T
        )

        return H_layer

    def _compute_layer_hessian_kfac_only(
        self, layer_name: str, method: Literal["inverse", "normal"] = "normal"
    ) -> Float[Array, "I*O I*O"]:
        """
        Compute layer Hessian or its inverse without eigenvalue corrections (standard KFAC).

        Constructs the Hessian using the Kronecker-factored eigen-decomposition,
        but in contrast to EKFAC using the eigenvalues of the covariances and not the
        empirical eigenvalue corrections.

        We are using the eigen-decomposition to allow for damping,
        i.e., being able to compare KFAC and EKFAC fairly.

        Depending on the selected method:
        - "normal" computes the standard KFAC approximation with damping:
              H ≈ (Q_G ⊗ Q_A) diag(Λ_G ⊗ Λ_A + λ) (Q_G ⊗ Q_A)ᵀ
        - "inverse" computes its damped inverse:
              H⁻¹ ≈ (Q_G ⊗ Q_A) diag(1 / (Λ_G ⊗ Λ_A + λ)) (Q_G ⊗ Q_A)ᵀ

        Note: The original KFAC formulation assumes weights shaped [d_out, d_in]
        with vec(∇W) = a ⊗ ∇s (column-major). In contrast, JAX uses [d_in, d_out],
        which yields vec(∇W') = ∇s ⊗ a due to the forward pass formulation y = xW'
        instead of y = Wx (as in PyTorch).

        Because JAX flattens arrays in row-major (C-style) order, the effective
        vectorization swaps again, giving vec_row(∇W') = a ⊗ ∇s. This matches
        the ordering used when comparing with the true Hessian or constructing
        Kronecker-factored curvature blocks.

        Since we store the eigenvalues in the shape [input_dim, output_dim],
        we can directly use them here by flattening in JAX-default row-major order without needing to transpose.
        """
        eigenvectors_A: Float[Array, "I I"] = self.eigenvectors.activations[layer_name]
        eigenvectors_G: Float[Array, "O O"] = self.eigenvectors.gradients[layer_name]

        Q: Float[Array, "I*O, I*O"] = jnp.kron(eigenvectors_A, eigenvectors_G)
        Lambda: Float[Array, "I O"] = self._compute_eigenvalue_lambda_kfac(layer_name)

        if method == "inverse":
            Lambda = 1.0 / (Lambda + self.damping())
        else:
            Lambda = Lambda + self.damping()
        return Q @ jnp.diag(Lambda.flatten()) @ Q.T

    def _compute_eigenvalue_lambda_kfac(self, layer_name: str) -> Float[Array, "I O"]:
        """Compute eigenvalue lambda for KFAC using the following formula:
        Λ = (Λ_G ⊗ Λ_A) = Λ_A @ Λ_G^T
        where Λ_G and Λ_A are the eigenvalues of the gradient and activation covariances.
        """
        A_eigvals: Float[Array, "I"] = self.eigenvalues.activations[layer_name]
        G_eigvals: Float[Array, "O"] = self.eigenvalues.gradients[layer_name]
        Lambda_kfac: Float[Array, "I O"] = jnp.outer(A_eigvals, G_eigvals)

        return Lambda_kfac

    def generate_pseudo_targets(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        loss_fn: Callable,
        rng_key: PRNGKeyArray | None = None,
    ) -> Float[Array, "n_samples targets"] | Int[Array, "n_samples"]:
        """
        Generate pseudo-targets based on the model's output distribution.

        This is used to compute the true Fisher Information Matrix rather than
        the empirical Fisher (which would use true labels).
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
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        rng_key: PRNGKeyArray,
    ) -> Int[Array, "n_samples"]:
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
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        rng_key: PRNGKeyArray,
    ) -> Float[Array, "n_samples"]:
        """Generate pseudo-targets by adding Gaussian noise to predictions."""
        preds = model.apply(params, training_data)

        if not isinstance(preds, jnp.ndarray):
            raise ValueError("Model predictions must be a jnp.ndarray for regression.")
        noise = self.config.build_config.pseudo_target_noise_std * jax.random.normal(
            rng_key, preds.shape
        )
        return preds + noise
