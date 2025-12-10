from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Literal, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import override

from deleuze.hessians.computer.computer import HessianComputer
from deleuze.hessians.utils.data import KFACData
from deleuze.metrics.full_matrix_metrics import FullMatrixMetric


@dataclass
class KFACComputer(HessianComputer):
    """
    Kronecker-Factored Approximate Curvature (KFAC) and Eigenvalue-Corrected KFAC (EKFAC) Hessian approximation.
    """

    compute_context: KFACData

    @override
    def compute_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute full Hessian approximation.
        """
        return self.compute_hessian_or_inverse_hessian(
            method="normal",
            damping=0.0 if damping is None else damping,
        )

    @override
    def compute_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Hessian-vector product."""
        return self.compute_ihvp_or_hvp(
            vectors,
            method="hvp",
            damping=0.0 if damping is None else damping,
        )

    @override
    def compare_hessians(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> float:
        """
        Compare the (E)KFAC Hessian approximation to a given comparison matrix
        """
        return self._compare_hessians(
            activations_eigenvectors=list(
                self.compute_context.activation_eigenvectors.values()
            ),
            gradients_eigenvectors=list(
                self.compute_context.gradient_eigenvectors.values()
            ),
            Lambdas=list(
                self.compute_lambdas(
                    self.compute_context.activation_eigenvalues,
                    self.compute_context.gradient_eigenvalues,
                )
            ),
            damping=0.0 if damping is None else damping,
            comparison_matrix=comparison_matrix,
            metric=metric.compute_fn(),
            method="normal",
        )

    @override
    def compute_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute inverse Hessian-vector product.
        """
        return self.compute_ihvp_or_hvp(
            vectors=vectors,
            method="ihvp",
            damping=0.0 if damping is None else damping,
        )

    def compute_inverse_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute full inverse Hessian.
        """
        return self.compute_hessian_or_inverse_hessian(
            method="inverse",
            damping=0.0 if damping is None else damping,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["method", "metric"])
    def _compare_hessians(
        activations_eigenvectors: List[Float[Array, "I I"]],
        gradients_eigenvectors: List[Float[Array, "O O"]],
        Lambdas: List[Float[Array, "I O"]],
        damping: Float[Array, ""],
        comparison_matrix: Float[Array, "n_params n_params"],
        metric: Callable[[jnp.ndarray, jnp.ndarray], float],
        method: Literal["normal", "inverse"] = "normal",
    ) -> float:
        """Compare (E)KFAC Hessian or its inverse to a given comparison matrix and prespecified metric."""
        kfac_hessian = KFACComputer._compute_hessian_or_inverse_hessian(
            activations_eigenvectors,
            gradients_eigenvectors,
            Lambdas,
            damping=damping,
            method=method,
        )

        true_hessian = comparison_matrix + damping * jnp.eye(kfac_hessian.shape[0])
        return metric(true_hessian, kfac_hessian)

    def compute_hessian_or_inverse_hessian(
        self, method: Literal["normal", "inverse"], damping: Float
    ) -> Float[Array, "n_params n_params"]:
        """
        Unified helper method to compute either the full Hessian or its inverse.

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
        return self._compute_hessian_or_inverse_hessian(
            eigenvectors_activations=list(
                self.compute_context.activation_eigenvectors.values()
            ),
            eigenvectors_gradients=list(
                self.compute_context.gradient_eigenvectors.values()
            ),
            Lambdas=list(
                self.compute_lambdas(
                    self.compute_context.activation_eigenvalues,
                    self.compute_context.gradient_eigenvalues,
                ).values()
            ),
            damping=damping,
            method=method,
        )

    @staticmethod
    def _get_damped_lambda(
        Lambda: Float[Array, "I O"],
        damping: Float[Array, ""],
        method: Literal["normal", "inverse"],
    ) -> Float[Array, "n_params n_params"]:
        if method == "inverse":
            return 1.0 / (Lambda + damping)
        else:
            return Lambda + damping

    @staticmethod
    @jax.jit
    def _compute_layer_hessian(eigenvectors_A, eigenvectors_G, Lambda):
        """Compute single layer hessian by list of eigenvectors and eigenvalues / corrections."""
        return jnp.einsum(
            "ij,j,jk->ik",
            jnp.kron(eigenvectors_A, eigenvectors_G),
            Lambda.flatten(),
            jnp.kron(eigenvectors_A, eigenvectors_G).T,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["method"])
    def _compute_hessian_or_inverse_hessian(
        eigenvectors_activations: List[Float[Array, "I I"]],
        eigenvectors_gradients: List[Float[Array, "O O"]],
        Lambdas: List[Float[Array, "I O"]],
        damping: Float[Array, ""],
        method: Literal["normal", "inverse"],
    ):
        """
        Compute layer Hessian or its inverse for KFAC or EKFAC
        based on a layer-based lists of the required components.

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

        Since we store the eigenvalues and corrections in the shape [input_dim, output_dim],
        we can directly use them here by flattening in JAX-default row-major order without needing to transpose.
        """

        hessian_list = [
            KFACComputer._compute_layer_hessian(
                layer_eigv_activations,
                layer_eigv_gradients,
                KFACComputer._get_damped_lambda(layer_lambda, damping, method),
            )
            for layer_eigv_activations, layer_eigv_gradients, layer_lambda in zip(
                eigenvectors_activations,
                eigenvectors_gradients,
                Lambdas,
            )
        ]

        return jax.scipy.linalg.block_diag(*hessian_list)

    def compute_ihvp_or_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        method: Literal["ihvp", "hvp"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute inverse Hessian-vector product or Hessian-vector product.

        Note, that the vector to be multiplied is reshaped in row-major order to
        match the JAX weight layout which is reflected by
        the eigenvalue corrections shape [input_dim, output_dim].
        """

        # Collect all layer data upfront
        layer_names = self.compute_context.layer_names()
        Lambdas = self.compute_lambdas(
            self.compute_context.activation_eigenvalues,
            self.compute_context.gradient_eigenvalues,
        )

        Q_activations_list = []
        Q_gradients_list = []
        Lambda_list = []
        v_layers = []

        offset = 0
        for layer_name in layer_names:
            Lambda: Float[Array, "I O"] = Lambdas[layer_name]
            input_dim, output_dim = Lambda.shape
            size = input_dim * output_dim

            # Extract and reshape vector for this layer
            v_flat: Float[Array, "*batch_size I*O"] = vectors[
                ..., offset : offset + size
            ]
            v_layer: Float[Array, "*batch_size I O"] = v_flat.reshape(
                v_flat.shape[:-1] + (input_dim, output_dim)
            )

            # Collect all components
            Q_activations_list.append(
                self.compute_context.activation_eigenvectors[layer_name]
            )
            Q_gradients_list.append(
                self.compute_context.gradient_eigenvectors[layer_name]
            )
            Lambda_list.append(Lambda)
            v_layers.append(v_layer)
            offset += size

        # Process all layers at once
        vp_pieces = self.compute_ihvp_or_hvp_all_layers(
            v_layers=v_layers,
            Q_activations=Q_activations_list,
            Q_gradients=Q_gradients_list,
            Lambdas=Lambda_list,
            damping=damping,
            method=method,
        )

        # Concatenate all layer results
        return jnp.concatenate(vp_pieces, axis=-1)

    @staticmethod
    @partial(jax.jit, static_argnames=["method"])
    def compute_ihvp_or_hvp_all_layers(
        v_layers: list[Float[Array, "*batch_size I O"]],
        Q_activations: list[Float[Array, "I I"]],
        Q_gradients: list[Float[Array, "O O"]],
        Lambdas: list[Float[Array, "I O"]],
        damping: Float[Array, ""],
        method: Literal["ihvp", "hvp"],
    ) -> list[Float[Array, "*batch_size num_params"]]:
        """
        Compute EKFAC-based (inverse) Hessian-vector product for all layers in one call.
        Returns a list of flattened vector products, one per layer.
        """
        vp_pieces = []

        for v_layer, Q_A, Q_G, Lambda in zip(
            v_layers, Q_activations, Q_gradients, Lambdas
        ):
            # Transform to eigenbasis
            V_tilde: Float[Array, "*batch_size I O"] = Q_A.T @ v_layer @ Q_G

            # Apply eigenvalue corrections + damping
            Lambda_damped: Float[Array, "I O"] = Lambda + damping

            if method == "ihvp":
                scaled: Float[Array, "*batch_size I O"] = V_tilde / Lambda_damped
            else:
                scaled: Float[Array, "*batch_size I O"] = V_tilde * Lambda_damped

            # Transform back to original basis
            vector_product: Float[Array, "*batch_size I O"] = Q_A @ scaled @ Q_G.T

            # Flatten last two dimensions: [*batch_size, I, O] -> [*batch_size, I*O]
            # This works for both single vector (I, O) and batched (*batch, I, O)
            batch_shape = vector_product.shape[:-2]
            flat_size = vector_product.shape[-2] * vector_product.shape[-1]
            vp_flat = vector_product.reshape(*batch_shape, flat_size)
            vp_pieces.append(vp_flat)

        return vp_pieces

    def compute_lambdas(
        self,
        activation_eigenvalues: Dict[str, Float[Array, "I"]],
        gradient_eigenvalues: Dict[str, Float[Array, "O"]],
    ) -> Dict[str, Float[Array, "I O"]]:
        """Compute eigenvalue lambda for KFAC using the following formula:
        Λ = (Λ_G ⊗ Λ_A) = Λ_A @ Λ_G^T
        where Λ_G and Λ_A are the eigenvalues of the gradient and activation covariances.
        """
        lambdas = {}
        for layer_name in activation_eigenvalues.keys():
            A_eigvals: Float[Array, "I"] = activation_eigenvalues[layer_name]
            G_eigvals: Float[Array, "O"] = gradient_eigenvalues[layer_name]
            lambdas[layer_name] = jnp.outer(A_eigvals, G_eigvals)
        return lambdas
