from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.utils.metrics.full_matrix_metrics import FullMatrixMetric


@dataclass
class HessianEstimator(ABC):
    compute_context: Any

    @abstractmethod
    def estimate_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "n_params n_params"]:
        """Compute Hessian approximation."""
        pass

    @abstractmethod
    def compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """Compare Hessian approximation with another Hessian matrix."""
        pass

    @abstractmethod
    def estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Hessian-vector product."""
        pass

    @abstractmethod
    def estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Inverse Hessian-vector product."""
        pass

    @staticmethod
    @partial(jax.jit, static_argnames=["method"])
    def _compute_hessian_or_inverse_hessian_estimate(
        eigenvectors_activations: List[Float[Array, "I I"]],
        eigenvectors_gradients: List[Float[Array, "O O"]],
        Lambdas: List[Float[Array, "I O"]],
        damping: Float[Array, ""],
        method: Literal["normal", "inverse"],
    ):
        """
        Computes a block-diagonal estimate for the Hessian or its inverse
        using Kronecker-factored eigenvectors and eigenvalues / corrections.

        Depending on the selected method:
        - "normal" computes:
              H ≈ (Q_G ⊗ Q_A) diag(Λ_G ⊗ Λ_A + λ) (Q_G ⊗ Q_A)ᵀ
        - "inverse" computes:
              H⁻¹ ≈ (Q_G ⊗ Q_A) diag(1 / (Λ_G ⊗ Λ_A + λ)) (Q_G ⊗ Q_A)ᵀ
        """

        hessian_list = [
            HessianEstimator._compute_layer_hessian_estimate(
                layer_eigv_activations,
                layer_eigv_gradients,
                HessianEstimator._get_damped_lambda(layer_lambda, damping, method),
            )
            for layer_eigv_activations, layer_eigv_gradients, layer_lambda in zip(
                eigenvectors_activations,
                eigenvectors_gradients,
                Lambdas,
            )
        ]

        return jax.scipy.linalg.block_diag(*hessian_list)

    @staticmethod
    def _get_damped_lambda(
        Lambda: Float[Array, "I O"],
        damping: Float[Array, ""],
        method: Literal["normal", "inverse"],
    ) -> Float[Array, "n_params n_params"]:
        """Compute the damped version of Lambda for the Hessian or its inverse."""
        if method == "inverse":
            return 1.0 / (Lambda + damping)
        else:
            return Lambda + damping

    @staticmethod
    @jax.jit
    def _compute_layer_hessian_estimate(
        eigenvectors_A: Float[Array, "I I"],
        eigenvectors_G: Float[Array, "O O"],
        Lambda: Float[Array, "I O"],
    ) -> Float[Array, "n_params n_params"]:
        """
        Computes the layer Hessian approximation given eigenvectors of activations and gradients,
        and the eigenvalue / correction matrix.

        Note: The formulation in KFAC, etc. assume weights shaped [d_out, d_in]
        with vec(∇W) = a ⊗ ∇s (column-major, similar to Pytorch).
        In contrast, JAX uses [d_in, d_out], which yields vec(∇W') = ∇s ⊗ a
        due to the forward pass formulation y = xW' instead of y = Wx (as in PyTorch).

        Because JAX flattens arrays in row-major (C-style) order, the effective
        vectorization swaps again, giving vec_row(∇W') = a ⊗ ∇s. This matches
        the ordering used when comparing with the true Hessian or constructing
        Kronecker-factored curvature blocks.

        Since we store the eigenvalues and corrections in the shape [input_dim, output_dim],
        we can directly use them here by flattening in JAX-default row-major order without needing to transpose.
        """
        return jnp.einsum(
            "ij,j,jk->ik",
            jnp.kron(eigenvectors_A, eigenvectors_G),
            Lambda.flatten(),
            jnp.kron(eigenvectors_A, eigenvectors_G).T,
        )

    def compute_ihvp_or_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        Lambdas: Dict[str, Float[Array, "I O"]],
        layer_names: List[str],
        method: Literal["ihvp", "hvp"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute inverse Hessian-vector product or Hessian-vector product.
        Unified method for different approaches, i.e., EK-FAC and K-FAC.
        Uses the respective eigenvectors which are stored in the compute context
        and the provided eigenvalues / corrections (Lambdas)

        Note, that the vector to be multiplied is reshaped in row-major order to
        match the JAX weight layout which is reflected by
        the eigenvalue corrections shape [input_dim, output_dim].
        """

        Q_activations_list = []
        Q_gradients_list = []
        Lambda_list = []
        v_layers = []

        offset = 0
        for layer_name in layer_names:
            Lambda = Lambdas[layer_name]
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

        # Compute (I)HVP for all layers
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
        Computes the inverse Hessian-vector product (IHVP) or Hessian-vector product (HVP) for multiple layers.
        Uses the Kronecker-factored eigenvectors and a corresponding eigenvalue / corrections matrix per layer.
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

    @staticmethod
    @partial(jax.jit, static_argnames=["method", "metric"])
    def _compare_hessian_estimates(
        activations_eigenvectors: List[Float[Array, "I I"]],
        gradients_eigenvectors: List[Float[Array, "O O"]],
        Lambdas: List[Float[Array, "I O"]],
        damping: Float[Array, ""],
        comparison_matrix: Float[Array, "n_params n_params"],
        metric: Callable[[jnp.ndarray, jnp.ndarray], float],
        method: Literal["normal", "inverse"] = "normal",
    ) -> float:
        """Compare (E)KFAC Hessian or its inverse to a given comparison matrix and prespecified metric."""
        kfac_hessian = HessianEstimator._compute_hessian_or_inverse_hessian_estimate(
            activations_eigenvectors,
            gradients_eigenvectors,
            Lambdas,
            damping=damping,
            method=method,
        )

        true_hessian = comparison_matrix + damping * jnp.eye(kfac_hessian.shape[0])
        return metric(true_hessian, kfac_hessian)
