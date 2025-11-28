import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Literal, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ...config.hessian_approximation_config import KFACRunConfig
from .kfac_data_provider import KFACProvider

logger = logging.getLogger(__name__)


@dataclass
class KFACComputer:
    """Class to compute K-FAC approximations of the Hessian."""

    config: KFACRunConfig

    def compute_hessian_or_inverse_hessian(
        self,
        provider: KFACProvider,
        damping: Float,
        method: Literal["normal", "inverse"],
    ):
        eigenvectors_activations, eigenvectors_gradients, Lambdas, _ = (
            provider.collect_data(
                use_eigenvalue_correction=self.config.use_eigenvalue_correction
            )
        )
        return self._compute_hessian_or_inverse_hessian(
            eigenvectors_activations=eigenvectors_activations,
            eigenvectors_gradients=eigenvectors_gradients,
            Lambdas=Lambdas,
            damping=damping,
            method=method,
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

    @staticmethod
    @jax.jit
    def _compute_layer_hessian(
        eigenvectors_A: Float[Array, "I I"],
        eigenvectors_G: Float[Array, "O I"],
        Lambda: Float[Array, "I O"],
    ):
        """Compute single layer hessian by list of eigenvectors and eigenvalues / corrections."""
        return jnp.einsum(
            "ij,j,jk->ik",
            jnp.kron(eigenvectors_A, eigenvectors_G),
            Lambda.flatten(),
            jnp.kron(eigenvectors_A, eigenvectors_G).T,
        )

    def compute_ihvp_or_hvp(
        self,
        provider: KFACProvider,
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
        eigenvectors_A, eigenvectors_G, Lambdas, vectors_reshaped = (
            provider.collect_data(
                use_eigenvalue_correction=self.config.use_eigenvalue_correction,
                vectors=vectors,
            )
        )
        vp = self.compute_ihvp_or_hvp_all_layers(
            v_layers=vectors_reshaped,
            eigenvectors_A=eigenvectors_A,
            eigenvectors_G=eigenvectors_G,
            Lambdas=Lambdas,
            damping=damping,
            method=method,
        )
        return vp

    @staticmethod
    @partial(jax.jit, static_argnames=["method"])
    def compute_ihvp_or_hvp_layer(
        vectors_reshaped: Float[Array, "*batch_size I O"],
        Q_A: Float[Array, "I I"],
        Q_S: Float[Array, "O O"],
        Lambda: Float[Array, "I O"],
        damping: Float[Array, ""] = jnp.array(0.0),
        method: Literal["ihvp", "hvp"] = "ihvp",
    ) -> Float[Array, "*batch_size I O"]:
        """
        Compute the EKFAC-based (inverse) Hessian-vector product for a single layer.
        """

        # Transform to eigenbasis
        V_tilde: Float[Array, "*batch_size I O"] = Q_A.T @ vectors_reshaped @ Q_S

        # Apply eigenvalue corrections + damping
        Lambda_damped: Float[Array, "I O"] = Lambda + damping

        if method == "ihvp":
            scaled: Float[Array, "*batch_size I O"] = V_tilde / Lambda_damped
        else:
            scaled: Float[Array, "*batch_size I O"] = V_tilde * Lambda_damped

        # Transform back to original basis
        vector_product: Float[Array, "*batch_size I O"] = Q_A @ scaled @ Q_S.T

        return vector_product

    @staticmethod
    @partial(jax.jit, static_argnames=["method"])
    def compute_ihvp_or_hvp_all_layers(
        v_layers: list[Float[Array, "*batch_size I O"]],
        eigenvectors_A: list[Float[Array, "I I"]],
        eigenvectors_G: list[Float[Array, "O O"]],
        Lambdas: list[Float[Array, "I O"]],
        damping: Float[Array, ""],
        method: Literal["ihvp", "hvp"],
    ) -> Float[Array, "*batch_size num_params"]:
        """
        Compute EKFAC-based (inverse) Hessian-vector product for all layers in one call.
        Returns a list of flattened vector products, one per layer.
        """
        vp_pieces = []

        for v_layer, Q_A, Q_G, Lambda in zip(
            v_layers, eigenvectors_A, eigenvectors_G, Lambdas
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

        # concatenate all layer pieces along the last dimension
        return jnp.concatenate(vp_pieces, axis=-1)

    def compare_hessians(
        self,
        provider: KFACProvider,
        comparison_matrix: Float[Array, "n_params n_params"],
        metric: Callable[[jnp.ndarray, jnp.ndarray], float],
        damping: Float,
        method: Literal["normal", "inverse"] = "normal",
    ) -> float:
        """Compare (E)KFAC Hessian or its inverse to a given comparison matrix and prespecified metric."""
        eigenvectors_A, eigenvectors_G, Lambdas, _ = provider.collect_data(
            use_eigenvalue_correction=self.config.use_eigenvalue_correction
        )
        return self._compare_hessians(
            eigenvectors_activations=eigenvectors_A,
            eigenvectors_gradients=eigenvectors_G,
            Lambdas=Lambdas,
            damping=0.0 if damping is None else damping,
            comparison_matrix=comparison_matrix,
            metric=metric,
            method=method,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["method", "metric"])
    def _compare_hessians(
        eigenvectors_activations: List[Float[Array, "I I"]],
        eigenvectors_gradients: List[Float[Array, "O O"]],
        Lambdas: List[Float[Array, "I O"]],
        damping: Float[Array, ""],
        comparison_matrix: Float[Array, "n_params n_params"],
        metric: Callable[[jnp.ndarray, jnp.ndarray], float],
        method: Literal["normal", "inverse"] = "normal",
    ) -> float:
        """Compare (E)KFAC Hessian or its inverse to a given comparison matrix and prespecified metric."""
        kfac_hessian = KFACComputer._compute_hessian_or_inverse_hessian(
            eigenvectors_activations=eigenvectors_activations,
            eigenvectors_gradients=eigenvectors_gradients,
            Lambdas=Lambdas,
            damping=damping,
            method=method,
        )

        true_hessian = comparison_matrix + damping * jnp.eye(kfac_hessian.shape[0])
        return metric(kfac_hessian, true_hessian)

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
