from typing import List, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import CollectorBasedHessianEstimator
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric


class FIMBlockComputer(CollectorBasedHessianEstimator):
    def _estimate_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute the Fisher Information Matrix block approximation.
        """
        damping = 0.0 if damping is None else damping
        return self._compute_fim_block(
            activations=[
                self.compute_context[0].activations[layer_name]
                for layer_name in self.compute_context[0].layer_names
            ],
            gradients=[
                self.compute_context[0].gradients[layer_name]
                for layer_name in self.compute_context[0].layer_names
            ],
            damping=damping,
        )

    def _estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the Fisher Information Matrix block-vector product.
        """
        damping = 0.0 if damping is None else damping
        return self._compute_fim_block_hvp(
            activations=[
                self.compute_context[0].activations[layer_name]
                for layer_name in self.compute_context[0].layer_names
            ],
            gradients=[
                self.compute_context[0].gradients[layer_name]
                for layer_name in self.compute_context[0].layer_names
            ],
            vectors=vectors,
            damping=damping,
        )

    def _compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """
        Compare the FIM block approximation with another Hessian matrix.
        """
        damping = 0.0 if damping is None else damping

        return metric.compute_fn()(
            comparison_matrix,
            self._compute_fim_block(
                activations=[
                    self.compute_context[0].activations[layer_name]
                    for layer_name in self.compute_context[0].layer_names
                ],
                gradients=[
                    self.compute_context[0].gradients[layer_name]
                    for layer_name in self.compute_context[0].layer_names
                ],
                damping=damping,
            ),
        )

    def _estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the inverse Fisher Information Matrix block-vector product.
        """
        damping = 0.0 if damping is None else damping
        return self._compute_fim_block_ihvp(
            activations=[
                self.compute_context[0].activations[layer]
                for layer in self.compute_context[0].layer_names
            ],
            gradients=[
                self.compute_context[0].gradients[layer]
                for layer in self.compute_context[0].layer_names
            ],
            vectors=vectors,
            damping=damping,
        )

    @staticmethod
    @jax.jit
    def _compute_fim_block(
        activations: List[Float[Array, "N I"]],
        gradients: List[Float[Array, "N O"]],
        damping: float,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute block-diagonal FIM: F = block_diag(F_1, ..., F_L) + λI.

        For each layer block: F_l = (1/N) Σ_n w^(n) (w^(n))^T
        where w^(n) = g_l^(n) ⊗ a_{l-1}^(n)

        Note, that this implementation is independant of the selection for the "true" or the "empirical"
        Fisher Information Matrix, as it only relies on the per-sample activations and gradients.

        For the true FIM, it is required that the gradients are computed w.r.t. to pseudo-targets sampled
        from the model's predictive distribution.
        """

        fim_blocks = []
        for act, grad in zip(activations, gradients):
            # compute per-sample outer products and vectorize
            # (compare eq. 14 / 15 in Grosse et al. (2023))
            per_sample_vecs = jnp.einsum("ni,nj->nij", act, grad).reshape(
                act.shape[0], -1
            )

            # compute outer product to get FIM block (eq. 16 in Grosse et al. (2023))
            fim_block = (per_sample_vecs.T @ per_sample_vecs) / act.shape[0]

            # add damping
            fim_block = fim_block + damping * jnp.eye(fim_block.shape[0])

            fim_blocks.append(fim_block)

        return jax.scipy.linalg.block_diag(*fim_blocks)

    @staticmethod
    @jax.jit
    def _compute_fim_block_hvp(
        activations: List[Float[Array, "N I"]],  # list of (N, I)
        gradients: List[Float[Array, "N O"]],  # list of (
        vectors: Float[Array, "*batch_size n_params"],  # (..., n_params)
        damping: float,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute (F + λI)v for block-diagonal FIM.

        For each layer block: F_l = (1/N) Σ_n w^(n) (w^(n))^T
        where w^(n) = g_l^(n) ⊗ a_{l-1}^(n)

        Uses efficient formulation: F_l v = (1/N) Σ_n <v, w^(n)> w^(n)
        to avoid materializing the full matrix (O(ND) vs O(D²)).
        """
        results = []
        offset = 0

        for act, grad in zip(activations, gradients):
            D = act.shape[1] * grad.shape[1]
            v_block = vectors[..., offset : offset + D]

            per_sample_vecs = jnp.einsum("ni,nj->nij", act, grad).reshape(
                act.shape[0], -1
            )

            coeffs = jnp.einsum("...d,nd->...n", v_block, per_sample_vecs)
            y_block = (
                jnp.einsum("...n,nd->...d", coeffs, per_sample_vecs) / act.shape[0]
            )
            # add damping to block
            y_block = y_block + damping * v_block

            results.append(y_block)
            offset += D

        return jnp.concatenate(results, axis=-1)

    @staticmethod
    @jax.jit
    def _compute_fim_block_ihvp(
        activations: List[Float[Array, "N I"]],
        gradients: List[Float[Array, "N O"]],
        vectors: Float[Array, "*batch_size n_params"],
        damping: float,
    ) -> Float[Array, "*batch_size n_params"]:
        """Computes (F + λI)^(-1)v for block-diagonal FIM.

        Computes the fim_block similarly to _compute_fim_block
        and solves the linear system (F_l + λI)y = v for each block separately.
        """

        results = []
        offset = 0

        for act, grad in zip(activations, gradients):
            D = act.shape[1] * grad.shape[1]
            v_block = vectors[..., offset : offset + D]

            per_sample_vecs = jnp.einsum("ni,nj->nij", act, grad).reshape(
                act.shape[0], -1
            )

            fim_block = (per_sample_vecs.T @ per_sample_vecs) / act.shape[0]
            fim_block = fim_block + damping * jnp.eye(fim_block.shape[0])

            y_block = jnp.linalg.solve(fim_block, v_block.T).T

            results.append(y_block)
            offset += D

        return jnp.concatenate(results, axis=-1)
