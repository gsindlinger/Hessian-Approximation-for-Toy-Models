from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import Array, Float
from typing_extensions import override

from ...models.dataclasses.hessian_compute_context import HessianComputeContext
from ...models.utils.loss import loss_wrapper_with_apply_fn
from ..hessian_approximations import HessianApproximation


@dataclass
class Hessian(HessianApproximation):
    """Hessian Calculation via automatic differentiation (JAX native)."""

    @staticmethod
    def get_param_index_mapping(params: Dict):
        """
        Build a mapping from parameter names to index ranges in the flattened vector.
        Helps to debug by identifying which parameter in the flattened parameter array corresponds to which entry.
        Returns:
            dict[str, tuple[int, int]] mapping each param path to (start, end)
        """
        leaves, _ = jax.tree_util.tree_flatten(params)
        flat_params, _ = flatten_util.ravel_pytree(params)

        index_mapping = {}
        idx = 0

        # Traverse parameter tree with paths
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
            path_str = "/".join(str(k) for k in path)
            size = leaf.size
            index_mapping[path_str] = (idx, idx + size)
            idx += size

        return index_mapping, flat_params.size

    @override
    def compute_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> jnp.ndarray:
        """
        Compute the exact Hessian matrix of the loss function for the whole model.
        Args:
            damping: Optional damping factor to add to the diagonal for numerical stability.
        """
        return self._compute_hessian(
            HessianComputeContext.get_data_and_params_for_hessian(self.model_context),
            damping=0.0 if damping is None else damping,
        )

    @staticmethod
    @jax.jit
    def _compute_hessian(
        data: HessianComputeContext,
        damping: Float,
    ) -> Float[Array, "n_params n_params"]:
        """
        JIT-compiled Hessian computation.
        Returns the Hessian matrix as a 2D array.
        """
        # Important: Flattening structure for linear modules with bias is the following: b, w
        # So for output dim 2, input dim 3, the order is: b1, b2, w1

        def loss_wrapper(p):
            return loss_wrapper_with_apply_fn(
                data.model_apply_fn,
                p,
                data.unravel_fn,
                data.loss_fn,
                data.training_data,
                data.training_targets,
            )

        return jax.hessian(loss_wrapper)(data.params_flat) + damping * jnp.eye(
            data.num_params
        )

    @override
    def compute_hvp(
        self,
        vectors: Float[Array, "n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params"]:
        """
        Compute the Hessian-vector product (HVP).
        """
        training_data, training_targets = self.model_context.dataset.get_train_data()

        # Flatten parameters once
        params_flat, unravel_fn = flatten_util.ravel_pytree(self.model_context.params)

        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        # Compute batched HVP
        result: Float[Array, "batch_size n_params"] = self.compute_hvp_batched(
            HessianComputeContext(
                training_data=training_data,
                training_targets=training_targets,
                params_flat=params_flat,
                unravel_fn=unravel_fn,
                model_apply_fn=self.model_context.model.apply,
                loss_fn=self.model_context.loss,
            ),
            vectors_2D,
            damping=0.0 if damping is None else damping,
        )
        return result.squeeze(0) if is_single else result

    @staticmethod
    @jax.jit
    def compute_hvp_batched(
        data: HessianComputeContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "n_params"]:
        """
        JIT-compiled Hessian-vector product (HVP) computation.
        """
        # Vectorize over the batch dimension
        return jax.vmap(lambda v: Hessian.compute_hvp_single(data, v, damping))(vectors)

    @staticmethod
    @jax.jit
    def compute_hvp_single(
        data: HessianComputeContext,
        vector: Float[Array, "n_params"],
        damping: Float,
    ) -> Float[Array, "n_params"]:
        """
        JIT-compiled Hessian-vector product (HVP) computation.
        """

        def loss_wrapper(p):
            return loss_wrapper_with_apply_fn(
                data.model_apply_fn,
                p,
                data.unravel_fn,
                data.loss_fn,
                data.training_data,
                data.training_targets,
            )

        # Use jax.jvp for efficient Hessian-vector product
        _, hvp_result = jax.jvp(
            lambda p: jax.grad(loss_wrapper)(p),
            (data.params_flat,),
            (vector,),
        )
        return hvp_result + damping * vector

    @override
    def compute_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params"]:
        """
        Compute the inverse Hessian-vector product (IHVP).
        """
        return self.compute_ihvp_batched(
            HessianComputeContext.get_data_and_params_for_hessian(self.model_context),
            vectors,
            damping=0.0 if damping is None else damping,
        )

    @staticmethod
    @jax.jit
    def compute_ihvp_batched(
        data: HessianComputeContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """
        JIT-compiled Inverse Hessian-vector product (IHVP) computation.
        """
        # Vectorize over the batch dimension
        hessian = Hessian._compute_hessian(data, damping)
        return jnp.linalg.solve(hessian, vectors.T).T
