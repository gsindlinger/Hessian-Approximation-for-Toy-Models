from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import Array, Float
from typing_extensions import override

from hessian_approximations.hessian_approximations import HessianApproximation
from models.dataclasses.hessian_compute_context import HessianComputeContext
from models.utils.loss import loss_wrapper_with_apply_fn


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
        compute_data: HessianComputeContext,
        damping: Float,
    ) -> Float[Array, "n_params n_params"]:
        def loss_single(p, x, y):
            params_unflat = compute_data.unravel_fn(p)
            preds = compute_data.model_apply_fn(params_unflat, x[None, ...])
            return compute_data.loss_fn(preds.squeeze(0), y)

        # This computes the per-sample Hessian: ∂²L_i/∂θ²
        @jax.jit
        def compute_sample_hessian(p_flat, x, y):
            return jax.hessian(lambda p: loss_single(p, x, y))(p_flat)

        def scan_body(carry, xy):
            p_flat, H = carry
            x_i, y_i = xy
            H_i = compute_sample_hessian(p_flat, x_i, y_i)
            return (p_flat, H + H_i), None

        p_flat = compute_data.params_flat
        X = compute_data.training_data
        Y = compute_data.training_targets

        H0 = jnp.zeros((p_flat.size, p_flat.size))

        (_, H_full), _ = jax.lax.scan(scan_body, init=(p_flat, H0), xs=(X, Y))

        H_full = H_full / X.shape[0]

        return H_full + damping * jnp.eye(H_full.shape[0])

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
