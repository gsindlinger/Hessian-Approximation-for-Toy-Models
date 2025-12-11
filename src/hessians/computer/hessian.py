from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import Array, Float

from src.hessians.utils.data import ModelContext
from src.utils.loss import loss_wrapper_with_apply_fn


@dataclass
class HessianComputer:
    """Hessian Calculation via automatic differentiation (JAX native)."""

    compute_context: ModelContext

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
            compute_context=self.compute_context,
            damping=0.0 if damping is None else damping,
        )

    def compute_hvp(
        self,
        vectors: Float[Array, "n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params"]:
        """
        Compute the Hessian-vector product (HVP).
        """

        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        # Compute batched HVP
        result: Float[Array, "batch_size n_params"] = self._compute_hvp_batched(
            compute_context=self.compute_context,
            vectors=vectors_2D,
            damping=0.0 if damping is None else damping,
        )
        return result.squeeze(0) if is_single else result

    def compute_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params"]:
        """
        Compute the inverse Hessian-vector product (IHVP).
        """
        return self.compute_ihvp_batched(
            self.compute_context,
            vectors,
            damping=0.0 if damping is None else damping,
        )

    @staticmethod
    @jax.jit
    def _compute_hessian(
        compute_context: ModelContext,
        damping: Float,
    ) -> Float[Array, "n_params n_params"]:
        def loss_single(p, x, y):
            params_unflat = compute_context.unravel_fn(p)
            preds = compute_context.model_apply_fn(params_unflat, x[None, ...])
            return compute_context.loss_fn(preds.squeeze(0), y)

        # This computes the per-sample Hessian: ∂²L_i/∂θ²
        @jax.jit
        def compute_sample_hessian(p_flat, x, y):
            return jax.hessian(lambda p: loss_single(p, x, y))(p_flat)

        def scan_body(carry, xy):
            p_flat, H = carry
            x_i, y_i = xy
            H_i = compute_sample_hessian(p_flat, x_i, y_i)
            return (p_flat, H + H_i), None

        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets

        H0 = jnp.zeros((p_flat.size, p_flat.size))

        (_, H_full), _ = jax.lax.scan(scan_body, init=(p_flat, H0), xs=(X, Y))

        H_full = H_full / X.shape[0]

        return H_full + damping * jnp.eye(H_full.shape[0])

    @staticmethod
    @jax.jit
    def _compute_hvp_batched(
        compute_context: ModelContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "n_params"]:
        """
        JIT-compiled Hessian-vector product (HVP) computation.
        """
        # Vectorize over the batch dimension
        return jax.vmap(
            lambda v: HessianComputer._compute_hvp_single(compute_context, v, damping)
        )(vectors)

    @staticmethod
    @jax.jit
    def _compute_hvp_single(
        compute_context: ModelContext,
        vector: Float[Array, "n_params"],
        damping: Float,
    ) -> Float[Array, "n_params"]:
        """
        JIT-compiled Hessian-vector product (HVP) computation.
        """

        targets = compute_context.targets
        assert targets is not None, (
            "Targets must be provided in ModelContext for HVP computation."
        )

        def loss_wrapper(p):
            return loss_wrapper_with_apply_fn(
                compute_context.model_apply_fn,
                p,
                compute_context.unravel_fn,
                compute_context.loss_fn,
                compute_context.inputs,
                targets,
            )

        # Use jax.jvp for efficient Hessian-vector product
        _, hvp_result = jax.jvp(
            lambda p: jax.grad(loss_wrapper)(p),
            (compute_context.params_flat,),
            (vector,),
        )
        return hvp_result + damping * vector

    @staticmethod
    @jax.jit
    def compute_ihvp_batched(
        compute_context: ModelContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """
        JIT-compiled Inverse Hessian-vector product (IHVP) computation.
        """
        # Vectorize over the batch dimension
        hessian = HessianComputer._compute_hessian(compute_context, damping)
        return jnp.linalg.solve(hessian, vectors.T).T
