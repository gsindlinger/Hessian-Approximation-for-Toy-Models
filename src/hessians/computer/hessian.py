from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import Array, Float

from src.hessians.utils.data import ModelContext


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
        result: Float[Array, "batch_size n_params"] = self._compute_hvp(
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
        return self._compute_ihvp(
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
    def _compute_hvp(
        compute_context: ModelContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Memory-efficient Hessian-vector product computation.
        Uses scan over samples and vmap over vectors, matching GNH strategy.
        """
        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets

        assert Y is not None, (
            "Targets must be provided in ModelContext for HVP computation."
        )

        def model_out(p, x):
            params = compute_context.unravel_fn(p)
            return compute_context.model_apply_fn(params, x[None, ...]).squeeze(0)

        def loss_single(p, x, y):
            """Loss for a single sample"""
            z = model_out(p, x)
            return compute_context.loss_fn(z, y)

        # Per-vector HVP function (scans over samples)
        @jax.jit
        def hvp_single(v):
            """Compute H @ v by accumulating over samples"""

            def body_fn(accum, xy):
                x_i, y_i = xy

                # Compute per-sample Hessian-vector product
                # hvp = ∇²L_i @ v for sample i
                def grad_fn(p):
                    return jax.grad(lambda p_: loss_single(p_, x_i, y_i))(p)

                # Use JVP to compute Hessian-vector product efficiently
                _, hvp_i = jax.jvp(grad_fn, (p_flat,), (v,))

                return accum + hvp_i, None

            # Accumulate HVP contributions across all samples
            summed, _ = jax.lax.scan(body_fn, jnp.zeros_like(v), (X, Y))

            # Average and add damping
            hvp = summed / X.shape[0]
            return hvp + damping * v

        # ------------------------------------------------------------
        # Chunking vectors to avoid OOM
        # ------------------------------------------------------------
        CHUNK_SIZE = 32
        n_vectors = vectors.shape[0]

        if n_vectors <= CHUNK_SIZE:
            return jax.vmap(hvp_single)(vectors)
        else:
            outs = []
            for i in range(0, n_vectors, CHUNK_SIZE):
                chunk = vectors[i : i + CHUNK_SIZE]
                outs.append(jax.vmap(hvp_single)(chunk))
            return jnp.concatenate(outs, axis=0)

    @staticmethod
    @jax.jit
    def _compute_ihvp(
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
