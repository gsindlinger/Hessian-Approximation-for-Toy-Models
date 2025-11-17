from enum import Enum

import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jaxtyping import Array, Float


class VectorMetric(Enum):
    """Metrics for comparing (batched) HVP / IHVP result vectors."""

    RELATIVE_ERROR = "relative_error"  # ||v_1 - v_2|| / ||v_1||
    COSINE_SIMILARITY = "cosine_similarity"  # ⟨v_1, v_2⟩ / (||v_1|| * ||v_2||)
    INNER_PRODUCT_DIFF = "inner_product_diff"  # |⟨x, v_1⟩ - ⟨x, v_2⟩|
    ABSOLUTE_L2_DIFF = "absolute_l2_diff"  # ||v_1 - v_2||
    SIGN_AGREEMENT = "sign_agreement"  # fraction of same-sign coordinates
    RELATIVE_ENERGY_DIFF = "relative_energy_diff"  # |‖v_1‖² - ‖v_2‖²| / ‖v_1‖²
    INNER_PRODUCT_RATIO = "inner_product_ratio"  # ⟨x, v_2⟩ / ⟨x, v_1⟩

    def compute(
        self,
        v_1: Float[Array, "..."],
        v_2: Float[Array, "..."],
        x: Float[Array, "..."] | None = None,
    ) -> Float:
        """Compute metric for a single sample."""
        match self:
            case VectorMetric.RELATIVE_ERROR:
                return norm(v_1 - v_2) / (norm(v_1) + 1e-10)

            case VectorMetric.COSINE_SIMILARITY:
                return jnp.dot(v_1, v_2) / (norm(v_1) * norm(v_2) + 1e-10)

            case VectorMetric.INNER_PRODUCT_DIFF:
                if x is None:
                    raise ValueError("x must be provided for INNER_PRODUCT_DIFF")
                ip_gt = jnp.dot(x, v_1)
                ip_approx = jnp.dot(x, v_2)
                return jnp.abs(ip_gt - ip_approx)

            case VectorMetric.ABSOLUTE_L2_DIFF:
                return norm(v_1 - v_2)

            case VectorMetric.SIGN_AGREEMENT:
                epsilon = 1e-4
                # Agree if either value is close to zero, or if signs match
                close_to_zero = (jnp.abs(v_1) < epsilon) | (jnp.abs(v_2) < epsilon)
                sign_agree = jnp.sign(v_1) == jnp.sign(v_2)
                same_sign = jnp.mean(close_to_zero | sign_agree)
                return same_sign

            case VectorMetric.RELATIVE_ENERGY_DIFF:
                e_gt = jnp.sum(v_1**2)
                e_approx = jnp.sum(v_2**2)
                return jnp.abs(e_gt - e_approx) / (e_gt + 1e-10)

            case VectorMetric.INNER_PRODUCT_RATIO:
                if x is None:
                    raise ValueError("x must be provided for INNER_PRODUCT_RATIO")
                ip_gt = jnp.dot(x, v_1)
                ip_approx = jnp.dot(x, v_2)
                return (ip_approx + 1e-10) / (ip_gt + 1e-10)

            case _:
                raise NotImplementedError(f"Metric {self} not implemented.")

    def compute_batched(
        self,
        v_1: Float[Array, "..."],
        v_2: Float[Array, "..."],
        x: Float[Array, "..."] | None = None,
        reduction: str = "mean",
    ) -> Float[Array, "..."]:
        """
        Compute metric between (possibly batched) vector results.

        Args:
            v_1: ground truth, shape (batch, dim) or (dim,)
            v_2: approximation, same shape
            x: optional auxiliary vector(s) for inner product metrics
            reduction: "mean", "sum" or "none", can be a shared vector of shape (dim,)
                        or a batch of vectors of shape (batch, dim).
        """
        # Handle single-sample input
        if v_1.ndim == 1:
            return self.compute(v_1, v_2, x)

        # If x is shared across batch, broadcast it
        if x is not None and x.ndim == 1:
            x = jnp.broadcast_to(x, v_1.shape)

        # vmap the single-sample computation
        batched_fn = jax.vmap(
            self.compute, in_axes=(0, 0, 0 if x is not None else None)
        )
        vals = batched_fn(v_1, v_2, x)

        if reduction == "mean":
            return jnp.mean(vals)
        elif reduction == "sum":
            return jnp.sum(vals)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
