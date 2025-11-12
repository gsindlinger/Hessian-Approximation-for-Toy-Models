from enum import Enum

import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jaxtyping import Array, Float


class VectorMetric(Enum):
    """Metrics for comparing (batched) HVP / IHVP result vectors."""

    RELATIVE_ERROR = "relative_error"  # ||v_gt - v_approx|| / ||v_gt||
    COSINE_SIMILARITY = (
        "cosine_similarity"  # ⟨v_gt, v_approx⟩ / (||v_gt|| * ||v_approx||)
    )
    INNER_PRODUCT_DIFF = "inner_product_diff"  # |⟨x, v_gt⟩ - ⟨x, v_approx⟩|
    ABSOLUTE_L2_DIFF = "absolute_l2_diff"  # ||v_gt - v_approx||
    SIGN_AGREEMENT = "sign_agreement"  # fraction of same-sign coordinates
    RELATIVE_ENERGY_DIFF = "relative_energy_diff"  # |‖v_gt‖² - ‖v_approx‖²| / ‖v_gt‖²
    INNER_PRODUCT_RATIO = "inner_product_ratio"  # ⟨x, v_approx⟩ / ⟨x, v_gt⟩

    def compute(
        self, v_gt: jnp.ndarray, v_approx: jnp.ndarray, x: jnp.ndarray | None = None
    ) -> Float:
        """Compute metric for a single sample."""
        match self:
            case VectorMetric.RELATIVE_ERROR:
                return norm(v_gt - v_approx) / (norm(v_gt) + 1e-10)

            case VectorMetric.COSINE_SIMILARITY:
                return jnp.dot(v_gt, v_approx) / (norm(v_gt) * norm(v_approx) + 1e-10)

            case VectorMetric.INNER_PRODUCT_DIFF:
                if x is None:
                    raise ValueError("x must be provided for INNER_PRODUCT_DIFF")
                ip_gt = jnp.dot(x, v_gt)
                ip_approx = jnp.dot(x, v_approx)
                return jnp.abs(ip_gt - ip_approx)

            case VectorMetric.ABSOLUTE_L2_DIFF:
                return norm(v_gt - v_approx)

            case VectorMetric.SIGN_AGREEMENT:
                epsilon = 1e-4
                # Agree if either value is close to zero, or if signs match
                close_to_zero = (jnp.abs(v_gt) < epsilon) | (
                    jnp.abs(v_approx) < epsilon
                )
                sign_agree = jnp.sign(v_gt) == jnp.sign(v_approx)
                same_sign = jnp.mean(close_to_zero | sign_agree)
                return same_sign

            case VectorMetric.RELATIVE_ENERGY_DIFF:
                e_gt = jnp.sum(v_gt**2)
                e_approx = jnp.sum(v_approx**2)
                return jnp.abs(e_gt - e_approx) / (e_gt + 1e-10)

            case VectorMetric.INNER_PRODUCT_RATIO:
                if x is None:
                    raise ValueError("x must be provided for INNER_PRODUCT_RATIO")
                ip_gt = jnp.dot(x, v_gt)
                ip_approx = jnp.dot(x, v_approx)
                return (ip_approx + 1e-10) / (ip_gt + 1e-10)

            case _:
                raise NotImplementedError(f"Metric {self} not implemented.")

    def compute_batched(
        self,
        v_gt: jnp.ndarray,
        v_approx: jnp.ndarray,
        x: jnp.ndarray | None = None,
        reduction: str = "mean",
    ) -> Float[Array, ...]:
        """
        Compute metric between (possibly batched) vector results.

        Args:
            v_gt: ground truth, shape (batch, dim) or (dim,)
            v_approx: approximation, same shape
            x: optional auxiliary vector(s) for inner product metrics
            reduction: "mean", "sum" or "none", can be a shared vector of shape (dim,)
                        or a batch of vectors of shape (batch, dim).
        """
        # Handle single-sample input
        if v_gt.ndim == 1:
            return self.compute(v_gt, v_approx, x)

        # If x is shared across batch, broadcast it
        if x is not None and x.ndim == 1:
            x = jnp.broadcast_to(x, v_gt.shape)

        # vmap the single-sample computation
        batched_fn = jax.vmap(
            self.compute, in_axes=(0, 0, 0 if x is not None else None)
        )
        vals = batched_fn(v_gt, v_approx, x)

        if reduction == "mean":
            return jnp.mean(vals)
        elif reduction == "sum":
            return jnp.sum(vals)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
