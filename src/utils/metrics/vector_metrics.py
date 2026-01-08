from enum import Enum
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class VectorMetric(str, Enum):
    """Metrics for comparing (batched) HVP / IHVP result vectors."""

    ABSOLUTE_L2_DIFF = "absolute_l2_diff"  # ||v_1 - v_2||
    RELATIVE_ERROR = "relative_error"  # ||v_1 - v_2|| / ||v_1||
    COSINE_SIMILARITY = "cosine_similarity"  # ⟨v_1, v_2⟩ / (||v_1|| * ||v_2||)
    INNER_PRODUCT_DIFF = "inner_product_diff"  # |⟨x, v_1⟩ - ⟨x, v_2⟩|
    RELATIVE_ENERGY_DIFF = "relative_energy_diff"  # |‖v_1‖² - ‖v_2‖²| / ‖v_1‖²
    INNER_PRODUCT_RATIO = "inner_product_ratio"  # ⟨x, v_2⟩ / ⟨x, v_1⟩

    def compute_fn(self) -> Callable:
        """Return to the corresponding enum belonging metric as function / callable."""

        # ------------------------------------------------------
        # Metric definitions (all pure JAX)
        # ------------------------------------------------------

        # ||v_1 - v_2|| / ||v_1||
        def relative_error(v1, v2, x=None):
            return jnp.linalg.norm(v1 - v2) / (jnp.linalg.norm(v1) + 1e-10)

        # ⟨v_1, v_2⟩ / (||v_1|| * ||v_2||)
        def cosine_similarity(v1, v2, x=None):
            dot = jnp.dot(v1, v2)
            nrm = jnp.linalg.norm(v1) * jnp.linalg.norm(v2)
            return dot / (nrm + 1e-10)

        # |⟨x, v_1⟩ - ⟨x, v_2⟩|
        def inner_product_diff(v1, v2, x):
            if x is None:
                raise ValueError("inner_product_diff requires auxiliary vector x")
            ip1 = jnp.dot(x, v1)
            ip2 = jnp.dot(x, v2)
            return jnp.abs(ip1 - ip2)

        # ||v_1 - v_2||
        def absolute_l2_diff(v1, v2, x=None):
            return jnp.linalg.norm(v1 - v2)

        # |‖v_1‖² - ‖v_2‖²| / ‖v_1‖²
        def relative_energy_diff(v1, v2, x=None):
            e1 = jnp.sum(v1**2)
            e2 = jnp.sum(v2**2)
            return jnp.abs(e1 - e2) / (e1 + 1e-10)

        # |⟨x, v_2⟩ / ⟨x, v_1⟩|
        def inner_product_ratio(v1, v2, x):
            if x is None:
                raise ValueError("inner_product_ratio requires auxiliary vector x")
            ip1 = jnp.dot(x, v1)
            ip2 = jnp.dot(x, v2)
            return jnp.abs((ip2 + 1e-10) / (ip1 + 1e-10))

        # ------------------------------------------------------
        # Dispatch table (Enum → pure JAX function)
        # ------------------------------------------------------
        table = {
            VectorMetric.RELATIVE_ERROR: relative_error,
            VectorMetric.COSINE_SIMILARITY: cosine_similarity,
            VectorMetric.INNER_PRODUCT_DIFF: inner_product_diff,
            VectorMetric.ABSOLUTE_L2_DIFF: absolute_l2_diff,
            VectorMetric.RELATIVE_ENERGY_DIFF: relative_energy_diff,
            VectorMetric.INNER_PRODUCT_RATIO: inner_product_ratio,
        }

        return jax.jit(table[self])

    # ----------------------------------------------------------
    #           Single-sample evaluation
    # ----------------------------------------------------------
    def compute_single(
        self,
        v1: Float[Array, "..."],
        v2: Float[Array, "..."],
        x: Optional[Float[Array, "..."]] = None,
    ) -> Float:
        fn = self.compute_fn()
        return fn(v1, v2, x)

    # ----------------------------------------------------------
    #             Batched evaluation
    # ----------------------------------------------------------

    def compute(
        self,
        v1: Float[Array, "*batch_size n_params"],
        v2: Float[Array, "*batch_size n_params"],
        x: Optional[Float[Array, "*batch_size n_params"]] = None,
        reduction: str = "mean",
    ) -> Float:
        """
        Compute metric across a batch of vector pairs.

        v1, v2: shape (batch, dim) or (dim,) - <em>v1: ground truth</em> and <em>v2: approximation</em>
        x: optional auxiliary vector(s) (same batching rules as v1/v2)
        reduction: "mean" or "sum"
        """

        # Single-sample case
        if v1.ndim == 1:
            return self.compute_single(v1, v2, x)

        # Determine in_axes for x based on its dimensionality
        if x is None:
            x_axis = None
        elif x.ndim == 1:
            x_axis = None  # broadcast single x to all batch elements
        else:
            x_axis = 0  # x is already batched

        fn = self.compute_fn()

        batched_fn = jax.vmap(fn, in_axes=(0, 0, x_axis))
        values = batched_fn(v1, v2, x)

        if reduction == "mean":
            return jnp.mean(values)
        elif reduction == "sum":
            return jnp.sum(values)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    @staticmethod
    def all_metrics() -> list["VectorMetric"]:
        """Return a list of all available vector metrics."""
        return list(VectorMetric)
