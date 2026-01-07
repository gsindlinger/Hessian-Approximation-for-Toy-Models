from enum import Enum
from typing import Callable, Dict, List

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class FullMatrixMetric(Enum):
    """Metrics for comparing full Hessian matrices."""

    # Element-wise norms
    FROBENIUS = "frobenius"  # ||A - B||_F
    RELATIVE_FROBENIUS = "relative_frobenius"  # Scale-invariant version

    # Spectral properties
    SPECTRAL = "spectral"
    RELATIVE_SPECTRAL = "spectral_relative"  # Scale-invariant version

    # Structural similarity
    COSINE_SIMILARITY = (
        "cosine_similarity"  # tr(A @ B.T) / (||A||_F * ||B||_F), scale-invariant
    )

    # Cheap summary statistics
    TRACE_DISTANCE = (
        "trace_distance"  # Difference in traces (divided by size) (semirelevant)
    )
    # CONDITION_NUMBER_LOG_RATIO = "condition_number_log_ratio"  # Invertibility quality

    def compute_fn(self) -> Callable:
        """Return function / metric for the corresponding metric."""

        # ------------------------------------------------------
        # Element-wise norms
        # ------------------------------------------------------

        def frobenius(A, B):
            # ||A - B||_F
            return jnp.linalg.norm(A - B, ord="fro")

        def relative_frobenius(A, B):
            # ||A - B||_F / ||A||_F   (scale-invariant)
            norm_diff = jnp.linalg.norm(A - B, ord="fro")
            norm_a = jnp.linalg.norm(A, ord="fro")
            return norm_diff / (norm_a + 1e-10)  # Avoid division by zero

        # ------------------------------------------------------
        # Spectral norms
        # ------------------------------------------------------

        def spectral(A, B):
            # ||A - B||_2 (spectral/operator norm)
            return jnp.linalg.norm(A - B, ord=2)

        def relative_spectral(A, B):
            # ||A - B||_2 / ||A||_2   (scale-invariant)
            norm_diff = jnp.linalg.norm(A - B, ord=2)
            norm_a = jnp.linalg.norm(A, ord=2)
            return norm_diff / (norm_a + 1e-10)

        # ------------------------------------------------------
        # Structural similarity
        # ------------------------------------------------------

        def cosine_similarity(A, B):
            # tr(A @ B^T) / (||A||_F * ||B||_F)
            inner = jnp.vdot(A, B)
            norm_a = jnp.linalg.norm(A, ord="fro")
            norm_b = jnp.linalg.norm(B, ord="fro")
            cosine = inner / (norm_a * norm_b + 1e-10)
            return cosine  # 1 = perfect match, 0 = orthogonal

        # ------------------------------------------------------
        # Cheap summary statistics
        # ------------------------------------------------------

        def trace_distance(A, B):
            # |tr(A) - tr(B)| / n   (scale-invariant)
            return jnp.abs(jnp.trace(A) - jnp.trace(B)) / A.shape[0]

        # def condition_number_log_ratio(A, B, threshold=15000, iterations=100):
        #     def symmetrize(M):
        #         return 0.5 * (M + M.T)

        #     def get_cond_exact(M):
        #         eigs = jnp.linalg.eigh(M)[0]
        #         eigs = jnp.clip(jnp.abs(eigs), a_min=1e-12)
        #         return jnp.log(jnp.max(eigs)) - jnp.log(jnp.min(eigs))

        #     def get_log_cond_iterative(M, iters):
        #         n = M.shape[0]

        #         key = jax.random.fold_in(jax.random.PRNGKey(0), M.shape[0])
        #         k1, k2 = jax.random.split(key)

        #         # 1. Lambda Max with simple momentum to help clustered eigs
        #         v = jax.random.normal(k1, (n,), dtype=M.dtype)
        #         v /= jnp.linalg.norm(v) + 1e-12

        #         def body_max(carry, _):
        #             v, v_old = carry
        #             v_next = M @ v
        #             # Subtle momentum (0.1) helps avoid stalling on clusters
        #             v_next = v_next + 0.1 * (v - v_old)
        #             v_next = v_next / (jnp.linalg.norm(v_next) + 1e-12)
        #             return (v_next, v), None

        #         (v_max, _), _ = jax.lax.scan(body_max, (v, v), None, length=iters)
        #         l_max = jnp.abs(jnp.dot(v_max, M @ v_max))

        #         # 2. Lambda Min via Inverse Iteration (Approx)
        #         v = jax.random.normal(k2, (n,), dtype=M.dtype)
        #         v /= jnp.linalg.norm(v) + 1e-12

        #         # We shift by l_max to find the smallest, but use
        #         # a small epsilon to keep it positive definite.
        #         shift = l_max * 1.0001

        #         def body_min(v, _):
        #             v = shift * v - M @ v
        #             return v / (jnp.linalg.norm(v) + 1e-12), None

        #         v_min, _ = jax.lax.scan(body_min, v, None, length=iters)
        #         l_shift = jnp.dot(v_min, shift * v_min - M @ v_min)
        #         l_min = jnp.clip(shift - l_shift, a_min=1e-12)

        #         return jnp.log(l_max) - jnp.log(l_min)

        #     def compute_log_cond(M):
        #         M = symmetrize(M)
        #         n = M.shape[0]
        #         # Return log_cond directly to avoid large number overflows
        #         if n <= threshold:
        #             return get_cond_exact(M)
        #         else:
        #             return get_log_cond_iterative(M, iterations)

        #     log_cond_A = compute_log_cond(A)
        #     log_cond_B = compute_log_cond(B)

        #     return jnp.abs(log_cond_A - log_cond_B)

        # ------------------------------------------------------
        # Dispatch table mapping enum → pure JAX function
        # ------------------------------------------------------
        table = {
            FullMatrixMetric.FROBENIUS: frobenius,
            FullMatrixMetric.RELATIVE_FROBENIUS: relative_frobenius,
            FullMatrixMetric.SPECTRAL: spectral,
            FullMatrixMetric.RELATIVE_SPECTRAL: relative_spectral,
            FullMatrixMetric.COSINE_SIMILARITY: cosine_similarity,
            FullMatrixMetric.TRACE_DISTANCE: trace_distance,
            # FullMatrixMetric.CONDITION_NUMBER_LOG_RATIO: condition_number_log_ratio,
        }

        return jax.jit(table[self])

    def compute(
        self, matrix_1: Float[Array, "..."], matrix_2: Float[Array, "..."]
    ) -> Float:
        return self.compute_fn()(matrix_1, matrix_2)


# ----------------------------------------------------------
# Metric groups
# ----------------------------------------------------------

MATRIX_METRICS = {
    "comprehensive": [
        FullMatrixMetric.RELATIVE_FROBENIUS,
        FullMatrixMetric.COSINE_SIMILARITY,
        FullMatrixMetric.TRACE_DISTANCE,
    ],
    "all_matrix": list(FullMatrixMetric),
}


# ----------------------------------------------------------
# Matrix comparison wrapper
# ----------------------------------------------------------


def compare_matrices(
    matrix_1, matrix_2, metrics: List[FullMatrixMetric] | None = None
) -> Dict[str, float]:
    if metrics is None:
        metrics = MATRIX_METRICS["comprehensive"]

    results = {}

    for metric in metrics:
        fn = metric.compute_fn()
        value = fn(matrix_1, matrix_2)
        results[metric.value] = float(value)

    return results
