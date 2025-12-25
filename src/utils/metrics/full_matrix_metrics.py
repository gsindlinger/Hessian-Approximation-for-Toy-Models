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

        def condition_number_log_ratio(A, B):
            # | log κ(A) - log κ(B) |

            def safe_cond(M):
                M = 0.5 * (M + M.T)  # Ensure symmetry
                eps = 1e-6 * jnp.eye(M.shape[0], dtype=M.dtype)  # Regularization
                M = M + eps

                eigs = jnp.linalg.eigh(M)[0]
                eigs = jnp.clip(jnp.abs(eigs), a_min=1e-12)  # Avoid zero eigenvalues
                return jnp.max(eigs) / jnp.min(eigs)

            cond_A = safe_cond(A)
            cond_B = safe_cond(B)
            return jnp.abs(jnp.log(cond_A + 1e-12) - jnp.log(cond_B + 1e-12))

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
