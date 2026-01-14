import logging
from enum import Enum
from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

logger = logging.getLogger(__name__)


class FullMatrixMetric(str, Enum):
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
    CONDITION_NUMBER_LOG_RATIO = "condition_number_log_ratio"  # Invertibility quality

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

        def condition_number_log_ratio(A, B, threshold=15000, iterations=100):
            def symmetrize(M):
                return 0.5 * (M + M.T)

            def get_cond_exact(M):
                eigs = jnp.linalg.eigh(M)[0]
                eigs = jnp.clip(jnp.abs(eigs), a_min=1e-12)
                return jnp.log(jnp.max(eigs)) - jnp.log(jnp.min(eigs))

            def get_log_cond_iterative(M, iters):
                logger.error("Iterative log-cond not implemented yet.")
                return 0.0

            def compute_log_cond(M):
                M = symmetrize(M)
                n = M.shape[0]
                # Return log_cond directly to avoid large number overflows
                if n <= threshold:
                    return get_cond_exact(M)
                else:
                    return get_log_cond_iterative(M, iterations)

            log_cond_A = compute_log_cond(A)
            log_cond_B = compute_log_cond(B)

            return jnp.abs(log_cond_A - log_cond_B)

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
            FullMatrixMetric.CONDITION_NUMBER_LOG_RATIO: condition_number_log_ratio,
        }

        return jax.jit(table[self])

    def compute(
        self, matrix_1: Float[Array, "..."], matrix_2: Float[Array, "..."]
    ) -> Float:
        return self.compute_fn()(matrix_1, matrix_2)

    @staticmethod
    def all_metrics(
        exclude: Optional[List["FullMatrixMetric"]] = None,
    ) -> List["FullMatrixMetric"]:
        if exclude is None:
            exclude = []
        return [metric for metric in FullMatrixMetric if metric not in exclude]


# ----------------------------------------------------------
# Matrix comparison wrapper
# ----------------------------------------------------------


def compare_matrices(
    matrix_1, matrix_2, metrics: List[FullMatrixMetric] | None = None
) -> Dict[str, float]:
    if metrics is None:
        metrics = FullMatrixMetric.all_metrics()

    results = {}

    for metric in metrics:
        fn = metric.compute_fn()
        value = fn(matrix_1, matrix_2)
        results[metric.value] = float(value)

    return results
