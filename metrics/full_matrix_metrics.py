from enum import Enum
from typing import Dict, List

import jax.numpy as jnp
from jax.numpy.linalg import eigh, norm
from jaxtyping import Array, Float


class FullMatrixMetric(Enum):
    """Metrics for comparing full Hessian matrices."""

    # Element-wise norms
    RELATIVE_FROBENIUS = "relative_frobenius"  # Scale-invariant version
    MAX_ELEMENTWISE = "max_elementwise"  # Largest absolute difference

    # Spectral properties
    SPECTRAL = "spectral"
    RELATIVE_SPECTRAL = "spectral_relative"  # Scale-invariant version

    # Structural similarity
    COSINE_SIMILARITY = (
        "cosine_similarity"  # tr(A @ B.T) / (||A||_F * ||B||_F), scale-invariant
    )

    # Eigenvalue-based (critical for optimization properties)
    EIGENVALUE_MAX = "eigenvalue_max"  # Max distance between eigenvalues
    EIGENVALUES_L2_DISTANCE = (
        "eigenvalues_l2_distance"  # Relative L2 distance of eigenvalues
    )

    # Cheap summary statistics
    TRACE_DISTANCE = "trace_distance"  # Difference in traces (divided by size)
    CONDITION_NUMBER_LOG_RATIO = "condition_number_log_ratio"  # Invertibility quality

    def compute(
        self, matrix_1: Float[Array, "..."], matrix_2: Float[Array, "..."]
    ) -> float:
        """
        Compute the metric between two matrices.
        """
        match self:
            case FullMatrixMetric.RELATIVE_FROBENIUS:
                # ||A - B||_F / ||A||_F
                norm_diff = norm(matrix_1 - matrix_2, ord="fro")
                norm_a = norm(matrix_1, ord="fro")
                return norm_diff / (norm_a + 1e-10)  # Avoid division by zero
            case FullMatrixMetric.MAX_ELEMENTWISE:
                # max|a_ij - b_ij|
                return jnp.max(jnp.abs(matrix_1 - matrix_2))  # type: ignore
            case FullMatrixMetric.SPECTRAL:
                # ||A - B||_2
                return float(norm(matrix_1 - matrix_2, ord=2))
            case FullMatrixMetric.RELATIVE_SPECTRAL:
                # ||A - B||_2 / ||A||_2 (scale-invariant)
                norm_diff = norm(matrix_1 - matrix_2, ord=2)
                norm_a = norm(matrix_1, ord=2)
                return norm_diff / (norm_a + 1e-10)  # Avoid division by zero
            case FullMatrixMetric.COSINE_SIMILARITY:
                # tr(A @ B^T) / (||A||_F * ||B||_F)
                inner_product = jnp.trace(matrix_1 @ matrix_2.T)
                norm_a = norm(matrix_1, ord="fro")
                norm_b = norm(matrix_2, ord="fro")
                cosine = inner_product / (norm_a * norm_b + 1e-10)
                return cosine  # 1 = perfect match, 0 = orthogonal
            case FullMatrixMetric.EIGENVALUE_MAX:
                # max|λ_i(A) - λ_i(B)|
                eigs_a = eigh(matrix_1)[0]
                eigs_b = eigh(matrix_2)[0]
                return float(jnp.max(jnp.abs(eigs_a - eigs_b)))
            case FullMatrixMetric.EIGENVALUES_L2_DISTANCE:
                # Relative eigenvalue error
                eigs_a = eigh(matrix_1)[0]
                eigs_b = eigh(matrix_2)[0]
                l2_diff = norm(eigs_a - eigs_b)
                l2_a = norm(eigs_a)
                return float(l2_diff / (l2_a + 1e-10))
            case FullMatrixMetric.TRACE_DISTANCE:
                # |tr(A) - tr(B)| - sum of eigenvalues, divide by size for scale-invariance
                return (
                    float(jnp.abs(jnp.trace(matrix_1) - jnp.trace(matrix_2)))
                    / matrix_1.shape[0]
                )
            case FullMatrixMetric.CONDITION_NUMBER_LOG_RATIO:
                # κ(A) / κ(B)
                def safe_cond(M):
                    eigs = jnp.abs(eigh(M)[0])
                    max_eig = jnp.max(eigs)
                    min_eig = jnp.min(eigs)
                    return max_eig / (min_eig + 1e-10)

                return float(
                    jnp.abs(jnp.log(safe_cond(matrix_1)) - jnp.log(safe_cond(matrix_2)))
                )

            case _:
                raise NotImplementedError(f"Metric {self} not implemented.")


MATRIX_METRICS = {
    "comprehensive": [
        FullMatrixMetric.RELATIVE_FROBENIUS,
        FullMatrixMetric.COSINE_SIMILARITY,
        FullMatrixMetric.TRACE_DISTANCE,
    ],
    "all_matrix": list(FullMatrixMetric),
}


def compare_matrices(
    matrix_1, matrix_2, metrics: List[FullMatrixMetric] | None = None
) -> Dict[str, float]:
    """
    Compare two matrices using specified metrics.

    Args:
        matrix_gt: Ground truth matrix (e.g., true Hessian)
        matrix_approx: Comparison matrix (e.g., approximation)
        metrics: List of metrics to compute

    Returns:
        Dictionary mapping metric names to values
    """
    if metrics is None:
        metrics = MATRIX_METRICS["comprehensive"]

    results = {}
    for metric in metrics:
        results[metric.value] = float(
            metric.compute(matrix_1=matrix_1, matrix_2=matrix_2)
        )

    return results
