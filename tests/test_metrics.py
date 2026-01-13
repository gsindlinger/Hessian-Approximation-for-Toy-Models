import jax.numpy as jnp
import pytest

from src.utils.metrics.full_matrix_metrics import FullMatrixMetric, compare_matrices
from src.utils.metrics.vector_metrics import VectorMetric

# ---------------------------------------------------------------------------
# VectorMetric Tests
# ---------------------------------------------------------------------------


def test_relative_error():
    v1 = jnp.array([1.0, 2.0, 3.0])
    v2 = jnp.array([1.0, 2.0, 4.0])
    expected = jnp.linalg.norm(v1 - v2) / jnp.linalg.norm(v1)
    assert pytest.approx(VectorMetric.RELATIVE_ERROR.compute_single(v1, v2)) == expected


def test_cosine_similarity():
    v1 = jnp.array([1.0, 0.0])
    v2 = jnp.array([1.0, 0.0])
    assert VectorMetric.COSINE_SIMILARITY.compute_single(v1, v2) == pytest.approx(1.0)


def test_inner_product_diff_requires_x():
    v1, v2 = jnp.ones(3), jnp.ones(3)
    with pytest.raises(ValueError):
        VectorMetric.INNER_PRODUCT_DIFF.compute_single(v1, v2, None)


def test_inner_product_diff():
    v1 = jnp.array([1.0, 0.0])
    v2 = jnp.array([2.0, 0.0])
    x = jnp.array([1.0, 1.0])
    expected = abs(jnp.dot(x, v1) - jnp.dot(x, v2))
    assert (
        pytest.approx(VectorMetric.INNER_PRODUCT_DIFF.compute_single(v1, v2, x))
        == expected
    )


def test_absolute_l2_diff():
    v1 = jnp.array([1.0, 2.0])
    v2 = jnp.array([1.0, 0.0])
    expected = jnp.linalg.norm(v1 - v2)
    assert (
        pytest.approx(VectorMetric.ABSOLUTE_L2_DIFF.compute_single(v1, v2)) == expected
    )


def test_relative_energy_diff():
    v1 = jnp.array([1.0, 2.0])
    v2 = jnp.array([1.0, 3.0])
    e1 = jnp.sum(v1**2)
    e2 = jnp.sum(v2**2)
    expected = abs(e1 - e2) / e1
    assert (
        pytest.approx(VectorMetric.RELATIVE_ENERGY_DIFF.compute_single(v1, v2))
        == expected
    )


def test_inner_product_ratio_requires_x():
    v1, v2 = jnp.ones(3), jnp.ones(3)
    with pytest.raises(ValueError):
        VectorMetric.INNER_PRODUCT_RATIO.compute_single(v1, v2, None)


def test_inner_product_ratio():
    v1 = jnp.array([1.0, 1.0])
    v2 = jnp.array([2.0, 2.0])
    x = jnp.array([1.0, 1.0])
    expected = jnp.dot(x, v2) / jnp.dot(x, v1)
    assert (
        pytest.approx(VectorMetric.INNER_PRODUCT_RATIO.compute_single(v1, v2, x))
        == expected
    )


def test_compute_batched_mean_reduction():
    v1 = jnp.array([[1.0, 1.0], [2.0, 2.0]])
    v2 = jnp.array([[1.0, 0.0], [2.0, 3.0]])
    metric = VectorMetric.ABSOLUTE_L2_DIFF

    expected = jnp.mean(
        jnp.array(
            [
                jnp.linalg.norm(v1[0] - v2[0]),
                jnp.linalg.norm(v1[1] - v2[1]),
            ]
        )
    )

    assert metric.compute(v1, v2) == pytest.approx(expected)


def test_compute_batched_sum_reduction():
    v1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    v2 = jnp.array([[1.0, 0.0], [3.0, 5.0]])
    metric = VectorMetric.RELATIVE_ERROR

    vals = [
        metric.compute_single(v1[0], v2[0]),
        metric.compute_single(v1[1], v2[1]),
    ]
    expected = jnp.sum(jnp.array(vals))

    assert metric.compute(v1, v2, reduction="sum") == pytest.approx(expected)


def test_compute_batched_broadcast_x():
    v1 = jnp.array([[1.0, 1.0], [2.0, 2.0]])
    v2 = jnp.array([[1.0, 0.0], [2.0, 3.0]])
    x = jnp.array([1.0, 1.0])
    metric = VectorMetric.INNER_PRODUCT_DIFF

    expected_vals = [
        metric.compute_single(v1[0], v2[0], x),
        metric.compute_single(v1[1], v2[1], x),
    ]
    expected = jnp.mean(jnp.array(expected_vals))

    assert metric.compute(v1, v2, x=x, reduction="mean") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# FullMatrixMetric Tests
# ---------------------------------------------------------------------------


def test_relative_frobenius():
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = A + 1.0
    expected = jnp.linalg.norm(A - B, "fro") / jnp.linalg.norm(A, "fro")
    assert FullMatrixMetric.RELATIVE_FROBENIUS.compute(A, B) == pytest.approx(expected)


def test_spectral_norm():
    A = jnp.eye(2)
    B = 2 * jnp.eye(2)
    expected = jnp.linalg.norm(A - B, 2)
    assert FullMatrixMetric.SPECTRAL.compute(A, B) == pytest.approx(float(expected))


def test_cosine_similarity_matrix():
    A = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    B = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    assert FullMatrixMetric.COSINE_SIMILARITY.compute(A, B) == pytest.approx(1.0)


def test_trace_distance():
    A = jnp.array([[1.0, 0.0], [0.0, 2.0]])
    B = jnp.array([[2.0, 0.0], [0.0, 5.0]])
    expected = abs(3 - 7) / 2
    assert FullMatrixMetric.TRACE_DISTANCE.compute(A, B) == expected


# ---------------------------------------------------------------------------
# compare_matrices Tests
# ---------------------------------------------------------------------------


def test_compare_matrices_custom_metrics():
    A = jnp.eye(2)
    B = 3 * jnp.eye(2)
    metrics = [FullMatrixMetric.TRACE_DISTANCE, FullMatrixMetric.SPECTRAL]
    result = compare_matrices(A, B, metrics)

    assert set(result.keys()) == {"trace_distance", "spectral"}
    assert result["spectral"] == pytest.approx(2.0)
