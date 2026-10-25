"""Exact metric preservation and damping-invariant analysis reuse."""

import jax.numpy as jnp
import numpy as np
import pytest

import experiments.analyze_hessians as analysis
from src.config import HessianApproximationMethod as Method
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric as Metric, compare_matrices
from src.utils.metrics.vector_metrics import VectorMetric


@pytest.mark.parametrize("shape", [(3, 3), (5, 7)])
def test_matrix_cosine_is_original_trace_formula(shape):
    rng = np.random.default_rng(42)
    a, b = (jnp.array(rng.normal(size=shape)) for _ in range(2))
    expected = jnp.trace(a @ b.T) / (jnp.linalg.norm(a, ord="fro") * jnp.linalg.norm(b, ord="fro") + 1e-10)
    np.testing.assert_allclose(Metric.COSINE_SIMILARITY.compute(a, b), expected, rtol=1e-12, atol=1e-12)


def test_metric_bundle_matches_each_original_metric():
    # Non-symmetric matrices ensure we retain true singular-value norms,
    # not a symmetry assumption or an extremal-eigenvalue approximation.
    a = jnp.array([[1., 4., -2.], [0., 2., 1.], [3., 0., -1.]])
    b = jnp.array([[2., -1., 0.], [2., 3., 1.], [0., 1., 4.]])
    metrics = Metric.all_metrics()
    expected = {m.value: float(m.compute(a, b)) for m in metrics}
    actual = compare_matrices(a, b, metrics)
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-12)


def test_metric_bundle_deduplicates_spectral_norm(monkeypatch):
    a, b = jnp.eye(3), jnp.ones((3, 3))
    original = jnp.linalg.norm
    calls = []

    def counted(array, *args, **kwargs):
        if kwargs.get("ord") == 2:
            calls.append(array.shape)
        return original(array, *args, **kwargs)

    monkeypatch.setattr(jnp.linalg, "norm", counted)
    compare_matrices(a, b, [Metric.SPECTRAL, Metric.RELATIVE_SPECTRAL])
    assert len(calls) == 2  # difference once, reference once
    calls.clear()
    compare_matrices(a, b, [Metric.SPECTRAL, Metric.RELATIVE_SPECTRAL], reference_spectral_norm=1.0)
    assert len(calls) == 1
    calls.clear()
    assert compare_matrices(a, b, []) == {}
    assert not calls


class _Estimator:
    def __init__(self, matrix):
        self.matrix = matrix
        self.materializations = 0

    def estimate_hessian(self):
        self.materializations += 1
        return self.matrix

    def estimate_hvp(self, vectors):
        return vectors @ self.matrix.T


class _Context:
    def __init__(self):
        self.estimators = {
            Method.EXACT: _Estimator(jnp.array([[2., 1.], [1., -1.]])),
            Method.KFAC: _Estimator(jnp.array([[1., 0.], [0., 3.]])),
            Method.GNH: _Estimator(jnp.array([[3., 0.], [0., 2.]])),
        }

    def get(self, method):
        return self.estimators[method]


def test_analysis_shares_reference_norm_and_materializes_each_matrix_once(monkeypatch):
    ctx = _Context()
    original = jnp.linalg.norm
    calls = []

    def counted(array, *args, **kwargs):
        if kwargs.get("ord") == 2:
            calls.append(array.shape)
        return original(array, *args, **kwargs)

    monkeypatch.setattr(jnp.linalg, "norm", counted)
    result = analysis.compare_matrices(
        ctx, Method.EXACT, [Method.EXACT, Method.KFAC, Method.GNH],
        [Metric.SPECTRAL, Metric.RELATIVE_SPECTRAL],
    )
    assert len(calls) == 3  # one reference, two different matrix differences
    assert all(est.materializations == 1 for est in ctx.estimators.values())
    assert set(result[Metric.SPECTRAL.value]) == {Method.KFAC.value, Method.GNH.value}


def test_undamped_cache_reuses_six_thresholds_but_not_other_checkpoints(monkeypatch):
    ctx = _Context()
    matrices_fn, hvps_fn = analysis.compare_matrices, analysis.compare_hvps
    counts = {"matrix": 0, "hvp": 0}

    def matrices(*args, **kwargs):
        counts["matrix"] += 1
        return matrices_fn(*args, **kwargs)

    def hvps(*args, **kwargs):
        counts["hvp"] += 1
        return hvps_fn(*args, **kwargs)

    monkeypatch.setattr(analysis, "compare_matrices", matrices)
    monkeypatch.setattr(analysis, "compare_hvps", hvps)

    def new_cache():
        return analysis.UndampedComparisonCache(
            ctx, [Method.EXACT, Method.KFAC, Method.GNH],
            [Metric.FROBENIUS], [VectorMetric.RELATIVE_ERROR],
            jnp.array([[1., 2.], [2., 3.]]), jnp.ones((2, 2)),
        )

    cache = new_cache()
    outputs = []
    for threshold in [1., .1, .01, .001, 1e-6, 1e-9]:
        outputs.append((cache.matrices(Method.EXACT), cache.hvps(Method.EXACT)))
    assert counts == {"matrix": 1, "hvp": 1}
    assert all(output == outputs[0] for output in outputs)
    cache.matrices(Method.GNH)
    cache.hvps(Method.GNH)
    assert counts == {"matrix": 2, "hvp": 2}
    # A new model/epoch receives a fresh instance; no cross-checkpoint reuse.
    new_cache().matrices(Method.EXACT)
    assert counts["matrix"] == 3
    assert not hasattr(cache, "ihvps")
