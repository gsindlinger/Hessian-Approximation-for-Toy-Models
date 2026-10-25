"""No-training checks for exact shared-token curvature and approximations."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.hessians.computer.computer import KroneckerEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.emac import EMacComputer
from src.hessians.computer.eidentity import EIdentityComputer
from src.hessians.computer.eshampoo import EShampooComputer
from src.hessians.computer.fim import FIMComputer
from src.hessians.computer.fim_block import FIMBlockComputer
from src.hessians.utils.data import DataActivationsGradients, parameter_gradients_from_factors


@pytest.fixture
def shared_data():
    rng = np.random.default_rng(123)
    return DataActivationsGradients(
        activations={
            "dense": jnp.asarray(rng.normal(size=(3, 2))),
            "shared": jnp.asarray(rng.normal(size=(3, 4, 2))),
            "vector": jnp.ones((3, 4, 1)),
        },
        gradients={
            "dense": jnp.asarray(rng.normal(size=(3, 3, 2))),
            "shared": jnp.asarray(rng.normal(size=(3, 4, 3, 2))),
            "vector": jnp.asarray(rng.normal(size=(3, 4, 3, 2))),
        },
        probs=jnp.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]),
        layer_names=["dense", "shared", "vector"],
    )


def _reference_gradients(a, g):
    """Independent explicit loops, so the reference cannot share einsum bugs."""
    a, g = np.asarray(a), np.asarray(g)
    if a.ndim == 2:
        a, g = a[:, None], g[:, None]
    out = np.zeros((a.shape[0], a.shape[-1], g.shape[-2], g.shape[-1]))
    for n in range(a.shape[0]):
        for k in range(g.shape[-1]):
            for t in range(a.shape[1]):
                out[n, :, :, k] += np.outer(a[n, t], g[n, t, :, k])
    return out


def test_fim_preserves_token_and_cross_parameter_terms(shared_data):
    d = shared_data
    all_grads = []
    for name in d.layer_names:
        ref = _reference_gradients(d.activations[name], d.gradients[name])
        np.testing.assert_allclose(
            parameter_gradients_from_factors(d.activations[name], d.gradients[name]), ref,
            atol=1e-12,
        )
        all_grads.append(ref.reshape(3, -1, 2))
    grads = np.concatenate(all_grads, axis=1)
    expected = np.einsum("npk,nqk,nk->pq", grads, grads, d.probs) / np.sum(d.probs)
    actual = FIMComputer(d, batch_size=2).build().estimate_hessian()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    block = FIMBlockComputer(d, batch_size=2).build().estimate_hessian()
    expected_block = np.zeros_like(expected)
    offset = 0
    for gradient in all_grads:
        end = offset + gradient.shape[1]
        expected_block[offset:end, offset:end] = expected[offset:end, offset:end]
        offset = end
    np.testing.assert_allclose(block, expected_block, rtol=1e-12, atol=1e-12)


def test_shared_ekfac_correction_sums_before_squaring(shared_data):
    d = shared_data
    a, g = d.activations["shared"], d.gradients["shared"]
    qa = np.linalg.qr(np.array([[0.3, 0.8], [-1.0, 0.2]]))[0]
    qg = np.linalg.qr(np.array([[0.3, 0.8, -0.1], [-1, 0.2, 0.5], [0.4, -0.7, 0.6]]))[0]
    grads = _reference_gradients(a, g)
    expected = np.zeros((2, 3))
    for n in range(3):
        for k in range(2):
            projected = qa.T @ grads[n, :, :, k] @ qg
            expected += float(d.probs[n, k]) * projected ** 2
    actual = KroneckerEstimator._compute_layer_lambda_chunk_sum(a, g, d.probs, qa, qg)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_shared_kfac_expand_and_mac_normalization(shared_data):
    d = shared_data
    a, g = np.asarray(d.activations["shared"]), np.asarray(d.gradients["shared"])
    expected_a = np.einsum("nti,ntj->ij", a, a) / 3
    expected_g = np.einsum("ntok,ntpk,nk->op", g, g, d.probs) / (4 * np.sum(d.probs))
    kfac = EKFACComputer._compute_covariances(d.activations, d.gradients, d.probs, batch_size=2)
    np.testing.assert_allclose(kfac["activation_cov"]["shared"], expected_a, atol=1e-12)
    np.testing.assert_allclose(kfac["gradient_cov"]["shared"], expected_g, atol=1e-12)
    mac = EMacComputer._compute_covariances(d.activations, d.gradients, d.probs, batch_size=2)
    np.testing.assert_allclose(mac["activation_cov"]["shared"], expected_a, atol=1e-12)
    np.testing.assert_array_equal(mac["gradient_cov"]["shared"], np.eye(3))


def test_shared_shampoo_uses_complete_parameter_gradient(shared_data):
    d = shared_data
    grads = _reference_gradients(d.activations["shared"], d.gradients["shared"])
    left, right = np.zeros((2, 2)), np.zeros((3, 3))
    for n in range(3):
        for k in range(2):
            w = grads[n, :, :, k]
            left += float(d.probs[n, k]) * w @ w.T
            right += float(d.probs[n, k]) * w.T @ w
    actual = EShampooComputer._compute_covariances(d.activations, d.gradients, d.probs, batch_size=2)
    np.testing.assert_allclose(actual["activation_cov"]["shared"], left / np.trace(left), atol=1e-12)
    np.testing.assert_allclose(actual["gradient_cov"]["shared"], right / np.sum(d.probs), atol=1e-12)


@pytest.mark.parametrize("computer", [EKFACComputer, EMacComputer, EShampooComputer, EIdentityComputer])
@pytest.mark.parametrize("corrected", [False, True])
def test_shared_approximators_cover_all_parameters(shared_data, computer, corrected):
    estimate = computer(
        shared_data, corr_context=shared_data,
        apply_eigenvalue_correction=corrected, batch_size=2,
    ).build().estimate_hessian()
    assert estimate.shape == (15, 15)
    assert np.isfinite(estimate).all()
    np.testing.assert_allclose(estimate, estimate.T, atol=1e-12)


def test_zero_gradient_shampoo_stays_finite():
    data = DataActivationsGradients(
        activations={"unused": jnp.ones((2, 4, 1))},
        gradients={"unused": jnp.zeros((2, 4, 3, 1))},
        probs=jnp.ones((2, 1)), layer_names=["unused"],
    )
    estimate = EShampooComputer(data, corr_context=data).build().estimate_hessian()
    np.testing.assert_array_equal(estimate, np.zeros((3, 3)))
