from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import flatten_util

from config.config import (
    Config,
)
from config.dataset_config import RandomClassificationConfig
from config.hessian_approximation_config import (
    KFACBuildConfig,
    KFACRunConfig,
)
from config.model_config import LinearModelConfig
from config.training_config import TrainingConfig
from data.data import AbstractDataset
from hessian_approximations.gauss_newton.gauss_newton import GaussNewton
from hessian_approximations.hessian.hessian import Hessian
from hessian_approximations.kfac.kfac import KFAC
from models.train import ApproximationModel, get_loss_fn, train_or_load
from utils.utils import (
    sample_gradient_from_output_distribution,
    sample_gradient_from_output_distribution_batched,
)

ModelTuple = Tuple[ApproximationModel, AbstractDataset, Any, Config]


class TestEKFAC:
    """Tests for various Hessian approximation methods including E-KFAC."""

    @pytest.fixture(params=["linear", "multi_layer"], scope="session")
    def config(self, request):
        """Parametrized configuration for testing across different setups."""
        if request.param == "linear":
            config = Config(
                dataset=RandomClassificationConfig(
                    n_samples=10000,
                    n_features=10,
                    n_informative=5,
                    n_classes=2,
                    train_test_split=1,
                ),
                model=LinearModelConfig(loss="cross_entropy", hidden_dim=[]),
                training=TrainingConfig(
                    epochs=50,
                    batch_size=100,
                    lr=0.001,
                    optimizer="sgd",
                    loss="cross_entropy",
                    save_checkpoint=True,
                ),
                seed=123,
            )
        elif request.param == "multi_layer":
            config = Config(
                dataset=RandomClassificationConfig(
                    n_samples=10000,
                    n_features=10,
                    n_informative=5,
                    n_classes=2,
                    train_test_split=1,
                ),
                model=LinearModelConfig(loss="cross_entropy", hidden_dim=[5]),
                training=TrainingConfig(
                    epochs=30,
                    batch_size=100,
                    lr=0.001,
                    optimizer="sgd",
                    loss="cross_entropy",
                    save_checkpoint=True,
                ),
                seed=123,
            )

        yield config

    def _compute_full_implicit_matrices(self, model_obj: KFAC, dim: int):
        """Helper method to compute full H and H^-1 matrices
        implicitly via hvp and ihvp for each basis vector."""
        hvp_full = []
        ihvp_full = []
        for i in range(dim):
            unit_vec = jnp.zeros(dim).at[i].set(1.0)
            hvp = model_obj.compute_hvp(vectors=unit_vec, damping=model_obj.damping())
            ihvp = model_obj.compute_ihvp(vectors=unit_vec, damping=model_obj.damping())
            hvp_full.append(hvp)
            ihvp_full.append(ihvp)
        return jnp.column_stack(hvp_full), jnp.column_stack(ihvp_full)

    def test_ekfac_existence(self, config: Config):
        """Test if Hessian approximation can be computed without errors."""
        ekfac_hessian = KFAC(full_config=config).compute_hessian()

        assert ekfac_hessian is not None
        assert jnp.isfinite(ekfac_hessian).all()

    def test_gradient_consistency(self, config: Config):
        """
        Verify E-KFAC collector vs true gradients (sanity test).
        The following should hold for each linear layer l:

            ∇_{W_l} log p(y | x; θ) =  a_{l-1}^T s_l

        where:
            - a_{l-1} = activations of the previous layer
            - s_l = preactivation gradients = ∇_{W_l a_{l-1}} log p(y | x; θ)
        """
        model_data = train_or_load(config)

        def normal_loss_fn(params, x, y, model):
            """Standard loss function without EKFAC wrapper."""
            pred = model.apply(params, x)
            return model_data.loss(
                pred, y, reduction="sum"
            )  # Sum to match EKFAC behavior

        # Compute true gradients via autodiff
        ground_truth_grads = jax.grad(normal_loss_fn)(
            model_data.params,
            model_data.dataset.get_train_data()[0],
            model_data.dataset.get_train_data()[1],
            model_data.model,
        )

        # Collect EKFAC statistics
        # Need to reload data, otherwise collector data is empty
        ekfac_build_config = KFACBuildConfig(
            use_pseudo_targets=False,
            recalc_ekfac_components=True,
            collector_batch_size=model_data.dataset.get_train_data()[0].shape[0],
        )
        ekfac_approx = KFAC.setup_with_run_and_build_config(
            full_config=config, build_config=ekfac_build_config
        )
        ekfac_approx.get_ekfac_components()

        # Compare each layer’s gradients
        for key in ground_truth_grads["params"].keys():
            gt_grad = ground_truth_grads["params"][key]["kernel"]

            a = ekfac_approx.collector.captured_data[key][0]
            g = ekfac_approx.collector.captured_data[key][1]

            # ∇_{W_l} ≈ a^T g
            ag = jnp.einsum("ni,no->io", a, g)

            assert jnp.allclose(gt_grad, ag, atol=1e-4), (
                f"Gradient mismatch for layer {key}"
            )

            # Verify that E[a ⊗ g] matches E[∇_{W_l}] (vectorized check)
            a_kron_g = jnp.zeros((a.shape[1] * g.shape[1],))
            for i in range(a.shape[0]):
                a_kron_g += jnp.kron(a[i], g[i])
            a_kron_g = a_kron_g / a.shape[0]

            gt_grad_flat = gt_grad.reshape(-1) / a.shape[0]
            assert jnp.allclose(gt_grad_flat, a_kron_g, atol=1e-4), (
                f"E[a kron g] does not match E[ground truth gradient] for layer {key}"
            )

    def test_kfac_hessian(self, config: Config):
        """Test KFAC Hessian computation (without eigenvalue correction)."""

        hessian = Hessian(full_config=config).compute_hessian()

        gnh = GaussNewton(full_config=config).compute_hessian()

        kfac_config = KFACRunConfig(
            use_eigenvalue_correction=False,
            damping_lambda=0.0,
        )
        kfac_model = KFAC.setup_with_run_and_build_config(
            full_config=config, run_config=kfac_config
        )
        kfac = kfac_model.compute_hessian()

        assert hessian.shape == gnh.shape == kfac.shape, (
            "Hessian, GNH, and KFAC Hessians should have the same shape."
        )

    def test_ekfac_hessian(self, config: Config):
        """Test KFAC Hessian computation (without eigenvalue correction)."""

        hessian = Hessian(full_config=config).compute_hessian()

        gnh = GaussNewton(full_config=config).compute_hessian()

        ekfac_config = KFACRunConfig(
            use_eigenvalue_correction=True,
        )
        ekfac_model = KFAC.setup_with_run_and_build_config(
            full_config=config, run_config=ekfac_config
        )
        ekfac = ekfac_model.compute_hessian()

        assert hessian.shape == gnh.shape == ekfac.shape, (
            "Hessian, GNH, and KFAC Hessians should have the same shape."
        )

    def test_ekfac_batched_processing_is_close_to_full_data(self, config: Config):
        """Test that E-KFAC with batch processing yields the same result as without batching on a single run."""

        ekfac_run_config = KFACRunConfig(use_eigenvalue_correction=True)
        ekfac_full_data_config = KFACBuildConfig(
            recalc_ekfac_components=True,
            collector_batch_size=None,  # Full data
            use_pseudo_targets=False,
        )

        ekfac_full_data_model = KFAC.setup_with_run_and_build_config(
            full_config=config,
            run_config=ekfac_run_config,
            build_config=ekfac_full_data_config,
        )
        ekfac_full_data = ekfac_full_data_model.compute_hessian()

        ekfac_batched_config = KFACBuildConfig(
            collector_batch_size=100,  # Smaller batches
            use_pseudo_targets=False,
        )

        ekfac_batched_model = KFAC.setup_with_run_and_build_config(
            full_config=config,
            run_config=ekfac_run_config,
            build_config=ekfac_batched_config,
        )
        ekfac_batched = ekfac_batched_model.compute_hessian()

        assert ekfac_full_data.shape == ekfac_batched.shape, (
            "E-KFAC Hessians from full data and batched processing should have the same shape."
        )

        # check whether covariances, eigenvectors, and eigenvalues match between the two methods
        for layer_name, (
            activations_cov,
            gradients_cov,
        ) in ekfac_full_data_model.kfac_data.covariances.items():
            batched_activations_cov = (
                ekfac_batched_model.kfac_data.covariances.activations[layer_name]
            )
            batched_gradients_cov = ekfac_batched_model.kfac_data.covariances.gradients[
                layer_name
            ]

            eigenvalue_corrections = (
                ekfac_full_data_model.kfac_data.eigenvalue_corrections[layer_name]
            )
            eigenvalue_corrections_batched = (
                ekfac_batched_model.kfac_data.eigenvalue_corrections[layer_name]
            )

            assert np.allclose(
                activations_cov,
                batched_activations_cov,
                atol=1e-7,
            ), f"Activation covariances mismatch for layer {layer_name}"
            assert np.allclose(
                gradients_cov,
                batched_gradients_cov,
                atol=1e-7,
            ), f"Gradient covariances mismatch for layer {layer_name}"

            # Check eigenvalue corrections
            assert np.allclose(
                eigenvalue_corrections,
                eigenvalue_corrections_batched,
                atol=1e-7,
            ), f"Eigenvalue corrections mismatch for layer {layer_name}"

    def test_ekfac_hessian_ihvp_and_hvp_consistency(self, config: Config):
        """Test E-KFAC: Check whether H @ (H^{-1} @ v) ≈ v and H^{-1} @ (H @ v) ≈ v."""
        model_data = train_or_load(config)
        x_train, y_train = model_data.dataset.get_train_data()
        x_train = jnp.asarray(x_train)
        loss_fn = get_loss_fn(config.model.loss)

        ekfac_model = KFAC.setup_with_run_and_build_config(
            full_config=config,
            run_config=KFACRunConfig(
                use_eigenvalue_correction=True, damping_lambda=0.1
            ),
        )

        test_vec = sample_gradient_from_output_distribution(
            model=model_data.model,
            params=model_data.params,
            x_train=x_train,
            loss_fn=loss_fn,
        )

        H = ekfac_model.compute_hessian(damping=ekfac_model.damping())
        H_inv = ekfac_model.compute_inverse_hessian(damping=ekfac_model.damping())
        ihvp = ekfac_model.compute_ihvp(test_vec, damping=ekfac_model.damping())
        hvp = ekfac_model.compute_hvp(test_vec, damping=ekfac_model.damping())

        # Round-trip tests
        ihvp_round_trip = H @ ihvp
        assert jnp.allclose(ihvp_round_trip, test_vec, rtol=0.01, atol=1e-5), (
            "IHVP round-trip failed."
        )
        hvp_round_trip = H_inv @ hvp
        assert jnp.allclose(hvp_round_trip, test_vec, rtol=0.01, atol=1e-5), (
            "HVP round-trip failed."
        )

    def test_kfac_hessian_ihvp_and_hvp_consistency(self, config: Config):
        """Check whether H @ (H^{-1} @ v) ≈ v and H^{-1} @ (H @ v) ≈ v."""
        model_data = train_or_load(config)
        x_train, y_train = model_data.dataset.get_train_data()
        x_train = jnp.asarray(x_train)
        loss_fn = get_loss_fn(config.model.loss)

        kfac_model = KFAC.setup_with_run_and_build_config(
            full_config=config,
            run_config=KFACRunConfig(
                use_eigenvalue_correction=False, damping_lambda=0.1
            ),
        )

        test_vec = sample_gradient_from_output_distribution(
            model=model_data.model,
            params=model_data.params,
            x_train=x_train,
            loss_fn=loss_fn,
        )

        H = kfac_model.compute_hessian(damping=kfac_model.damping())
        H_inv = kfac_model.compute_inverse_hessian(damping=kfac_model.damping())
        ihvp = kfac_model.compute_ihvp(test_vec, damping=kfac_model.damping())
        hvp = kfac_model.compute_hvp(test_vec, damping=kfac_model.damping())

        # Round-trip tests
        ihvp_round_trip = H @ ihvp
        assert jnp.allclose(ihvp_round_trip, test_vec, rtol=0.01, atol=1e-5), (
            "IHVP round-trip failed."
        )
        hvp_round_trip = H_inv @ hvp
        assert jnp.allclose(hvp_round_trip, test_vec, rtol=0.01, atol=1e-5), (
            "HVP round-trip failed."
        )

    def test_ekfac_explicit_vs_implicit_equivalence(self, config: Config):
        """EKFAC explicit vs implicit Hessian and inverse equivalence check."""
        model_data = train_or_load(config)

        ekfac_model = KFAC.setup_with_run_and_build_config(
            full_config=config,
            run_config=KFACRunConfig(
                use_eigenvalue_correction=True, damping_lambda=0.1
            ),
        )

        dim = flatten_util.ravel_pytree(model_data.params)[0].shape[0]
        H_explicit = ekfac_model.compute_hessian(damping=ekfac_model.damping())
        H_inv_explicit = ekfac_model.compute_inverse_hessian(
            damping=ekfac_model.damping()
        )
        H_implicit, H_inv_implicit = self._compute_full_implicit_matrices(
            ekfac_model, dim
        )

        assert jnp.allclose(H_explicit, H_implicit, atol=1e-3), (
            "E-KFAC Hessian explicit vs implicit mismatch"
        )
        assert jnp.allclose(H_inv_explicit, H_inv_implicit, atol=1e-3), (
            "E-KFAC Inverse Hessian explicit vs implicit mismatch"
        )

    def test_kfac_explicit_vs_implicit_equivalence(self, config: Config):
        """KFAC explicit vs implicit Hessian and inverse equivalence check."""
        model_data = train_or_load(config)
        dim = flatten_util.ravel_pytree(model_data.params)[0].shape[0]

        kfac_model = KFAC.setup_with_run_and_build_config(
            full_config=config,
            run_config=KFACRunConfig(
                use_eigenvalue_correction=False, damping_lambda=0.1
            ),
        )

        H_explicit = kfac_model.compute_hessian(damping=kfac_model.damping())
        H_inv_explicit = kfac_model.compute_inverse_hessian(
            damping=kfac_model.damping()
        )
        H_implicit, H_inv_implicit = self._compute_full_implicit_matrices(
            kfac_model, dim
        )

        assert jnp.allclose(H_explicit, H_implicit, atol=1e-3), (
            "KFAC Hessian explicit vs implicit mismatch"
        )
        assert jnp.allclose(H_inv_explicit, H_inv_implicit, atol=1e-3), (
            "KFAC Inverse Hessian explicit vs implicit mismatch"
        )

    def test_ekfac_ihvp_batched_shape_and_finiteness(self, config: Config):
        """Verify batched IHVP shape and finiteness."""

        ekfac_model = KFAC.setup_with_run_and_build_config(
            full_config=config,
            run_config=KFACRunConfig(
                use_eigenvalue_correction=True, damping_lambda=0.1
            ),
        )
        test_vectors = sample_gradient_from_output_distribution_batched(
            ekfac_model.model_context
        )
        ihvp_batched = ekfac_model.compute_ihvp(
            vectors=test_vectors, damping=ekfac_model.damping()
        )

        assert ihvp_batched.shape == test_vectors.shape, (
            f"Batched IHVP shape {ihvp_batched.shape} doesn't match input {test_vectors.shape}"
        )
        assert jnp.isfinite(ihvp_batched).all(), (
            "Batched IHVP contains non-finite values"
        )

    def test_ekfac_ihvp_batched_vs_single_consistency(self, config: Config):
        """Verify batched IHVP matches single-vector IHVP (EKFAC + KFAC)."""
        ekfac_model = KFAC.setup_with_run_and_build_config(
            full_config=config,
            run_config=KFACRunConfig(
                use_eigenvalue_correction=True, damping_lambda=0.1
            ),
        )
        kfac_model = KFAC.setup_with_run_and_build_config(
            full_config=config,
            run_config=KFACRunConfig(
                use_eigenvalue_correction=False, damping_lambda=0.1
            ),
        )

        test_vectors = sample_gradient_from_output_distribution_batched(
            ekfac_model.model_context
        )

        ihvp_batched_ekfac = ekfac_model.compute_ihvp(
            vectors=test_vectors, damping=ekfac_model.damping()
        )
        ihvp_batched_kfac = kfac_model.compute_ihvp(
            vectors=test_vectors, damping=kfac_model.damping()
        )

        for i in range(test_vectors.shape[0]):
            ihvp_single_ekfac = ekfac_model.compute_ihvp(
                vectors=test_vectors[i], damping=ekfac_model.damping()
            )
            ihvp_single_kfac = kfac_model.compute_ihvp(
                vectors=test_vectors[i], damping=kfac_model.damping()
            )

            assert jnp.allclose(
                ihvp_batched_ekfac[i], ihvp_single_ekfac, rtol=1e-6, atol=1e-4
            ), f"Batched IHVP != single-vector IHVP for vector {i} (EKFAC)"
            assert jnp.allclose(
                ihvp_batched_kfac[i], ihvp_single_kfac, rtol=1e-6, atol=1e-4
            ), f"Batched IHVP != single-vector IHVP for vector {i} (KFAC)"

    def test_ekfac_ihvp_hessian_roundtrip_batched(self, config: Config):
        """Verify H @ (H^{-1} @ V) ≈ V for batched IHVPs."""

        ekfac_model = KFAC.setup_with_run_and_build_config(
            full_config=config,
            run_config=KFACRunConfig(
                use_eigenvalue_correction=True, damping_lambda=0.1
            ),
        )

        test_vectors = sample_gradient_from_output_distribution_batched(
            ekfac_model.model_context
        )

        H = ekfac_model.compute_hessian(damping=ekfac_model.damping())
        ihvp_batched = ekfac_model.compute_ihvp(
            vectors=test_vectors, damping=ekfac_model.damping()
        )
        hvp_roundtrip = H @ ihvp_batched.T  # Apply H to each vector

        assert jnp.allclose(hvp_roundtrip.T, test_vectors, rtol=1e-2, atol=1e-5), (
            "Round-trip H @ (H^{-1} @ V) ≈ V failed for batched IHVP."
        )
