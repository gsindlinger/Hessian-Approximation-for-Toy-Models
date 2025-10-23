from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from config.config import (
    Config,
    KFACConfig,
    LinearModelConfig,
    RandomClassificationConfig,
    TrainingConfig,
)
from data.data import AbstractDataset
from hessian_approximations.factory import (
    create_hessian_by_name,
    hessian_approximation,
)
from hessian_approximations.gauss_newton.gauss_newton import GaussNewton
from hessian_approximations.hessian.hessian import Hessian
from hessian_approximations.kfac.kfac import KFAC
from main import train_and_evaluate
from models.train import ApproximationModel, get_loss_fn

ModelTuple = Tuple[ApproximationModel, AbstractDataset, Any, Config]


class TestEKFAC:
    """Tests for various Hessian approximation methods including E-KFAC."""

    @pytest.fixture(params=["linear"])
    def random_classification_config(self, request):
        """Parametrized configuration for testing across different setups."""
        if request.param == "linear":
            # Simple linear binary classification
            return Config(
                dataset=RandomClassificationConfig(
                    n_samples=1000,
                    n_features=10,
                    n_informative=5,
                    n_classes=2,
                    random_state=42,
                    train_test_split=1,
                ),
                model=LinearModelConfig(loss="cross_entropy", hidden_dim=[10]),
                training=TrainingConfig(
                    epochs=5,
                    batch_size=100,
                    lr=0.01,
                    optimizer="sgd",
                    loss="cross_entropy",
                ),
            )

        elif request.param == "mlp":
            # Nonlinear MLP with one hidden layer
            return Config(
                dataset=RandomClassificationConfig(
                    n_samples=800,
                    n_features=20,
                    n_informative=10,
                    n_classes=2,
                    random_state=42,
                    train_test_split=1,
                ),
                model=LinearModelConfig(loss="cross_entropy", hidden_dim=[16]),
                training=TrainingConfig(
                    epochs=5,
                    batch_size=100,
                    lr=0.01,
                    optimizer="sgd",
                    loss="cross_entropy",
                ),
            )

        elif request.param == "multiclass":
            # Multiclass setup for testing more complex gradient structures
            return Config(
                dataset=RandomClassificationConfig(
                    n_samples=1000,
                    n_features=15,
                    n_informative=10,
                    n_classes=5,
                    random_state=42,
                    train_test_split=1,
                ),
                model=LinearModelConfig(loss="cross_entropy", hidden_dim=[]),
                training=TrainingConfig(
                    epochs=5,
                    batch_size=100,
                    lr=0.01,
                    optimizer="sgd",
                    loss="cross_entropy",
                ),
            )

    @pytest.fixture
    def trained_model(self, random_classification_config) -> ModelTuple:
        """Train a small model for testing."""
        model, dataset, params = train_and_evaluate(random_classification_config)
        return model, dataset, params, random_classification_config

    def test_ekfac_existance(
        self, trained_model: Tuple[ApproximationModel, AbstractDataset, Any, Config]
    ):
        """Test if Hessian approximation can be computed without errors."""
        model, dataset, params, config = trained_model

        x, y = dataset.get_train_data()
        hessian_method = create_hessian_by_name("kfac")

        ekfac_hessian = hessian_approximation(
            method=hessian_method,
            model=model,
            parameters=params,
            test_data=jnp.asarray(x),
            test_targets=jnp.asarray(y),
            loss=get_loss_fn(config.model.loss),
        )

        assert ekfac_hessian is not None
        assert jnp.isfinite(ekfac_hessian).all()

    def test_gradient_consistency(self, trained_model: ModelTuple):
        """
        Verify E-KFAC collector vs true gradients (sanity test).
        The following should hold for each linear layer l:

            ∇_{W_l} log p(y | x; θ) = s_l a_{l-1}^T

        where:
            - a_{l-1} = activations of the previous layer
            - s_l = preactivation gradients = ∇_{W_l a_{l-1}} log p(y | x; θ)

        This test checks that the EKFAC collector correctly captures (a_{l-1}, s_l) and that the reconstructed gradient (s_l a_{l-1}^T) matches the true gradient computed via autodiff.
        """
        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        def normal_loss_fn(params, x, y, model):
            """Standard loss function without EKFAC wrapper."""
            pred = model.apply(params, x)
            return loss_fn(pred, y, reduction="sum")  # Sum to match EKFAC behavior

        # Compute true gradients via autodiff
        ground_truth_grads = jax.grad(normal_loss_fn)(
            params, dataset.get_train_data()[0], dataset.get_train_data()[1], model
        )

        # Collect EKFAC statistics
        ekfac_approx = KFAC(
            config=KFACConfig(reload_data=True, use_eigenvalue_correction=False)
        )
        ekfac_approx.generate_ekfac_components(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )

        # Compare each layer’s gradients
        for key in ground_truth_grads["params"].keys():
            gt_grad = ground_truth_grads["params"][key]["kernel"]

            a = ekfac_approx.collector.captured_data[key][0]
            g = ekfac_approx.collector.captured_data[key][1]

            # ∇_{W_l} ≈ s_l a_{l−1}ᵀ
            ag = jnp.einsum("ni,no->io", a, g)

            assert jnp.allclose(gt_grad, ag, atol=1e-5), (
                f"Gradient mismatch for layer {key}"
            )

            # Verify that E[a ⊗ g] matches E[∇_{W_l}] (vectorized check)
            a_kron_g = jnp.zeros((a.shape[1] * g.shape[1],))
            for i in range(a.shape[0]):
                a_kron_g += jnp.kron(a[i], g[i])
            a_kron_g = a_kron_g / a.shape[0]

            gt_grad_flat = gt_grad.reshape(-1) / a.shape[0]
            assert jnp.allclose(gt_grad_flat, a_kron_g, atol=1e-5), (
                f"E[a kron g] does not match E[ground truth gradient] for layer {key}"
            )

    def test_kfac_hessian(self, trained_model: ModelTuple):
        """Test KFAC Hessian computation (without eigenvalue correction)."""

        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        hessian = Hessian().compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )
        hessian = np.array(hessian)  # for easier debugging

        gnh = GaussNewton().compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )
        gnh = np.array(gnh)  # for easier debugging

        kfac_config = KFACConfig(
            reload_data=True, use_eigenvalue_correction=False, batch_size=None
        )
        kfac_model = KFAC(config=kfac_config)
        kfac = kfac_model.compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )
        kfac = np.array(kfac)  # for easier debugging

        assert hessian.shape == gnh.shape == kfac.shape, (
            "Hessian, GNH, and KFAC Hessians should have the same shape."
        )

        # # check whether the sign for each cell entry is the same to a good extent
        # sign_hessian_kfac = np.sign(hessian) == np.sign(kfac)
        # sign_hessian_gnh = np.sign(hessian) == np.sign(gnh)
        # sign_gnh_kfac = np.sign(gnh) == np.sign(kfac)

        # assert np.mean(sign_hessian_kfac) > 0.8, (
        #     "Hessian and KFAC Hessian signs differ too much."
        # )
        # assert np.mean(sign_hessian_gnh) > 0.8, "Hessian and GNH signs differ too much."
        # assert np.mean(sign_gnh_kfac) > 0.8, (
        #     "GNH and KFAC Hessian signs differ too much."
        # )

        # get the absolute differences between the matrices and divide by norm to get relative error
        diff_hessian_gnh = np.abs(hessian - gnh) / np.linalg.norm(hessian)
        diff_hessian_kfac = np.abs(hessian - kfac) / np.linalg.norm(hessian)
        diff_gnh_kfac = np.abs(gnh - kfac) / np.linalg.norm(gnh)

        assert np.max(diff_hessian_gnh) < 0.2, (
            f"Max relative difference between Hessian and GNH is too large: {np.max(diff_hessian_gnh)}"
        )
        assert np.max(diff_hessian_kfac) < 0.2, (
            f"Max relative difference between Hessian and KFAC is too large: {np.max(diff_hessian_kfac)}"
        )
        assert np.max(diff_gnh_kfac) < 0.2, (
            f"Max relative difference between GNH and KFAC is too large: {np.max(diff_gnh_kfac)}"
        )

        print("Top-left 10x10 submatrices of Hessian, GNH, and KFAC:")
        print(hessian[:10, :10])
        print(gnh[:10, :10])
        print(kfac[:10, :10])
        print("---")

    def test_ekfac_hessian(self, trained_model: ModelTuple):
        """Test KFAC Hessian computation (without eigenvalue correction)."""

        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        hessian = Hessian().compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )
        hessian = np.array(hessian)  # for easier debugging

        gnh = GaussNewton().compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )
        gnh = np.array(gnh)  # for easier debugging

        ekfac_config = KFACConfig(reload_data=True, use_eigenvalue_correction=True)
        ekfac_model = KFAC(config=ekfac_config)
        ekfac = ekfac_model.compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )
        ekfac = np.array(ekfac)  # for easier debugging

        assert hessian.shape == gnh.shape == ekfac.shape, (
            "Hessian, GNH, and KFAC Hessians should have the same shape."
        )

        # get the absolute differences between the matrices and divide by norm to get relative error
        diff_hessian_gnh = np.abs(hessian - gnh) / np.linalg.norm(hessian)
        diff_hessian_ekfac = np.abs(hessian - ekfac) / np.linalg.norm(hessian)
        diff_gnh_ekfac = np.abs(gnh - ekfac) / np.linalg.norm(gnh)

        assert np.max(diff_hessian_gnh) < 0.2, (
            f"Max relative difference between Hessian and GNH is too large: {np.max(diff_hessian_gnh)}"
        )
        assert np.max(diff_hessian_ekfac) < 0.2, (
            f"Max relative difference between Hessian and KFAC is too large: {np.max(diff_hessian_ekfac)}"
        )
        assert np.max(diff_gnh_ekfac) < 0.2, (
            f"Max relative difference between GNH and KFAC is too large: {np.max(diff_gnh_ekfac)}"
        )

    def test_ekfac_batched_processing_is_close_to_full_data(
        self, trained_model: ModelTuple
    ):
        """Test that E-KFAC with batch processing yields the same result as without batching on a single run."""

        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        ekfac_full_data_config = KFACConfig(
            reload_data=True,
            use_eigenvalue_correction=True,
            batch_size=None,  # Full data
            use_pseudo_targets=False,
        )
        ekfac_full_data_model = KFAC(config=ekfac_full_data_config)
        ekfac_full_data = ekfac_full_data_model.compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )
        ekfac_full_data = np.array(ekfac_full_data)  # for easier debugging

        ekfac_batched_config = KFACConfig(
            reload_data=True,
            use_eigenvalue_correction=True,
            batch_size=100,  # Smaller batches
            use_pseudo_targets=False,
        )
        ekfac_batched_model = KFAC(config=ekfac_batched_config)
        ekfac_batched = ekfac_batched_model.compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )
        ekfac_batched = np.array(ekfac_batched)  # for easier debugging

        assert ekfac_full_data.shape == ekfac_batched.shape, (
            "E-KFAC Hessians from full data and batched processing should have the same shape."
        )

        # check whether covariances, eigenvectors, and eigenvalues match between the two methods
        for layer_name, (
            activations_cov,
            gradients_cov,
        ) in ekfac_full_data_model.covariances.items():
            batched_activations_cov = ekfac_batched_model.covariances.activations[
                layer_name
            ]
            batched_gradients_cov = ekfac_batched_model.covariances.gradients[
                layer_name
            ]

            eigenvalue_corrections = ekfac_full_data_model.eigenvalue_corrections[
                layer_name
            ]
            eigenvalue_corrections_batched = ekfac_batched_model.eigenvalue_corrections[
                layer_name
            ]

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

            # Don't check eigenvectors, since they can differ by sign and ordering to due instability of eigendecomposition

            # Check eigenvalue corrections
            assert np.allclose(
                eigenvalue_corrections,
                eigenvalue_corrections_batched,
                atol=1e-7,
            ), f"Eigenvalue corrections mismatch for layer {layer_name}"

    def test_pseudo_targets(self, trained_model: ModelTuple):
        """Test that E-KFAC with pseudo-targets yields different results than without pseudo-targets."""

        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        ekfac_no_pseudo_config = KFACConfig(
            reload_data=True,
            use_eigenvalue_correction=True,
            batch_size=None,
            use_pseudo_targets=False,
        )
        ekfac_no_pseudo_model = KFAC(config=ekfac_no_pseudo_config)

        ekfac_with_pseudo_config = KFACConfig(
            reload_data=True,
            use_eigenvalue_correction=True,
            batch_size=None,
            use_pseudo_targets=True,
            pseudo_target_noise_std=0.1,
        )
        ekfac_with_pseudo_model = KFAC(config=ekfac_with_pseudo_config)

        true_targets = jnp.asarray(dataset.get_train_data()[1])
        pseudo_targets = ekfac_with_pseudo_model.generate_pseudo_targets(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            loss_fn=loss_fn,
        )

        print("True targets:", true_targets[:10])
        print("Pseudo targets:", pseudo_targets[:10])

        # assert that more than 50% are the same (otherwise we could randomly guess)
        matching = jnp.sum(true_targets == pseudo_targets)
        assert matching > 0.5 * true_targets.shape[0], (
            "Pseudo-targets should match true targets on more than 50% of samples."
        )

        # TODO: Right now it seems the hessians differ quite a lot when using pseudo-targets.
        #  We should investigate whether this is expected behavior or if there's a bug.

        ekfac_no_pseudo = ekfac_no_pseudo_model.compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=true_targets,
            loss_fn=loss_fn,
        )
        ekfac_no_pseudo = np.array(ekfac_no_pseudo)

        ekfac_with_pseudo = ekfac_with_pseudo_model.compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=pseudo_targets,
            loss_fn=loss_fn,
        )
        ekfac_with_pseudo = np.array(ekfac_with_pseudo)

        print("Top-left 10x10 submatrices of E-KFAC with and without pseudo-targets:")
        print(ekfac_no_pseudo[:10, :10])
        print("---")
        print(ekfac_with_pseudo[:10, :10])

        assert ekfac_no_pseudo.shape == ekfac_with_pseudo.shape, (
            "E-KFAC Hessians with and without pseudo-targets should have the same shape."
        )
