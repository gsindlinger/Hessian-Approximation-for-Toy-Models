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
from hessian_approximations.gauss_newton.gauss_newton import GaussNewton
from hessian_approximations.hessian.hessian import Hessian
from hessian_approximations.hessian_approximations import (
    create_hessian_by_name,
    hessian_approximation,
)
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

        kfac_config = KFACConfig(reload_data=True, use_eigenvalue_correction=False)
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

        # check whether the sign for each cell entry is the same to a good extent
        sign_hessian_ekfac = np.sign(hessian) == np.sign(ekfac)
        sign_hessian_gnh = np.sign(hessian) == np.sign(gnh)
        sign_gnh_kfac = np.sign(gnh) == np.sign(ekfac)

        assert np.mean(sign_hessian_ekfac) > 0.5, (
            "Hessian and KFAC Hessian signs differ too much."
        )
        assert np.mean(sign_hessian_gnh) > 0.5, "Hessian and GNH signs differ too much."
        assert np.mean(sign_gnh_kfac) > 0.5, (
            "GNH and KFAC Hessian signs differ too much."
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

        print("Top-left 10x10 submatrices of Hessian, GNH, and EKFAC:")
        print(hessian[:10, :10])
        print("---")
        print(gnh[:10, :10])
        print("---")
        print(ekfac[:10, :10])
