import jax.numpy as jnp
import pytest

from config.config import (
    Config,
    LinearModelConfig,
    RandomRegressionConfig,
    TrainingConfig,
    UCIDatasetConfig,
)
from hessian_approximations.factory import (
    create_hessian_by_name,
    hessian_approximation,
)
from main import train_and_evaluate
from models.train import get_loss_fn


class TestHessianApproximations:
    """Tests for various Hessian approximation methods."""

    @pytest.fixture
    def random_regression_config(self):
        """Config for random regression with multiple features and targets."""
        return Config(
            dataset=RandomRegressionConfig(
                n_samples=1000,
                n_features=100,
                n_targets=10,
                noise=20,
                random_state=42,
                train_test_split=1,
            ),
            model=LinearModelConfig(loss="mse", hidden_dim=[]),
            training=TrainingConfig(
                epochs=200,
                lr=0.001,
                batch_size=100,
                optimizer="sgd",
                loss="mse",
            ),
        )

    @pytest.fixture
    def energy_regression_config(self):
        """Config for energy dataset regression."""
        return Config(
            dataset=UCIDatasetConfig(
                name="energy",
                train_test_split=1,
            ),
            model=LinearModelConfig(name="linear", loss="mse", hidden_dim=[]),
            training=TrainingConfig(
                epochs=200,
                lr=0.01,
                batch_size=768,
                optimizer="sgd",
                loss="mse",
            ),
        )

    @pytest.fixture
    def random_model_trained(self, random_regression_config):
        """Trained model for random regression."""
        model, dataset, params = train_and_evaluate(random_regression_config)
        return model, dataset, params, random_regression_config

    @pytest.fixture
    def energy_model_trained(self, energy_regression_config):
        model, dataset, params = train_and_evaluate(energy_regression_config)
        return model, dataset, params, energy_regression_config

    def compute_hessians(self, model, params, dataset, config):
        """Helper to compute all Hessian approximations."""
        hessian_methods = {
            "exact-hessian-regression": create_hessian_by_name(
                "exact-hessian-regression"
            ),
            "hessian": create_hessian_by_name("hessian"),
            "gauss-newton": create_hessian_by_name("gauss-newton"),
        }

        results = {}
        x, y = dataset.get_train_data()

        for name, method in hessian_methods.items():
            hessian_matrix = hessian_approximation(
                method=method,
                model=model,
                parameters=params,
                test_data=jnp.asarray(x),
                test_targets=jnp.asarray(y),
                loss=get_loss_fn(config.model.loss),
            )
            results[name] = hessian_matrix

        return results

    def test_hessian_equivalence_energy(self, energy_model_trained):
        """
        Test that for linear regression, H = H_GNH and both match H_exact.

        For linear models with MSE loss, the Hessian computed via automatic
        differentiation should match both the Gauss-Newton approximation and
        the exact analytical formula.
        """
        model, dataset, params, config = energy_model_trained
        hessians = self.compute_hessians(model, params, dataset, config)

        # Compare all pairs
        comparisons = [
            ("exact-hessian-regression", "hessian"),
            ("exact-hessian-regression", "gauss-newton"),
            ("hessian", "gauss-newton"),
        ]

        for name1, name2 in comparisons:
            diff = jnp.linalg.norm(hessians[name1] - hessians[name2], "fro")
            assert diff < 1e-4, (
                f"Frobenius norm difference between {name1} and {name2} "
                f"is too large: {diff}"
            )

            max_diff = jnp.max(jnp.abs(hessians[name1] - hessians[name2]))
            assert max_diff < 1e-4, (
                f"Max absolute difference between {name1} and {name2} "
                f"is too large: {max_diff}"
            )

    def test_hessian_equivalence_random(self, random_model_trained):
        """
        Test Hessian equivalence on random high-dimensional data.

        Similar to energy dataset test but with higher dimensionality.
        """
        model, dataset, params, config = random_model_trained
        hessians = self.compute_hessians(model, params, dataset, config)

        comparisons = [
            ("exact-hessian-regression", "hessian"),
            ("exact-hessian-regression", "gauss-newton"),
            ("hessian", "gauss-newton"),
        ]

        for name1, name2 in comparisons:
            diff = jnp.linalg.norm(hessians[name1] - hessians[name2], "fro")
            assert diff < 1e-4, (
                f"Frobenius norm difference between {name1} and {name2} "
                f"is too large: {diff}"
            )
