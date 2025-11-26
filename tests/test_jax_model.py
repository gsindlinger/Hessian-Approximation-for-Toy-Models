import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from config.config import (
    Config,
    LinearModelConfig,
    ModelConfig,
    RandomRegressionConfig,
    TrainingConfig,
    UCIDatasetConfig,
)
from models.train import train_or_load


class TestLinearRegression:
    """Tests for basic linear regression implementations."""

    @pytest.fixture(params=["simple_regression", "energy_dataset"])
    def regression_config(self, request):
        """Config for different regression scenarios."""
        if request.param == "simple_regression":
            return Config(
                dataset=RandomRegressionConfig(
                    n_samples=1000,
                    n_features=1,
                    n_targets=1,
                    noise=10,
                    train_test_split=1,
                ),
                model=ModelConfig(name="linear", loss="mse"),
                training=TrainingConfig(
                    epochs=100,
                    lr=0.01,
                    optimizer="sgd",
                    batch_size=100,
                    loss="mse",
                ),
            )
        elif request.param == "energy_dataset":
            return Config(
                dataset=UCIDatasetConfig(
                    train_test_split=1,
                ),
                model=LinearModelConfig(loss="mse", hidden_dim=[]),
                training=TrainingConfig(
                    epochs=200,
                    lr=0.01,
                    batch_size=768,
                    optimizer="sgd",
                    loss="mse",
                ),
            )

    @pytest.fixture
    def model_trained(self, regression_config):
        """Trained model for regression."""
        model_data = train_or_load(regression_config)
        return (
            model_data.model,
            model_data.dataset,
            model_data.params,
            regression_config,
        )

    def test_jax_vs_sklearn_regression(self, model_trained):
        """
        Test that JAX linear model matches sklearn after training.

        Verifies that the custom JAX implementation produces predictions
        that are very close to sklearn's LinearRegression on both simple
        and real-world datasets.
        """
        model, dataset, params, config = model_trained
        x, y = dataset.get_train_data()

        # For simple regression, use interpolated test points
        # For energy dataset, use training data directly
        if (
            isinstance(config.dataset, RandomRegressionConfig)
            and config.dataset.n_features == 1
        ):
            x_test = jnp.linspace(jnp.min(x), jnp.max(x), 100).reshape(-1, 1)
        else:
            x_test = x

        # JAX predictions
        y_pred_jax = model.apply(params, x_test)

        # sklearn predictions
        sk_model = LinearRegression()
        sk_model.fit(np.array(x), np.array(y))
        y_pred_sklearn = sk_model.predict(np.array(x_test))

        # Compare predictions
        mse = mean_squared_error(np.array(y_pred_jax), np.array(y_pred_sklearn))

        # MSE should be very small
        assert mse < 0.1, f"MSE between JAX and sklearn is too large: {mse}"
