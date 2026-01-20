from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.config import (
    LossType,
    ModelArchitecture,
    ModelConfig,
    OptimizerType,
    TrainingConfig,
)
from src.utils.data.data import Dataset, RandomRegressionDataset
from src.utils.loss import get_loss
from src.utils.models.approximation_model import ApproximationModel
from src.utils.optimizers import optimizer
from src.utils.train import train_model


class TestLinearRegression:
    """Tests for basic linear regression implementations."""

    @pytest.fixture(params=["simple_regression", "multi_feature_regression"])
    def config(self, request, tmp_path_factory):
        """Create model configuration for testing."""
        base = tmp_path_factory.mktemp(request.param)

        if request.param == "simple_regression":
            n_features = 1
        else:
            n_features = 10

        model_config = ModelConfig(
            architecture=ModelArchitecture.LINEAR,
            input_dim=n_features,
            hidden_dim=None,
            output_dim=1,
            loss=LossType.MSE,
            training=TrainingConfig(
                learning_rate=0.01,
                optimizer=OptimizerType.ADAMW,
                epochs=200,
                batch_size=1,
            ),
            directory=str(base / "model"),
        )

        return {
            "model_config": model_config,
            "n_features": n_features,
        }

    @pytest.fixture
    def dataset(self, config):
        """Create a random regression dataset for testing."""
        seed = 0
        n_features = config["n_features"]

        if n_features == 1:
            return RandomRegressionDataset(
                n_samples=100,
                n_features=1,
                n_targets=1,
                noise=10.0,
                seed=seed,
            )
        else:
            return RandomRegressionDataset(
                n_samples=500,
                n_features=10,
                n_targets=1,
                noise=10.0,
                seed=seed,
            )

    @pytest.fixture
    def model_and_params(
        self, config, dataset: Dataset
    ) -> Tuple[ApproximationModel, dict, Dataset]:
        """Train a model and return it with its parameters and dataset."""
        model_config = config["model_config"]

        # Verify dimensions match
        assert model_config.input_dim == dataset.input_dim()
        assert model_config.output_dim == dataset.output_dim()

        # Train the model
        model, params, _ = train_model(
            model_config=model_config,
            dataloader=dataset.get_dataloader(
                batch_size=model_config.training.batch_size, seed=0, shuffle=True
            ),
            loss_fn=get_loss(model_config.loss),
            optimizer=optimizer(
                model_config.training.optimizer, lr=model_config.training.learning_rate
            ),
            epochs=model_config.training.epochs,
        )

        return model, params, dataset

    def test_jax_vs_sklearn_regression(self, model_and_params):
        """
        Test that JAX linear model matches sklearn after training.

        Verifies that the custom JAX implementation produces predictions
        that are very close to sklearn's LinearRegression.
        """
        model, params, dataset = model_and_params
        x, y = dataset.inputs, dataset.targets

        # For 1D regression, evaluate on a dense grid
        if dataset.input_dim() == 1:
            x_test = jnp.linspace(jnp.min(x), jnp.max(x), 200).reshape(-1, 1)
        else:
            x_test = x

        # JAX predictions
        y_pred_jax = model.apply(params, x_test)

        # sklearn predictions
        sk_model = LinearRegression(fit_intercept=False)
        sk_model.fit(np.asarray(x), np.asarray(y))
        y_pred_sklearn = sk_model.predict(np.asarray(x_test))

        # Compare predictions
        mse = mean_squared_error(
            np.asarray(y_pred_jax),
            np.asarray(y_pred_sklearn),
        )

        assert mse < 0.1, f"MSE between JAX and sklearn is too large: {mse:. 6f}"
