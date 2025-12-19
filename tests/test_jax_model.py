import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.utils.data.data import Dataset, RandomRegressionDataset
from src.utils.loss import mse_loss
from src.utils.models.linear_model import LinearModel
from src.utils.optimizers import optimizer
from src.utils.train import train_model


class TestLinearRegression:
    """Tests for basic linear regression implementations."""

    @pytest.fixture(params=["simple_regression", "multi_feature_regression"])
    def dataset(self, request):
        seed = 0

        if request.param == "simple_regression":
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
    def model_and_params(self, dataset: Dataset):
        model = LinearModel(
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            hidden_dim=[],
            seed=0,
        )

        model, params, _ = train_model(
            model,
            dataset.get_dataloader(batch_size=1, seed=0, shuffle=True),
            loss_fn=mse_loss,
            optimizer=optimizer("adamw", lr=0.01),
            epochs=200,
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

        assert mse < 0.1, f"MSE between JAX and sklearn is too large: {mse}"
