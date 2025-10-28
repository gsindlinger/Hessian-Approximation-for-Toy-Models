import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util

from config.config import (
    Config,
    LinearModelConfig,
    LiSSAConfig,
    RandomClassificationConfig,
    RandomRegressionConfig,
    TrainingConfig,
)
from hessian_approximations.factory import create_hessian_by_name
from hessian_approximations.lissa.lissa import LiSSA
from main import train
from models.train import get_loss_fn


class TestLiSSA:
    """Tests for the LiSSA inverse Hessian–vector product approximation."""

    # ------------------------------
    # Fixtures
    # ------------------------------
    @pytest.fixture
    def classification_config(self):
        """Simple linear classification model with cross-entropy loss."""
        return Config(
            dataset=RandomClassificationConfig(
                n_samples=10000,
                n_features=5,
                n_informative=3,
                n_classes=2,
                random_state=42,
                train_test_split=1.0,
            ),
            model=LinearModelConfig(loss="cross_entropy", hidden_dim=[5]),
            training=TrainingConfig(
                epochs=100,
                batch_size=100,
                lr=0.001,
                optimizer="sgd",
                loss="cross_entropy",
            ),
        )

    # ------------------------------
    @pytest.fixture
    def random_regression_config(self):
        """Config for random regression with multiple features and targets."""
        return Config(
            dataset=RandomRegressionConfig(
                n_samples=1000,
                n_features=100,
                n_targets=10,
                noise=0.1,
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
    def trained_model(self, classification_config):
        """Train a simple classification model for testing LiSSA."""
        model, dataset, params = train(classification_config)
        return model, dataset, params, classification_config

    @pytest.fixture
    def regression_trained_model(self, random_regression_config):
        """Train a regression model for testing LiSSA."""
        model, dataset, params = train(random_regression_config)
        return model, dataset, params, random_regression_config

    # ------------------------------
    # Tests
    # ------------------------------

    def test_lissa_matches_exact_inverse(self, regression_trained_model):
        """
        Test that LiSSA's IHVP approximation is close to the IHVP
        computed by the exact Hessian method.
        """
        model, dataset, params, config = regression_trained_model
        x_train, y_train = dataset.get_train_data()
        loss_fn = get_loss_fn(config.model.loss)

        # Create LiSSA and exact Hessian methods
        lissa_method = LiSSA(
            config=LiSSAConfig(
                num_samples=3,
                recursion_depth=500,
                alpha=0.005,
                damping=0.001,
                batch_size=128,
                seed=42,
                convergence_tol=1e-3,
            )
        )
        hessian_method = create_hessian_by_name("hessian")

        # Flatten parameters and sample random vector
        params_flat, _ = flatten_util.ravel_pytree(params)
        rng = jax.random.PRNGKey(0)
        v = jax.random.normal(rng, params_flat.shape, dtype=params_flat.dtype)

        # Compute IHVPs
        ihvp_lissa = lissa_method.compute_ihvp(
            model=model,
            params=params,
            training_data=jnp.asarray(x_train),
            training_targets=jnp.asarray(y_train),
            loss_fn=loss_fn,
            vector=v,
        )

        ihvp_exact = hessian_method.compute_ihvp(
            model=model,
            params=params,
            training_data=jnp.asarray(x_train),
            training_targets=jnp.asarray(y_train),
            loss_fn=loss_fn,
            vector=v,
        )

        # Compare numerically
        rel_diff = jnp.linalg.norm(ihvp_lissa - ihvp_exact) / jnp.linalg.norm(
            ihvp_exact
        )
        max_diff = jnp.max(jnp.abs(ihvp_lissa - ihvp_exact))

        assert rel_diff < 0.05, f"Relative norm difference too large: {rel_diff}"
        assert max_diff < 0.1, f"Max absolute difference too large: {max_diff}"

    def test_lissa_is_stochastic_but_consistent(self, trained_model):
        """
        Test that different random seeds yield similar IHVPs (stochastic consistency).
        """
        model, dataset, params, config = trained_model
        x_train, y_train = dataset.get_train_data()
        loss_fn = get_loss_fn(config.model.loss)

        params_flat, _ = flatten_util.ravel_pytree(params)
        rng = jax.random.PRNGKey(0)
        v = jax.random.normal(rng, params_flat.shape, dtype=params_flat.dtype)

        results = []
        for seed in [0, 1, 2]:
            lissa = LiSSA(
                config=LiSSAConfig(
                    num_samples=1,
                    recursion_depth=300,
                    alpha=0.005,
                    damping=0.001,
                    batch_size=128,
                    seed=seed,
                )
            )
            ihvp = lissa.compute_ihvp(
                model=model,
                params=params,
                training_data=jnp.asarray(x_train),
                training_targets=jnp.asarray(y_train),
                loss_fn=loss_fn,
                vector=v,
            )
            results.append(ihvp)

        # Average pairwise difference
        diffs = [
            jnp.linalg.norm(results[i] - results[j])
            for i in range(3)
            for j in range(i + 1, 3)
        ]
        avg_diff = jnp.mean(jnp.array(diffs))

        assert avg_diff < 1.0, (
            f"LiSSA estimates vary too much between seeds: avg diff {avg_diff}"
        )
