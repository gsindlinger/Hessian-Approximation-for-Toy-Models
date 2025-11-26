import copy

import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util

from config.config import Config
from config.dataset_config import RandomClassificationConfig, RandomRegressionConfig
from config.hessian_approximation_config import FisherInformationConfig
from config.model_config import LinearModelConfig
from config.training_config import TrainingConfig
from hessian_approximations.fim.fisher_information import FisherInformation
from hessian_approximations.gauss_newton.gauss_newton import GaussNewton
from metrics.full_matrix_metrics import FullMatrixMetric
from metrics.vector_metrics import VectorMetric


class TestFIMGaussNewton:
    @pytest.fixture
    def random_regression_config(self):
        """Config for random regression with multiple features and targets."""
        return Config(
            dataset=RandomRegressionConfig(
                n_samples=3000,
                n_features=10,
                n_targets=2,
                noise=10,
                train_test_split=1,
            ),
            model=LinearModelConfig(loss="mse", hidden_dim=[5]),
            training=TrainingConfig(
                epochs=200,
                lr=0.001,
                batch_size=100,
                optimizer="sgd",
                loss="mse",
            ),
        )

    @pytest.fixture
    def random_classification_config(self) -> Config:
        """Config for random classification with multiple features and classes."""
        return Config(
            dataset=RandomClassificationConfig(
                n_samples=3000,
                n_features=20,
                n_classes=5,
                n_informative=10,
                train_test_split=1,
            ),
            model=LinearModelConfig(loss="cross_entropy", hidden_dim=[10]),
            training=TrainingConfig(
                epochs=200,
                lr=0.001,
                batch_size=100,
                optimizer="sgd",
                loss="cross_entropy",
            ),
        )

    def test_fim_gnh_comparison_classification(
        self,
        random_classification_config: Config,
    ):
        """Test comparing FIM and Gauss-Newton Hessians for classification."""
        config = random_classification_config

        fim_config = FisherInformationConfig(fisher_type="true")
        random_classification_config.hessian_approximation = fim_config
        fim = FisherInformation(full_config=config)

        config_copy = copy.deepcopy(config)
        gnh = GaussNewton(full_config=config_copy)

        fim_hessian = fim.compute_hessian(damping=0.0)
        gnh_hessian = gnh.compute_hessian(damping=0.0)

        assert (
            FullMatrixMetric.COSINE_SIMILARITY.compute(fim_hessian, gnh_hessian) > 0.95
        ), (
            f"Cosine similarity too low: {FullMatrixMetric.COSINE_SIMILARITY.compute(fim_hessian, gnh_hessian)}"
        )

    def test_fim_gnh_ihvp_comparison_classification(
        self,
        random_classification_config: Config,
    ):
        """Test comparing FIM and Gauss-Newton Hessians for classification."""
        config = random_classification_config

        fim_config = FisherInformationConfig(fisher_type="true")
        random_classification_config.hessian_approximation = fim_config
        fim = FisherInformation(full_config=config)

        config_copy = copy.deepcopy(config)
        gnh = GaussNewton(full_config=config_copy)

        params_flat = flatten_util.ravel_pytree(gnh.model_context.params)

        test_vector = jnp.ones(params_flat[0].shape)
        random_test_vector = jnp.array(
            jax.random.normal(jax.random.PRNGKey(0), shape=params_flat[0].shape)
        )

        fim_ivhp = fim.compute_ihvp(test_vector, damping=0.1)
        gnh_ihvp = gnh.compute_ihvp(test_vector, damping=0.1)
        assert VectorMetric.RELATIVE_ERROR.compute_single(fim_ivhp, gnh_ihvp) < 0.1, (
            f"Relative error too high: {VectorMetric.RELATIVE_ERROR.compute_single(fim_ivhp, gnh_ihvp)}"
        )

        fim_ivhp_random = fim.compute_ihvp(random_test_vector, damping=0.1)
        gnh_ihvp_random = gnh.compute_ihvp(random_test_vector, damping=0.1)
        assert (
            VectorMetric.RELATIVE_ERROR.compute_single(fim_ivhp_random, gnh_ihvp_random)
            < 0.1
        ), (
            f"Relative error too high: {VectorMetric.RELATIVE_ERROR.compute_single(fim_ivhp_random, gnh_ihvp_random)}"
        )

    def test_fim_gnh_hvp_comparison_classification(
        self,
        random_classification_config: Config,
    ):
        """Test comparing FIM and Gauss-Newton Hessians for classification."""
        config = random_classification_config

        fim_config = FisherInformationConfig(fisher_type="true")
        random_classification_config.hessian_approximation = fim_config
        fim = FisherInformation(full_config=config)

        config_copy = copy.deepcopy(config)
        gnh = GaussNewton(full_config=config_copy)

        params_flat = flatten_util.ravel_pytree(gnh.model_context.params)

        test_vector = jnp.ones(params_flat[0].shape)
        random_test_vector = jnp.array(
            jax.random.normal(jax.random.PRNGKey(0), shape=params_flat[0].shape)
        )

        fim_hvp = fim.compute_hvp(test_vector, damping=1)
        gnh_hvp = gnh.compute_hvp(test_vector, damping=1)

        assert VectorMetric.RELATIVE_ERROR.compute_single(fim_hvp, gnh_hvp) < 0.2, (
            f"Relative error too high: {VectorMetric.RELATIVE_ERROR.compute_single(fim_hvp, gnh_hvp)}"
        )

        fim_hvp_random = fim.compute_hvp(random_test_vector, damping=1)
        gnh_hvp_random = gnh.compute_hvp(random_test_vector, damping=1)
        assert (
            VectorMetric.RELATIVE_ERROR.compute_single(fim_hvp_random, gnh_hvp_random)
            < 0.2
        ), (
            f"Relative error too high: {VectorMetric.RELATIVE_ERROR.compute_single(fim_hvp_random, gnh_hvp_random)}"
        )

    def test_fim_gnh_comparison_regression(
        self,
        random_regression_config: Config,
    ):
        """Test comparing FIM and Gauss-Newton Hessians for regression."""
        config = random_regression_config

        fim_config = FisherInformationConfig(fisher_type="true")
        random_regression_config.hessian_approximation = fim_config
        fim = FisherInformation(full_config=config)

        config_copy = copy.deepcopy(config)
        gnh = GaussNewton(full_config=config_copy)

        fim_hessian = fim.compute_hessian(damping=0.1)
        gnh_hessian = gnh.compute_hessian(damping=0.1)

        assert (
            FullMatrixMetric.COSINE_SIMILARITY.compute(fim_hessian, gnh_hessian) > 0.95
        ), (
            f"Cosine similarity too low: {FullMatrixMetric.COSINE_SIMILARITY.compute(fim_hessian, gnh_hessian)}"
        )

    def test_fim_gnh_ihvp_comparison_regression(
        self,
        random_regression_config: Config,
    ):
        """Test comparing FIM and Gauss-Newton Hessians for regression."""
        config = random_regression_config

        fim_config = FisherInformationConfig(fisher_type="true")
        random_regression_config.hessian_approximation = fim_config
        fim = FisherInformation(full_config=config)

        config_copy = copy.deepcopy(config)
        gnh = GaussNewton(full_config=config_copy)

        params_flat = flatten_util.ravel_pytree(gnh.model_context.params)

        test_vector = jnp.ones(params_flat[0].shape)
        random_test_vector = jnp.array(
            jax.random.normal(jax.random.PRNGKey(0), shape=params_flat[0].shape)
        )

        fim_ivhp = fim.compute_ihvp(test_vector, damping=0.1)
        gnh_ihvp = gnh.compute_ihvp(test_vector, damping=0.1)
        assert VectorMetric.RELATIVE_ERROR.compute_single(fim_ivhp, gnh_ihvp) < 0.1, (
            f"Relative error too high: {VectorMetric.RELATIVE_ERROR.compute_single(fim_ivhp, gnh_ihvp)}"
        )

        fim_ivhp_random = fim.compute_ihvp(random_test_vector, damping=0.1)
        gnh_ihvp_random = gnh.compute_ihvp(random_test_vector, damping=0.1)
        assert (
            VectorMetric.RELATIVE_ERROR.compute_single(fim_ivhp_random, gnh_ihvp_random)
            < 0.1
        ), (
            f"Relative error too high: {VectorMetric.RELATIVE_ERROR.compute_single(fim_ivhp_random, gnh_ihvp_random)}"
        )

    def test_fim_gnh_hvp_comparison_regression(
        self,
        random_regression_config: Config,
    ):
        """Test comparing FIM and Gauss-Newton Hessians for regression."""
        config = random_regression_config

        fim_config = FisherInformationConfig(fisher_type="true")
        random_regression_config.hessian_approximation = fim_config
        fim = FisherInformation(full_config=config)

        config_copy = copy.deepcopy(config)
        gnh = GaussNewton(full_config=config_copy)

        params_flat = flatten_util.ravel_pytree(gnh.model_context.params)

        test_vector = jnp.ones(params_flat[0].shape)
        random_test_vector = jnp.array(
            jax.random.normal(jax.random.PRNGKey(0), shape=params_flat[0].shape)
        )

        fim_hvp = fim.compute_hvp(test_vector, damping=0.1)
        gnh_hvp = gnh.compute_hvp(test_vector, damping=0.1)

        assert VectorMetric.RELATIVE_ERROR.compute_single(fim_hvp, gnh_hvp) < 0.6, (
            f"Relative error too high: {VectorMetric.RELATIVE_ERROR.compute_single(fim_hvp, gnh_hvp)}"
        )

        fim_hvp_random = fim.compute_hvp(random_test_vector, damping=0.1)
        gnh_hvp_random = gnh.compute_hvp(random_test_vector, damping=0.1)
        assert (
            VectorMetric.RELATIVE_ERROR.compute_single(fim_hvp_random, gnh_hvp_random)
            < 0.6
        ), (
            f"Relative error too high: {VectorMetric.RELATIVE_ERROR.compute_single(fim_hvp_random, gnh_hvp_random)}"
        )
