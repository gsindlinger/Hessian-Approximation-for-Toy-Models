from src.config.config import Config
from src.config.dataset_config import RandomClassificationConfig
from src.config.hessian_approximation_config import KFACBuildConfig, KFACRunConfig
from src.config.model_config import LinearModelConfig
from src.config.training_config import TrainingConfig
from src.hessian_approximations.hessian.hessian import Hessian
from src.hessian_approximations.kfac.kfac_service import KFAC
from src.metrics.full_matrix_metrics import FullMatrixMetric
from src.metrics.vector_metrics import VectorMetric
from src.utils import logging
from src.utils.utils import sample_gradient_from_output_distribution_batched

logger = logging.get_stream_logger()
logger.info("Starting simple_run script...")


def run_simple_run():
    """Run a simple RUM experiment with a linear model on random classification data."""

    config = Config(
        dataset=RandomClassificationConfig(
            n_samples=10000,
            n_features=50,
            n_informative=15,
            n_classes=10,
            random_state=42,
            train_test_split=1,
        ),
        model=LinearModelConfig(loss="cross_entropy", hidden_dim=[10]),
        training=TrainingConfig(
            epochs=1000,
            batch_size=100,
            lr=0.001,
            optimizer="sgd",
            loss="cross_entropy",
            save_checkpoint=True,
        ),
    )

    # Compute true Hessian and K-FAC/EK-FAC approximations
    kfac_model = KFAC.setup_with_run_and_build_config(
        full_config=config,
        build_config=KFACBuildConfig(use_pseudo_targets=True),
        run_config=KFACRunConfig(use_eigenvalue_correction=False),
    )
    kfac_hessian = kfac_model.compute_hessian(damping=kfac_model.damping())

    ekfac_model = KFAC.setup_with_run_and_build_config(
        full_config=config,
        build_config=KFACBuildConfig(use_pseudo_targets=True),
        run_config=KFACRunConfig(use_eigenvalue_correction=True),
    )
    ekfac_hessian = ekfac_model.compute_hessian(damping=kfac_model.damping())

    hessian = Hessian(full_config=config).compute_hessian(damping=kfac_model.damping())

    # Compare directly the Hessians
    metric = FullMatrixMetric.RELATIVE_FROBENIUS
    kfac_comparison_result = kfac_model.compare_hessians(
        comparison_matrix=hessian,
        metric=metric,
    )

    ekfac_comparison_result = ekfac_model.compare_hessians(
        comparison_matrix=hessian,
        metric=metric,
    )

    # IHVP
    test_vectors_1 = sample_gradient_from_output_distribution_batched(
        model_data=kfac_model.model_context,
        n_vectors=5,
    )

    test_vectors_2 = sample_gradient_from_output_distribution_batched(
        model_data=ekfac_model.model_context,
        n_vectors=5,
    )

    kfac_ihvp = kfac_model.compute_ihvp(
        vectors=test_vectors_1, damping=kfac_model.damping()
    )
    ekfac_ihvp = ekfac_model.compute_ihvp(
        vectors=test_vectors_1, damping=kfac_model.damping()
    )
    true_ihvp = Hessian(full_config=config).compute_ihvp(
        vectors=test_vectors_1, damping=kfac_model.damping()
    )

    metric_ihvp = VectorMetric.RELATIVE_ERROR
    kfac_ihvp_comparison = metric_ihvp.compute(
        v1=true_ihvp,
        v2=kfac_ihvp,
        reduction="mean",
    )

    ekfac_ihvp_comparison = metric_ihvp.compute(
        v1=true_ihvp,
        v2=ekfac_ihvp,
        reduction="mean",
    )

    metric_ihvp_dot_prod = VectorMetric.INNER_PRODUCT_DIFF

    kfac_ihvp_dot_prod = metric_ihvp_dot_prod.compute(
        v1=true_ihvp,
        v2=kfac_ihvp,
        x=test_vectors_2,
        reduction="mean",
    )

    ekfac_ihvp_dot_prod = metric_ihvp_dot_prod.compute(
        v1=true_ihvp,
        v2=ekfac_ihvp,
        x=test_vectors_2,
        reduction="mean",
    )

    print("**************************************")
    print("K-FAC vs True Hessian Comparison:")
    print(kfac_comparison_result)
    print("\nEK-FAC vs True Hessian Comparison:")
    print(ekfac_comparison_result)

    print("**************************************")
    print("\nK-FAC IHVP vs True IHVP Comparison (Relative Error):")
    print(kfac_ihvp_comparison)
    print("\nEK-FAC IHVP vs True IHVP Comparison (Relative Error):")
    print(ekfac_ihvp_comparison)

    print("**************************************")
    print("\nK-FAC IHVP vs True IHVP Comparison (Inner Product Difference):")
    print(kfac_ihvp_dot_prod)
    print("\nEK-FAC IHVP vs True IHVP Comparison (Inner Product Difference):")
    print(ekfac_ihvp_dot_prod)


if __name__ == "__main__":
    run_simple_run()
