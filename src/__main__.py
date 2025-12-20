import json
import logging

import jax.numpy as jnp
from jax.random import PRNGKey

from src.config import Config, HessianApproximationConfig, ModelConfig
from src.hessians.approximator.ekfac import EKFACApproximator
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.fim import FIMComputer
from src.hessians.computer.fim_block import FIMBlockComputer
from src.hessians.computer.gnh import GNHComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.hessian_block import BlockHessianComputer
from src.hessians.computer.kfac import KFACComputer
from src.hessians.utils.data import EKFACData, ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets, sample_gradients
from src.utils.data.data import RandomClassificationDataset
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.loss import cross_entropy_loss
from src.utils.metrics.full_matrix_metrics import MATRIX_METRICS
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.mlp import MLP
from src.utils.optimizers import optimizer
from src.utils.train import (
    check_saved_model,
    load_model_checkpoint,
    save_model_checkpoint,
    train_model,
)

logger = logging.getLogger(__name__)


def simple_run():
    # Define config which serves mostly as reference for paths and model metadata
    config = Config(
        dataset_path="random_classification",
        model=ModelConfig(
            model_name="mlp",
            directory="data/checkpoints/mlp",
            metadata={
                "hidden_dim": [10, 10],
                "activation": "relu",
            },
        ),
        seed=42,
        hessian_approximation=HessianApproximationConfig(
            method="KFAC",
            directory="data/hessian_approximations/",
        ),
    )

    # Prepare dataset and model
    dataset = RandomClassificationDataset(
        n_samples=3000, n_features=50, n_informative=10, n_classes=3, seed=config.seed
    )
    assert config.model.metadata is not None, (
        "Model metadata must be provided in config."
    )
    model = MLP(
        input_dim=dataset.input_dim(),
        output_dim=dataset.output_dim(),
        hidden_dim=config.model.metadata["hidden_dim"],
        seed=config.seed,
        activation=config.model.metadata["activation"],
    )

    model_directory = config.model.directory
    assert model_directory is not None, "Model directory must be provided in config."

    # Train or load model
    if check_saved_model(model_directory, model=model):
        params, _, _ = load_model_checkpoint(model_directory, model=model)
    else:
        model, params, _ = train_model(
            model,
            dataset.get_dataloader(
                batch_size=JAXDataLoader.get_batch_size(), seed=config.seed
            ),
            loss_fn=cross_entropy_loss,
            optimizer=optimizer("adamw", lr=1e-3),
            epochs=100,
        )
        save_model_checkpoint(
            model=model,
            params=params,
            directory=model_directory,
        )
    logger.info("Model trained and loaded successfully.")

    # Generate gradient samples to check for HVP and IHVP later
    gradient_samples_1 = sample_gradients(
        model=model,
        params=params,
        inputs=dataset.inputs,
        targets=dataset.targets,
        loss_fn=cross_entropy_loss,
        n_vectors=50,
        rng_key=PRNGKey(config.seed),
    )

    gradient_samples_2 = sample_gradients(
        model=model,
        params=params,
        inputs=dataset.inputs,
        targets=dataset.targets,
        loss_fn=cross_entropy_loss,
        n_vectors=50,
        rng_key=PRNGKey(config.seed + 1),
    )

    # Generate pseudo-targets for EKFAC/FIM (run twice to get two different datasets for two different runs)
    collector_data_1 = dataset.replace_targets(
        generate_pseudo_targets(
            model=model,
            inputs=dataset.inputs,
            params=params,
            loss_fn=cross_entropy_loss,
            rng_key=PRNGKey(config.seed),
        )
    )

    collector_data_2 = dataset.replace_targets(
        generate_pseudo_targets(
            model=model,
            inputs=dataset.inputs,
            params=params,
            loss_fn=cross_entropy_loss,
            rng_key=PRNGKey(config.seed + 1),
        )
    )

    # Collect activations and gradients for computing the covariances and their eigenvectors
    collector_run_dir_1 = "data/activation_gradient_collector/run1/"
    collector = CollectorActivationsGradients(model=model, params=params)
    collector.collect(
        collector_data_1.inputs,
        collector_data_1.targets,
        cross_entropy_loss,
        save_directory=collector_run_dir_1,
    )

    # Collect activations and gradients for computing the eigenvalue corrections
    collector_run_dir_2 = "data/activation_gradient_collector/run2/"
    collector.collect(
        collector_data_2.inputs,
        collector_data_2.targets,
        cross_entropy_loss,
        save_directory=collector_run_dir_2,
    )

    # Compute KFAC components and save them
    ekfac_approximator = EKFACApproximator(
        collected_data_path=collector_run_dir_1,
        collected_data_path_second=collector_run_dir_2,
    )
    ekfac_approximator.build(
        config=config,
        save_directory=config.hessian_approximation.directory,
    )

    # Load EKFAC data and compute Hessians
    assert config.hessian_approximation.directory is not None, (
        "Hessian approximation directory must be provided in config."
    )
    ekfac_data, _ = EKFACApproximator.load_data(
        directory=config.hessian_approximation.directory
    )
    assert isinstance(ekfac_data, EKFACData)
    logger.info("K-FAC approximation loaded successfully.")

    damping = ekfac_data.mean_eigenvalues_aggregated * 0.1

    # Compute K-FAC Hessian
    kfac_computer = KFACComputer(compute_context=ekfac_data)
    kfac_hessian = kfac_computer.estimate_hessian(damping=damping)
    kfac_hvp = kfac_computer.estimate_hvp(vectors=gradient_samples_1, damping=damping)
    kfac_ihvp = kfac_computer.estimate_ihvp(vectors=gradient_samples_2, damping=damping)
    logger.info(
        f"K-FAC Hessian, HVP, and IHVP computed with shapes: {kfac_hessian.shape}, {kfac_hvp.shape}, {kfac_ihvp.shape}"
    )

    # Compute E-KFAC Hessian
    ekfac_computer = EKFACComputer(compute_context=ekfac_data)
    ekfac_hessian = ekfac_computer.estimate_hessian(damping=damping)
    ekfac_hvp = ekfac_computer.estimate_hvp(vectors=gradient_samples_1, damping=damping)
    ekfac_ihvp = ekfac_computer.estimate_ihvp(
        vectors=gradient_samples_2, damping=damping
    )
    logger.info(
        f"E-KFAC Hessian, HVP, and IHVP computed with shapes: {ekfac_hessian.shape}, {ekfac_hvp.shape}, {ekfac_ihvp.shape}"
    )

    # Compute FIM
    model_context = ModelContext.create(
        dataset=dataset,
        model=model,
        params=params,
        loss_fn=cross_entropy_loss,
    )
    fim_computer = FIMComputer(compute_context=model_context)
    fim = fim_computer.estimate_hessian(damping=damping)
    fim_hvp = fim_computer.estimate_hvp(vectors=gradient_samples_1, damping=damping)
    fim_ihvp = fim_computer.estimate_ihvp(vectors=gradient_samples_2, damping=damping)
    logger.info(
        f"FIM, HVP, and IHVP computed with shapes: {fim.shape}, {fim_hvp.shape}, {fim_ihvp.shape}"
    )

    # Compute Block FIM
    block_fim_data = CollectorActivationsGradients.load(directory=collector_run_dir_1)
    block_fim_computer = FIMBlockComputer(compute_context=block_fim_data)
    block_fim = block_fim_computer.estimate_hessian(damping=damping)
    block_fim_hvp = block_fim_computer.estimate_hvp(
        vectors=gradient_samples_1, damping=damping
    )
    block_fim_ihvp = block_fim_computer.estimate_ihvp(
        vectors=gradient_samples_2, damping=damping
    )
    logger.info(
        f"Block FIM, HVP, and IHVP computed with shapes: {block_fim.shape}, {block_fim_hvp.shape}, {block_fim_ihvp.shape}"
    )

    # Compute GNH
    gnh_computer = GNHComputer(compute_context=model_context)
    gnh = gnh_computer.estimate_hessian(damping=damping)
    gnh_hvp = gnh_computer.estimate_hvp(vectors=gradient_samples_1, damping=damping)
    gnh_ihvp = gnh_computer.estimate_ihvp(vectors=gradient_samples_2, damping=damping)
    logger.info(
        f"GNH, HVP, and IHVP computed with shapes: {gnh.shape}, {gnh_hvp.shape}, {gnh_ihvp.shape}"
    )

    # Compute the true Hessian (no precomputation needed)
    hessian_computer = HessianComputer(compute_context=model_context)
    full_hessian = hessian_computer.compute_hessian(damping=damping)
    full_hvp = hessian_computer.compute_hvp(vectors=gradient_samples_1, damping=damping)
    full_ihvp = hessian_computer.compute_ihvp(
        vectors=gradient_samples_2, damping=damping
    )
    logger.info(
        f"Full Hessian, HVP, and IHVP computed with shapes: {full_hessian.shape}, {full_hvp.shape}, {full_ihvp.shape}"
    )

    # Compute block Hessian for comparison
    block_hessian_computer = BlockHessianComputer(compute_context=model_context)
    block_hessian = block_hessian_computer.estimate_hessian(damping=damping)
    block_hessian_hvp = block_hessian_computer.estimate_hvp(
        vectors=gradient_samples_1, damping=damping
    )
    block_hessian_ihvp = block_hessian_computer.estimate_ihvp(
        vectors=gradient_samples_2, damping=damping
    )
    logger.info(
        f"Block Hessian computed with shape: {block_hessian.shape}, {block_hessian_hvp.shape}, {block_hessian_ihvp.shape}"
    )

    matrix_results = {}
    matrix_metrics = MATRIX_METRICS["all_matrix"]
    for metric in matrix_metrics:
        kfac_vs_full = kfac_computer.compare_full_hessian_estimates(
            comparison_matrix=full_hessian,
            metric=metric,
            damping=damping,
        )
        ekfac_vs_full = ekfac_computer.compare_full_hessian_estimates(
            comparison_matrix=full_hessian,
            metric=metric,
            damping=damping,
        )
        fim_vs_full = fim_computer.compare_full_hessian_estimates(
            comparison_matrix=full_hessian,
            metric=metric,
            damping=damping,
        )
        block_fim_vs_full = block_fim_computer.compare_full_hessian_estimates(
            comparison_matrix=full_hessian,
            metric=metric,
            damping=damping,
        )
        gnh_vs_full = gnh_computer.compare_full_hessian_estimates(
            comparison_matrix=full_hessian,
            metric=metric,
            damping=damping,
        )
        block_hessian_vs_full = block_hessian_computer.compare_full_hessian_estimates(
            comparison_matrix=full_hessian,
            metric=metric,
            damping=damping,
        )
        kfac_vs_gnh = kfac_computer.compare_full_hessian_estimates(
            comparison_matrix=gnh,
            metric=metric,
            damping=damping,
        )
        ekfac_vs_gnh = ekfac_computer.compare_full_hessian_estimates(
            comparison_matrix=gnh,
            metric=metric,
            damping=damping,
        )
        fim_vs_gnh = fim_computer.compare_full_hessian_estimates(
            comparison_matrix=gnh,
            metric=metric,
            damping=damping,
        )
        block_fim_vs_gnh = block_fim_computer.compare_full_hessian_estimates(
            comparison_matrix=gnh,
            metric=metric,
            damping=damping,
        )

        matrix_results[metric.value] = {
            "kfac_vs_full": float(kfac_vs_full),
            "ekfac_vs_full": float(ekfac_vs_full),
            "fim_vs_full": float(fim_vs_full),
            "block_fim_vs_full": float(block_fim_vs_full),
            "gnh_vs_full": float(gnh_vs_full),
            "block_hessian_vs_full": float(block_hessian_vs_full),
            "kfac_vs_gnh": float(kfac_vs_gnh),
            "ekfac_vs_gnh": float(ekfac_vs_gnh),
            "fim_vs_gnh": float(fim_vs_gnh),
            "block_fim_vs_gnh": float(block_fim_vs_gnh),
        }

    hvp_results = {}
    for metric in VectorMetric.all_metrics():
        kfac_vs_full_hvp = metric.compute(full_hvp, kfac_hvp, gradient_samples_2)
        ekfac_vs_full_hvp = metric.compute(full_hvp, ekfac_hvp, gradient_samples_2)
        fim_vs_full_hvp = metric.compute(full_hvp, fim_hvp, gradient_samples_2)
        block_fim_vs_full_hvp = metric.compute(
            full_hvp, block_fim_hvp, gradient_samples_2
        )
        gnh_vs_full_hvp = metric.compute(full_hvp, gnh_hvp, gradient_samples_2)
        block_hessian_vs_full_hvp = metric.compute(
            full_hvp, block_hessian_hvp, gradient_samples_2
        )
        kfac_vs_gnh_hvp = metric.compute(gnh_hvp, kfac_hvp, gradient_samples_2)
        ekfac_vs_gnh_hvp = metric.compute(gnh_hvp, ekfac_hvp, gradient_samples_2)
        fim_vs_gnh_hvp = metric.compute(gnh_hvp, fim_hvp, gradient_samples_2)
        block_fim_vs_gnh_hvp = metric.compute(
            gnh_hvp, block_fim_hvp, gradient_samples_2
        )

        hvp_results[metric.name] = {
            "kfac_vs_full_hvp": float(kfac_vs_full_hvp),
            "ekfac_vs_full_hvp": float(ekfac_vs_full_hvp),
            "fim_vs_full_hvp": float(fim_vs_full_hvp),
            "block_fim_vs_full_hvp": float(block_fim_vs_full_hvp),
            "gnh_vs_full_hvp": float(gnh_vs_full_hvp),
            "block_hessian_vs_full_hvp": float(block_hessian_vs_full_hvp),
            "kfac_vs_gnh_hvp": float(kfac_vs_gnh_hvp),
            "ekfac_vs_gnh_hvp": float(ekfac_vs_gnh_hvp),
            "fim_vs_gnh_hvp": float(fim_vs_gnh_hvp),
            "block_fim_vs_gnh_hvp": float(block_fim_vs_gnh_hvp),
        }

    ihvp_results = {}
    for metric in VectorMetric.all_metrics():
        kfac_vs_full_ihvp = metric.compute(full_ihvp, kfac_ihvp, gradient_samples_2)
        ekfac_vs_full_ihvp = metric.compute(full_ihvp, ekfac_ihvp, gradient_samples_2)
        fim_vs_full_ihvp = metric.compute(full_ihvp, fim_ihvp, gradient_samples_2)
        block_fim_vs_full_ihvp = metric.compute(
            full_ihvp, block_fim_ihvp, gradient_samples_2
        )
        gnh_vs_full_ihvp = metric.compute(full_ihvp, gnh_ihvp, gradient_samples_2)
        block_hessian_vs_full_ihvp = metric.compute(
            full_ihvp, block_hessian_ihvp, gradient_samples_2
        )
        kfac_vs_gnh_ihvp = metric.compute(gnh_ihvp, kfac_ihvp, gradient_samples_2)
        ekfac_vs_gnh_ihvp = metric.compute(gnh_ihvp, ekfac_ihvp, gradient_samples_2)
        fim_vs_gnh_ihvp = metric.compute(gnh_ihvp, fim_ihvp, gradient_samples_2)
        block_fim_vs_gnh_ihvp = metric.compute(
            gnh_ihvp, block_fim_ihvp, gradient_samples_2
        )
        ihvp_results[metric.name] = {
            "kfac_vs_full_ihvp": float(kfac_vs_full_ihvp),
            "ekfac_vs_full_ihvp": float(ekfac_vs_full_ihvp),
            "fim_vs_full_ihvp": float(fim_vs_full_ihvp),
            "block_fim_vs_full_ihvp": float(block_fim_vs_full_ihvp),
            "gnh_vs_full_ihvp": float(gnh_vs_full_ihvp),
            "block_hessian_vs_full_ihvp": float(block_hessian_vs_full_ihvp),
            "kfac_vs_gnh_ihvp": float(kfac_vs_gnh_ihvp),
            "ekfac_vs_gnh_ihvp": float(ekfac_vs_gnh_ihvp),
            "fim_vs_gnh_ihvp": float(fim_vs_gnh_ihvp),
            "block_fim_vs_gnh_ihvp": float(block_fim_vs_gnh_ihvp),
        }

    logging.info(f"Matrix comparison results: {json.dumps(matrix_results, indent=2)}")
    logging.info(f"HVP comparison results: {json.dumps(hvp_results, indent=2)}")
    logging.info(f"IHVP comparison results: {json.dumps(ihvp_results, indent=2)}")

    # plot eigenvalue comparison
    import matplotlib.pyplot as plt

    num_samples = 10
    kfac_eigenvalues = jnp.linalg.eigvalsh(kfac_hessian)[-num_samples:]
    ekfac_eigenvalues = jnp.linalg.eigvalsh(ekfac_hessian)[-num_samples:]
    full_eigenvalues = jnp.linalg.eigvalsh(full_hessian)[-num_samples:]
    fim_eigenvalues = jnp.linalg.eigvalsh(fim)[-num_samples:]
    block_fim_eigenvalues = jnp.linalg.eigvalsh(block_fim)[-num_samples:]
    gnh_eigenvalues = jnp.linalg.eigvalsh(gnh)[-num_samples:]
    block_hessian_eigenvalues = jnp.linalg.eigvalsh(block_hessian)[-num_samples:]

    plt.figure(figsize=(8, 6))
    plt.plot(
        full_eigenvalues,
        label="Full Hessian Eigenvalues",
        marker="o",
        linestyle="None",
        markersize=4,
    )
    plt.plot(
        kfac_eigenvalues,
        label="K-FAC Hessian Eigenvalues",
        marker="x",
        linestyle="None",
        markersize=4,
    )
    plt.plot(
        ekfac_eigenvalues,
        label="E-KFAC Hessian Eigenvalues",
        marker="^",
        linestyle="None",
        markersize=4,
    )
    plt.plot(
        fim_eigenvalues,
        label="FIM Eigenvalues",
        marker="s",
        linestyle="None",
        markersize=4,
    )
    plt.plot(
        block_fim_eigenvalues,
        label="Block FIM Eigenvalues",
        marker="v",
        linestyle="None",
        markersize=4,
    )
    plt.plot(
        gnh_eigenvalues,
        label="GNH Eigenvalues",
        marker="d",
        linestyle="None",
        markersize=4,
    )
    plt.plot(
        block_hessian_eigenvalues,
        label="Block Hessian Eigenvalues",
        marker="*",
        linestyle="None",
        markersize=4,
    )
    plt.ylabel("Eigenvalue")
    plt.xlabel("Index")
    plt.title("Eigenvalue Comparison: Full Hessian vs K-FAC Hessian")
    plt.legend()
    plt.show()

    logging.info("Displaying eigenvalue comparison plot.")


if __name__ == "__main__":
    simple_run()
