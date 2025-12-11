import logging

import jax.numpy as jnp
from jax.random import PRNGKey

from src.config import Config, HessianApproximationConfig, ModelConfig
from src.hessians.approximator.ekfac import EKFACApproximator
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.kfac import KFACComputer
from src.hessians.utils.data import EKFACData, ModelContext
from src.hessians.utils.pseudo_gradients import generate_pseudo_targets
from src.utils.data.data import RandomClassificationDataset
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.loss import cross_entropy_loss
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
                "hidden_dim": [10],
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
        collected_data_path_snd=collector_run_dir_2,
    )
    ekfac_approximator.build(
        config=config,
        save_directory=config.hessian_approximation.directory,
    )

    # Load EKFAC data and compute Hessians
    ekfac_data, _ = EKFACApproximator.load_data(
        directory=config.hessian_approximation.directory
    )
    assert isinstance(ekfac_data, EKFACData)
    logger.info("K-FAC approximation loaded successfully.")

    damping = ekfac_data.mean_eigenvalues_aggregated * 0.1

    # Compute K-FAC Hessian
    kfac_computer = KFACComputer(compute_context=ekfac_data)
    kfac_hessian = kfac_computer.estimate_hessian(damping=damping)

    logger.info(f"K-FAC Hessian computed with shape: {kfac_hessian.shape}")

    # Compute E-KFAC Hessian
    ekfac_computer = EKFACComputer(compute_context=ekfac_data)
    ekfac_hessian = ekfac_computer.estimate_hessian(damping=damping)
    logger.info(f"E-KFAC Hessian computed with shape: {ekfac_hessian.shape}")

    # Compute the true Hessian (no precomputation needed)
    hessian_computer = HessianComputer(
        compute_context=ModelContext.create(
            dataset=dataset,
            model=model,
            params=params,
            loss_fn=cross_entropy_loss,
        )
    )
    full_hessian = hessian_computer.compute_hessian(damping=damping)
    logger.info(f"Full Hessian computed with shape: {full_hessian.shape}")

    # plot eigenvalue comparison
    import matplotlib.pyplot as plt

    kfac_eigenvalues = jnp.linalg.eigvalsh(kfac_hessian)
    ekfac_eigenvalues = jnp.linalg.eigvalsh(ekfac_hessian)
    full_eigenvalues = jnp.linalg.eigvalsh(full_hessian)

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
    plt.xlabel("Index")
    plt.title("Eigenvalue Comparison: Full Hessian vs K-FAC Hessian")
    plt.legend()
    plt.show()

    logging.info("Displaying eigenvalue comparison plot.")


if __name__ == "__main__":
    simple_run()
