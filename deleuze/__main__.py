import logging

import jax.numpy as jnp

from deleuze.config import Config, DatasetConfig, ModelConfig
from deleuze.hessians.approximator.ekfac import EKFACApproximator
from deleuze.hessians.approximator.kfac import KFACApproximator
from deleuze.hessians.computer.ekfac import EKFACComputer
from deleuze.hessians.computer.hessian import Hessian
from deleuze.hessians.computer.kfac import KFACComputer
from deleuze.hessians.utils.data import ModelContext
from deleuze.utils.loss import cross_entropy_loss
from deleuze.utils.optimizers import optimizer
from deleuze.utils.train import (
    check_saved_model,
    load_model_checkpoint,
    save_model_checkpoint,
    train_model,
)

from .models.mlp import MLP
from .utils.data.data import RandomClassificationDataset

logger = logging.getLogger(__name__)


def simple_run():
    seed = 42
    dataset = RandomClassificationDataset(
        n_samples=3000, n_features=50, n_informative=10, n_classes=3, seed=seed
    )
    model = MLP(
        input_dim=dataset.input_dim(),
        output_dim=dataset.output_dim(),
        hidden_dim=[10],
        seed=seed,
        activation="relu",
    )

    model_directory = "data/checkpoints"
    if check_saved_model(model_directory, model=model):
        params, _, _ = load_model_checkpoint(model_directory, model=model)
    else:
        model, params, _ = train_model(
            model,
            dataset.get_dataloader(batch_size=32, seed=seed),
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

    config = Config(
        data=DatasetConfig(
            dataset="random_classification",
            file_path=None,
        ),
        model=ModelConfig(
            model_name="mlp",
            params_path=f"{model_directory}/checkpoint.msgpack",
        ),
        seed=seed,
    )

    # compute KFAC components
    kfac_approximator = KFACApproximator(config=config)
    kfac_approximator.build(
        model=model,
        params=params,
        dataset=dataset,
        loss_fn=cross_entropy_loss,
    )
    approx_directory = "data/kfac_approximation"
    kfac_approximator.save(directory=approx_directory)

    kfac_approximator_loaded = KFACApproximator.load(directory=approx_directory)
    assert isinstance(kfac_approximator_loaded, KFACApproximator)
    logger.info("K-FAC approximation loaded successfully.")

    damping = kfac_approximator.data.mean_eigenvalues_aggregated * 0.1

    kfac_computer = KFACComputer(compute_context=kfac_approximator_loaded.data)
    kfac_hessian = kfac_computer.compute_hessian(damping=damping)

    logger.info(f"K-FAC Hessian computed with shape: {kfac_hessian.shape}")

    # compute EKFAC components
    ekfac_approximator = EKFACApproximator(config=config)
    ekfac_approximator.build(
        model=model,
        params=params,
        dataset=dataset,
        loss_fn=cross_entropy_loss,
    )
    approx_directory_ekfac = "data/ekfac_approximation"
    ekfac_approximator.save(directory=approx_directory_ekfac)
    ekfac_approximator_loaded = EKFACApproximator.load(directory=approx_directory_ekfac)
    assert isinstance(ekfac_approximator_loaded, EKFACApproximator)
    logger.info("E-KFAC approximation loaded successfully.")

    ekfac_computer = EKFACComputer(compute_context=ekfac_approximator_loaded.data)
    ekfac_hessian = ekfac_computer.compute_hessian(damping=damping)
    logger.info(f"E-KFAC Hessian computed with shape: {ekfac_hessian.shape}")

    # compute the hessian (no precomputation needed)

    hessian_computer = Hessian(
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
