import jax.numpy as jnp

from config.config import Config
from config.dataset_config import RandomClassificationConfig
from config.hessian_approximation_config import KFACBuildConfig, KFACRunConfig
from config.model_config import LinearModelConfig
from config.training_config import TrainingConfig
from hessian_approximations.gauss_newton.gauss_newton import GaussNewton
from hessian_approximations.hessian.hessian import Hessian
from hessian_approximations.kfac.kfac import KFAC
from metrics.full_matrix_metrics import FullMatrixMetric
from metrics.vector_metrics import VectorMetric
from models.train import train_or_load
from utils.utils import sample_gradient_from_output_distribution_batched


def main():
    config = Config(
        model=LinearModelConfig(
            loss="cross_entropy",
            hidden_dim=[20, 10, 5],
        ),
        dataset=RandomClassificationConfig(
            n_samples=2000,
            n_features=50,
            n_classes=20,
            n_informative=10,
            train_test_split=1,
        ),
        training=TrainingConfig(
            epochs=1000,
            lr=0.001,
            optimizer="sgd",
            batch_size=100,
            loss="cross_entropy",
        ),
        seed=42,
    )

    test_vectors = sample_gradient_from_output_distribution_batched(
        model_data=train_or_load(config),
        n_vectors=10,
    )

    hessian_ihvp = Hessian(full_config=config).compute_ihvp(
        vectors=test_vectors, damping=0.1
    )
    gnh_ihvp = GaussNewton(full_config=config).compute_ihvp(
        vectors=test_vectors, damping=0.1
    )
    ekfac_model = KFAC.setup_with_run_and_build_config(
        full_config=config,
        build_config=KFACBuildConfig(use_pseudo_targets=True),
        run_config=KFACRunConfig(use_eigenvalue_correction=True),
    )
    ekfac_ihvp = ekfac_model.compute_ihvp(vectors=test_vectors, damping=0.1)

    kfac_model = KFAC.setup_with_run_and_build_config(
        full_config=config,
        build_config=KFACBuildConfig(use_pseudo_targets=True),
        run_config=KFACRunConfig(use_eigenvalue_correction=False),
    )
    kfac_ihvp = kfac_model.compute_ihvp(vectors=test_vectors, damping=0.1)

    hessian = Hessian(full_config=config).compute_hessian(damping=0.1)
    gnh = GaussNewton(full_config=config).compute_hessian(damping=0.1)
    ekfac_hessian = ekfac_model.compute_hessian(damping=0.1)
    kfac_hessian = kfac_model.compute_hessian(damping=0.1)

    matrix_diff = FullMatrixMetric.RELATIVE_FROBENIUS.compute(hessian, gnh)
    print("Relative Frobenius norm between Hessian and GNH:", matrix_diff)

    # plot eigenvalues of both hessians
    import matplotlib.pyplot as plt

    hessian_eigenvalues = jnp.linalg.eigvals(hessian)
    gnh_eigenvalues = jnp.linalg.eigvals(gnh)
    ekfac_eigenvalues = jnp.linalg.eigvals(ekfac_hessian)
    kfac_eigenvalues = jnp.linalg.eigvals(kfac_hessian)

    plt.figure(figsize=(12, 8))

    # Sort eigenvalues for better visualization
    hessian_sorted = sorted(hessian_eigenvalues, reverse=True)[:50]
    gnh_sorted = sorted(gnh_eigenvalues, reverse=True)[:50]
    ekfac_sorted = sorted(ekfac_eigenvalues, reverse=True)[:50]
    kfac_sorted = sorted(kfac_eigenvalues, reverse=True)[:50]

    plt.plot(
        hessian_sorted,
        marker="o",
        markersize=4,
        linewidth=2,
        label="Hessian",
        color="#2E86AB",
        alpha=0.8,
    )
    plt.plot(
        gnh_sorted,
        marker="s",
        markersize=4,
        linewidth=2,
        label="Gauss-Newton",
        color="#A23B72",
        alpha=0.8,
    )
    plt.plot(
        ekfac_sorted,
        marker="^",
        markersize=4,
        linewidth=2,
        label="E-KFAC",
        color="#F18F01",
        alpha=0.8,
    )
    plt.plot(
        kfac_sorted,
        marker="D",
        markersize=4,
        linewidth=2,
        label="K-FAC",
        color="#C73E1D",
        alpha=0.8,
    )

    plt.title(
        "Eigenvalue Comparison Across Hessian Approximation Methods",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Eigenvalue Index (sorted descending)", fontsize=12)
    plt.ylabel("Eigenvalue Magnitude", fontsize=12)
    plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # plot last 20 entries of hessians as colour plot in subplots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(hessian[-20:, -20:], cmap="viridis")
    plt.title("Hessian (last 20x20 block)")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gnh[-20:, -20:], cmap="viridis")
    plt.title("GNH (last 20x20 block)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    print(
        "Relative Error between Hessian IHVP and GNH IHVP:",
        VectorMetric.RELATIVE_ERROR.compute(hessian_ihvp, gnh_ihvp),
    )

    # plot the first 4 vectors as lines plots each in subplots
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(4, 8):
        ax = axs[i // 2, i % 2]
        ax.plot(hessian_ihvp[i], label="Hessian IHVP")
        ax.plot(gnh_ihvp[i], label="GNH IHVP", linestyle="dashed")
        ax.set_title(f"IHVP Comparison for Vector {i + 1}")
        ax.legend()
    plt.tight_layout()
    plt.show()

    print("Hessian comparison complete.")


if __name__ == "__main__":
    print("Starting K-FAC Hessian computation...")
    main()
