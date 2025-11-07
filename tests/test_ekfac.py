from typing import Any, Generator, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import flatten_util

from config.config import (
    Config,
    KFACBuildConfig,
    KFACConfig,
    KFACRunConfig,
    LinearModelConfig,
    RandomClassificationConfig,
    TrainingConfig,
)
from data.data import AbstractDataset
from hessian_approximations.factory import (
    create_hessian_by_name,
    hessian_approximation,
)
from hessian_approximations.gauss_newton.gauss_newton import GaussNewton
from hessian_approximations.hessian.hessian import Hessian
from hessian_approximations.kfac.kfac import KFAC, KFACStorage
from main import train
from models.train import ApproximationModel, get_loss_fn

ModelTuple = Tuple[ApproximationModel, AbstractDataset, Any, Config]


class TestEKFAC:
    """Tests for various Hessian approximation methods including E-KFAC."""

    @pytest.fixture(params=["linear", "multi_layer"], scope="session")
    def random_classification_config(self, request):
        """Parametrized configuration for testing across different setups."""
        if request.param == "linear":
            return Config(
                dataset=RandomClassificationConfig(
                    n_samples=10000,
                    n_features=10,
                    n_informative=5,
                    n_classes=2,
                    random_state=42,
                    train_test_split=1,
                ),
                model=LinearModelConfig(loss="cross_entropy", hidden_dim=[]),
                training=TrainingConfig(
                    epochs=100,
                    batch_size=100,
                    lr=0.001,
                    optimizer="sgd",
                    loss="cross_entropy",
                    save_checkpoint=True,
                ),
            )
        elif request.param == "multi_layer":
            return Config(
                dataset=RandomClassificationConfig(
                    n_samples=10000,
                    n_features=10,
                    n_informative=5,
                    n_classes=2,
                    random_state=42,
                    train_test_split=1,
                ),
                model=LinearModelConfig(loss="cross_entropy", hidden_dim=[5]),
                training=TrainingConfig(
                    epochs=100,
                    batch_size=100,
                    lr=0.001,
                    optimizer="sgd",
                    loss="cross_entropy",
                    save_checkpoint=True,
                ),
            )

    @pytest.fixture(scope="session")
    def trained_model(
        self, random_classification_config, request
    ) -> Generator[ModelTuple, None, None]:
        """Train a small model for testing. Cached per configuration."""

        model, dataset, params = train(random_classification_config, reload_model=True)

        ekfac_build_config = KFACBuildConfig()
        ekfac_model = KFAC(
            config=KFACConfig(
                build_config=ekfac_build_config,
            )
        )
        ekfac_model.get_ekfac_components(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=get_loss_fn(random_classification_config.model.loss),
        )

        yield model, dataset, params, random_classification_config

        # Cleanup happens after all tests for this parameter are done
        KFACStorage().delete_storage()

    # Keep the session-level cleanup as a fallback
    @pytest.fixture(scope="session", autouse=True)
    def cleanup_kfac_data(self):
        yield  # run all tests
        # Final cleanup after all sessions
        KFACStorage().delete_storage()

    def test_ekfac_existence(self, trained_model: ModelTuple):
        """Test if Hessian approximation can be computed without errors."""
        model, dataset, params, config = trained_model

        x, y = dataset.get_train_data()
        hessian_method = create_hessian_by_name("kfac")

        ekfac_hessian = hessian_approximation(
            method=hessian_method,
            model=model,
            parameters=params,
            test_data=jnp.asarray(x),
            test_targets=jnp.asarray(y),
            loss=get_loss_fn(config.model.loss),
        )

        assert ekfac_hessian is not None
        assert jnp.isfinite(ekfac_hessian).all()

    def test_gradient_consistency(self, trained_model: ModelTuple):
        """
        Verify E-KFAC collector vs true gradients (sanity test).
        The following should hold for each linear layer l:

            ∇_{W_l} log p(y | x; θ) = s_l a_{l-1}^T

        where:
            - a_{l-1} = activations of the previous layer
            - s_l = preactivation gradients = ∇_{W_l a_{l-1}} log p(y | x; θ)

        This test checks that the EKFAC collector correctly captures (a_{l-1}, s_l) and that the reconstructed gradient (s_l a_{l-1}^T) matches the true gradient computed via autodiff.
        """
        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        def normal_loss_fn(params, x, y, model):
            """Standard loss function without EKFAC wrapper."""
            pred = model.apply(params, x)
            return loss_fn(pred, y, reduction="sum")  # Sum to match EKFAC behavior

        # Compute true gradients via autodiff
        ground_truth_grads = jax.grad(normal_loss_fn)(
            params, dataset.get_train_data()[0], dataset.get_train_data()[1], model
        )

        # Collect EKFAC statistics
        # Need to reload data, otherwise collector data is empty
        ekfac_approx = KFAC(
            config=KFACConfig(
                build_config=KFACBuildConfig(recalc_ekfac_components=True)
            )
        )
        ekfac_approx.get_ekfac_components(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )

        # Compare each layer’s gradients
        for key in ground_truth_grads["params"].keys():
            gt_grad = ground_truth_grads["params"][key]["kernel"]

            a = ekfac_approx.collector.captured_data[key][0]
            g = ekfac_approx.collector.captured_data[key][1]

            # ∇_{W_l} ≈ s_l a_{l−1}ᵀ
            ag = jnp.einsum("ni,no->io", a, g)

            assert jnp.allclose(gt_grad, ag, atol=1e-4), (
                f"Gradient mismatch for layer {key}"
            )

            # Verify that E[a ⊗ g] matches E[∇_{W_l}] (vectorized check)
            a_kron_g = jnp.zeros((a.shape[1] * g.shape[1],))
            for i in range(a.shape[0]):
                a_kron_g += jnp.kron(a[i], g[i])
            a_kron_g = a_kron_g / a.shape[0]

            gt_grad_flat = gt_grad.reshape(-1) / a.shape[0]
            assert jnp.allclose(gt_grad_flat, a_kron_g, atol=1e-4), (
                f"E[a kron g] does not match E[ground truth gradient] for layer {key}"
            )

    def test_kfac_hessian(self, trained_model: ModelTuple):
        """Test KFAC Hessian computation (without eigenvalue correction)."""

        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        hessian = Hessian().compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )

        gnh = GaussNewton().compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )

        kfac_config = KFACRunConfig(
            use_eigenvalue_correction=False,
            damping_lambda=0.0,
        )
        kfac_model = KFAC(config=KFACConfig(run_config=kfac_config))
        kfac = kfac_model.compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )

        assert hessian.shape == gnh.shape == kfac.shape, (
            "Hessian, GNH, and KFAC Hessians should have the same shape."
        )

    def test_ekfac_hessian(self, trained_model: ModelTuple):
        """Test KFAC Hessian computation (without eigenvalue correction)."""

        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        hessian = Hessian().compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )

        gnh = GaussNewton().compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )

        ekfac_build_config = KFACBuildConfig(
            recalc_ekfac_components=True,
            use_pseudo_targets=True,
        )
        ekfac_config = KFACRunConfig(
            use_eigenvalue_correction=True,
        )
        ekfac_model = KFAC(
            config=KFACConfig(run_config=ekfac_config, build_config=ekfac_build_config)
        )
        ekfac = ekfac_model.compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )

        assert hessian.shape == gnh.shape == ekfac.shape, (
            "Hessian, GNH, and KFAC Hessians should have the same shape."
        )

    def test_ekfac_batched_processing_is_close_to_full_data(
        self, trained_model: ModelTuple
    ):
        """Test that E-KFAC with batch processing yields the same result as without batching on a single run."""

        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        ekfac_run_config = KFACRunConfig(use_eigenvalue_correction=True)

        ekfac_full_data_config = KFACBuildConfig(
            recalc_ekfac_components=True,
            collector_batch_size=None,  # Full data
            use_pseudo_targets=False,
        )

        ekfac_full_data_model = KFAC(
            config=KFACConfig(
                build_config=ekfac_full_data_config, run_config=ekfac_run_config
            )
        )
        ekfac_full_data = ekfac_full_data_model.compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )

        ekfac_batched_config = KFACBuildConfig(
            recalc_ekfac_components=False,
            collector_batch_size=100,  # Smaller batches
            use_pseudo_targets=False,
        )
        ekfac_batched_model = KFAC(
            config=KFACConfig(
                build_config=ekfac_batched_config, run_config=ekfac_run_config
            )
        )
        ekfac_batched = ekfac_batched_model.compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(dataset.get_train_data()[0]),
            training_targets=jnp.asarray(dataset.get_train_data()[1]),
            loss_fn=loss_fn,
        )

        assert ekfac_full_data.shape == ekfac_batched.shape, (
            "E-KFAC Hessians from full data and batched processing should have the same shape."
        )

        # check whether covariances, eigenvectors, and eigenvalues match between the two methods
        for layer_name, (
            activations_cov,
            gradients_cov,
        ) in ekfac_full_data_model.covariances.items():
            batched_activations_cov = ekfac_batched_model.covariances.activations[
                layer_name
            ]
            batched_gradients_cov = ekfac_batched_model.covariances.gradients[
                layer_name
            ]

            eigenvalue_corrections = ekfac_full_data_model.eigenvalue_corrections[
                layer_name
            ]
            eigenvalue_corrections_batched = ekfac_batched_model.eigenvalue_corrections[
                layer_name
            ]

            assert np.allclose(
                activations_cov,
                batched_activations_cov,
                atol=1e-7,
            ), f"Activation covariances mismatch for layer {layer_name}"
            assert np.allclose(
                gradients_cov,
                batched_gradients_cov,
                atol=1e-7,
            ), f"Gradient covariances mismatch for layer {layer_name}"

            # Check eigenvalue corrections
            assert np.allclose(
                eigenvalue_corrections,
                eigenvalue_corrections_batched,
                atol=1e-7,
            ), f"Eigenvalue corrections mismatch for layer {layer_name}"

    def test_ekfac_ihvp_single_vector(self, trained_model: ModelTuple):
        """
        Test EKFAC IHVP computation for a single vector.

        Verifies that:
        1. H^{-1} @ v produces a valid output
        2. H @ (H^{-1} @ v) ≈ v (round-trip consistency)
        """
        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        # Setup EKFAC
        damping_lambda = 0.1
        ekfac_build_config = KFACBuildConfig(
            recalc_ekfac_components=True,
            use_pseudo_targets=True,
        )
        ekfac_config = KFACRunConfig(
            use_eigenvalue_correction=True,
            damping_lambda=damping_lambda,
        )

        ekfac_model = KFAC(
            config=KFACConfig(run_config=ekfac_config, build_config=ekfac_build_config)
        )

        # Generate EKFAC components
        x_train, y_train = dataset.get_train_data()

        # select a random data point of the training set to generate the test vector
        random_index_1 = 500
        x1_sample = jnp.asarray(x_train[random_index_1])

        # Generate a test vector from gradients
        test_vector = ekfac_model.generate_pseudo_targets(
            model=model,
            params=params,
            training_data=x1_sample,
            loss_fn=loss_fn,
        )

        # get gradients for the test vector
        test_vector = jax.grad(
            lambda p: loss_fn(model.apply(p, x1_sample), test_vector)
        )(params)
        test_vector, _ = flatten_util.ravel_pytree(test_vector)

        # for debugging:
        test_vector = jnp.zeros_like(test_vector).at[0].set(1.0)

        # Compute IHVP for EKFAC
        ihvp_ekfac = ekfac_model.compute_ihvp(
            model=model,
            params=params,
            training_data=jnp.asarray(x_train),
            training_targets=jnp.asarray(y_train),
            loss_fn=loss_fn,
            vector=test_vector,
        )

        hessian_ekfac_explicit = ekfac_model.compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(x_train),
            training_targets=jnp.asarray(y_train),
            loss_fn=loss_fn,
        )

        ekfac_lambda = ekfac_model.eigenvalue_corrections["output"] + damping_lambda

        hessian_inv_ekfac_explicit = ekfac_model.compute_inverse_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(x_train),
            training_targets=jnp.asarray(y_train),
            loss_fn=loss_fn,
        )

        # Compute IHVP for KFAC
        ekfac_model.config.run_config.use_eigenvalue_correction = False
        ihvp_kfac = ekfac_model.compute_ihvp(
            model=model,
            params=params,
            training_data=jnp.asarray(x_train),
            training_targets=jnp.asarray(y_train),
            loss_fn=loss_fn,
            vector=test_vector,
        )

        hessian_kfac_explicit = ekfac_model.compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(x_train),
            training_targets=jnp.asarray(y_train),
            loss_fn=loss_fn,
        )

        hessian_inv_kfac_explicit = ekfac_model.compute_inverse_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(x_train),
            training_targets=jnp.asarray(y_train),
            loss_fn=loss_fn,
        )

        kfac_lambda = (
            ekfac_model._compute_eigenvalue_lambda_kfac("output") + damping_lambda
        )

        true_hessian = Hessian().compute_hessian(
            model=model,
            params=params,
            training_data=jnp.asarray(x_train),
            training_targets=jnp.asarray(y_train),
            loss_fn=loss_fn,
        ) + ekfac_model.damping() * jnp.eye(test_vector.shape[0], test_vector.shape[0])

        true_hessian = np.array(true_hessian)
        true_hessian_ihvp = jnp.linalg.solve(
            true_hessian,
            test_vector,
        )
        true_hessian_hvp = (true_hessian) @ true_hessian_ihvp

        # calculate ihvp for each unit vector basis to get the full inverse hessian matrix for ekfac and kfac
        dim = test_vector.shape[0]
        ihvp_ekfac_full = []
        ihvp_kfac_full = []

        hvp_efkac_full = []
        hvp_kfac_full = []

        for i in range(dim):
            unit_vector = jnp.zeros_like(test_vector).at[i].set(1.0)

            ihvp_ekfac_i = ekfac_model.compute_ihvp(
                model=model,
                params=params,
                training_data=jnp.asarray(x_train),
                training_targets=jnp.asarray(y_train),
                loss_fn=loss_fn,
                vector=unit_vector,
            )
            ihvp_ekfac_full.append(ihvp_ekfac_i)

            hvp_ekfac_i = ekfac_model.compute_hvp(
                model=model,
                params=params,
                training_data=jnp.asarray(x_train),
                training_targets=jnp.asarray(y_train),
                loss_fn=loss_fn,
                vector=unit_vector,
            )
            hvp_efkac_full.append(hvp_ekfac_i)

            ekfac_model.config.run_config.use_eigenvalue_correction = False
            ihvp_kfac_i = ekfac_model.compute_ihvp(
                model=model,
                params=params,
                training_data=jnp.asarray(x_train),
                training_targets=jnp.asarray(y_train),
                loss_fn=loss_fn,
                vector=unit_vector,
            )
            ihvp_kfac_full.append(ihvp_kfac_i)

            hvp_kfac_i = ekfac_model.compute_hvp(
                model=model,
                params=params,
                training_data=jnp.asarray(x_train),
                training_targets=jnp.asarray(y_train),
                loss_fn=loss_fn,
                vector=unit_vector,
            )
            hvp_kfac_full.append(hvp_kfac_i)

        hessian_inv_ekfac_implicit = jnp.column_stack(ihvp_ekfac_full)
        hessian_inv_kfac_implicit = jnp.column_stack(ihvp_kfac_full)

        hessian_ekfac_implicit = jnp.column_stack(hvp_efkac_full)
        hessian_kfac_implicit = jnp.column_stack(hvp_kfac_full)

        # # plot lambdas for ekfac and kfac to compare as lineplot by flattening them in a single plot
        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(10, 5))
        # plt.plot(ekfac_lambda.flatten(), label="E-KFAC Lambda")
        # plt.plot(kfac_lambda.flatten(), label="KFAC Lambda")
        # plt.title("Eigenvalue Corrections (Lambda) Comparison")
        # plt.xlabel("Index")
        # plt.ylabel("Lambda Value")
        # plt.legend()
        # plt.show()

        # # and their difference
        # plt.figure(figsize=(5, 5))
        # plt.title("Difference Lambda (E-KFAC - KFAC)")
        # plt.imshow(ekfac_lambda - kfac_lambda, cmap="viridis", aspect="auto")
        # plt.colorbar()
        # plt.show()

        # # plot heatmap of full hessian matrix for ekfac and kfac computed via explicit and implicit methods
        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 3, 1)
        # plt.title("H (E-KFAC) Explicit")
        # plt.imshow(hessian_ekfac_explicit, cmap="viridis", aspect="auto")
        # plt.colorbar()
        # plt.subplot(1, 3, 2)
        # plt.title("H (E-KFAC) Implicit")
        # plt.imshow(hessian_ekfac_implicit, cmap="viridis", aspect="auto")
        # plt.colorbar()
        # plt.subplot(1, 3, 3)
        # plt.title("H (True)")
        # plt.imshow(true_hessian, cmap="viridis", aspect="auto")
        # plt.colorbar()
        # plt.show()

        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 3, 1)
        # plt.title("H (KFAC) Explicit")
        # plt.imshow(hessian_kfac_explicit, cmap="viridis", aspect="auto")
        # plt.colorbar()
        # plt.subplot(1, 3, 2)
        # plt.title("H (KFAC) Implicit")
        # plt.imshow(hessian_kfac_implicit, cmap="viridis", aspect="auto")
        # plt.colorbar()
        # plt.subplot(1, 3, 3)
        # plt.title("H (True)")
        # plt.imshow(true_hessian, cmap="viridis", aspect="auto")
        # plt.colorbar()
        # plt.show()

        # plot heatmap of difference between hessian computed via explicit and via implicit methods
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Difference H (E-KFAC)")
        plt.imshow(
            hessian_ekfac_explicit - hessian_ekfac_implicit,
            cmap="viridis",
            aspect="auto",
        )
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title("Difference H (KFAC)")
        plt.imshow(
            hessian_kfac_explicit - hessian_kfac_implicit,
            cmap="viridis",
            aspect="auto",
        )
        plt.colorbar()
        plt.suptitle(
            "Difference between Explicit and Implicit Hessian \n (H (Explicit) - H (Implicit))",
        )
        plt.tight_layout(rect=(0, 0, 1, 0.95))  # reserve top 5% for title
        plt.show()

        # plot H_implicit @ H_implicit^-1 for ekfac and kfac
        hessian_check_ekfac_implicit = (
            hessian_ekfac_implicit @ hessian_inv_ekfac_implicit
        )
        hessian_check_kfac_implicit = hessian_kfac_implicit @ hessian_inv_kfac_implicit
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("H @ H^{-1} (E-KFAC) (both implicit)")
        plt.imshow(hessian_check_ekfac_implicit, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title("H @ H^{-1} (KFAC) (both implicit)")
        plt.imshow(hessian_check_kfac_implicit, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.show()

        # plot heatmap of hessian inverse computed via ihvp and via direct inverse
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("H^{-1} Implicit (KFAC)")
        plt.imshow(hessian_inv_kfac_implicit, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title("H^{-1} Explicit (KFAC)")
        plt.imshow(hessian_inv_kfac_explicit, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("H^{-1} Implicit (E-KFAC)")
        plt.imshow(hessian_inv_ekfac_implicit, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title("H^{-1} Explicit (E-KFAC)")
        plt.imshow(hessian_inv_ekfac_explicit, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.show()

        # plot heatmap of difference between hessian inverse computed via ihvp and via direct inverse
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Difference H^{-1} (KFAC)")
        plt.imshow(
            hessian_inv_kfac_implicit - hessian_inv_kfac_explicit,
            cmap="viridis",
            aspect="auto",
        )
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title("Difference H^{-1} (E-KFAC)")
        plt.imshow(
            hessian_inv_ekfac_implicit - hessian_inv_ekfac_explicit,
            cmap="viridis",
            aspect="auto",
        )
        plt.colorbar()
        plt.suptitle(
            "Difference between Implicit and Explicit Inverse Hessian \n (H^{-1} (Implicit) - H^{-1} (Explicit))",
        )
        plt.tight_layout(rect=(0, 0, 1, 0.95))  # reserve top 5% for title
        plt.show()

        # plot heatmap of full inverse hessian matrix for ekfac and kfac

        hessian_check_ekfac = hessian_ekfac_explicit @ hessian_inv_ekfac_implicit
        hessian_check_kfac = hessian_kfac_explicit @ hessian_inv_kfac_implicit

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("H @ H^{-1} (E-KFAC) (implicit)")
        plt.imshow(hessian_check_ekfac, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title("H @ H^{-1} (KFAC) (implicit)")
        plt.imshow(hessian_check_kfac, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.show()

        # Basic sanity checks
        assert ihvp_ekfac.shape == test_vector.shape, (
            f"IHVP shape {ihvp_ekfac.shape} doesn't match input vector shape {test_vector.shape}"
        )
        assert jnp.isfinite(ihvp_ekfac).all(), "IHVP contains non-finite values"

        assert ihvp_kfac.shape == test_vector.shape, (
            f"IHVP shape {ihvp_kfac.shape} doesn't match input vector shape {test_vector.shape}"
        )
        assert jnp.isfinite(ihvp_kfac).all(), "IHVP contains non-finite values"

        # Verify round-trip: H @ (H^{-1} @ v) ≈ v
        hvp_ekfac = hessian_ekfac_explicit @ ihvp_ekfac
        hvp_kfac = hessian_kfac_explicit @ ihvp_kfac

        assert jnp.allclose(hvp_ekfac, test_vector, rtol=0.2, atol=1), (
            f"Round-trip test failed: H @ (H^{{-1}} @ v) != v\n"
            f"Max absolute error: {jnp.max(jnp.abs(hvp_ekfac - test_vector))}\n"
            f"Relative error: {jnp.linalg.norm(hvp_ekfac - test_vector) / jnp.linalg.norm(test_vector)}"
        )

        assert jnp.allclose(hvp_kfac, test_vector, rtol=0.2, atol=1), (
            f"Round-trip test failed: H @ (H^{{-1}} @ v) != v\n"
            f"Max absolute error: {jnp.max(jnp.abs(hvp_kfac - test_vector))}\n"
            f"Relative error: {jnp.linalg.norm(hvp_kfac - test_vector) / jnp.linalg.norm(test_vector)}"
        )

    def test_ekfac_ihvp_batched_vectors(self, trained_model: ModelTuple):
        """
        Test EKFAC IHVP computation for multiple vectors in batch.

        Verifies that:
        1. Batched IHVP handles multiple vectors correctly
        2. Batched computation matches single-vector computation
        3. H @ (H^{-1} @ V) ≈ V for batch of vectors
        """
        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        # Setup EKFAC
        damping_lambda = 0.1
        ekfac_build_config = KFACBuildConfig(
            recalc_ekfac_components=True,
            use_pseudo_targets=True,
        )
        ekfac_config = KFACRunConfig(
            use_eigenvalue_correction=True,
            damping_lambda=damping_lambda,
        )

        ekfac_model = KFAC(
            config=KFACConfig(run_config=ekfac_config, build_config=ekfac_build_config)
        )

        # Generate EKFAC components
        x_train, y_train = dataset.get_train_data()
        ekfac_model.get_ekfac_components(
            model=model,
            params=params,
            training_data=jnp.asarray(x_train),
            training_targets=jnp.asarray(y_train),
            loss_fn=loss_fn,
        )

        # Generate multiple test vectors from gradients
        n_vectors = 5
        pseudo_targets = ekfac_model.generate_pseudo_targets(
            model=model,
            params=params,
            training_data=jnp.asarray(x_train[:n_vectors]),
            loss_fn=loss_fn,
        )

        # Compute gradients for each example
        gradient_vecs = []
        for i in range(n_vectors):
            grad = jax.grad(
                lambda p: loss_fn(model.apply(p, x_train[i]), pseudo_targets[i])
            )(params)
            gradient_vecs.append(flatten_util.ravel_pytree(grad)[0])

        test_vectors = jnp.stack(gradient_vecs)

        # Compute batched IHVP
        ihvp_batched = ekfac_model.compute_ihvp(
            model=model,
            params=params,
            training_data=jnp.asarray(x_train),
            training_targets=jnp.asarray(y_train),
            loss_fn=loss_fn,
            vector=test_vectors,
        )

        ekfac_model.config.run_config.use_eigenvalue_correction = False
        ihvp_batched_kfac = ekfac_model.compute_ihvp(
            model=model,
            params=params,
            training_data=jnp.asarray(x_train),
            training_targets=jnp.asarray(y_train),
            loss_fn=loss_fn,
            vector=test_vectors,
        )

        # Basic sanity checks
        assert ihvp_batched.shape == test_vectors.shape, (
            f"Batched IHVP shape {ihvp_batched.shape} doesn't match input shape {test_vectors.shape}"
        )
        assert jnp.isfinite(ihvp_batched).all(), (
            "Batched IHVP contains non-finite values"
        )

        assert ihvp_batched_kfac.shape == test_vectors.shape, (
            f"Batched IHVP shape {ihvp_batched_kfac.shape} doesn't match input shape {test_vectors.shape}"
        )
        assert jnp.isfinite(ihvp_batched_kfac).all(), (
            "Batched IHVP contains non-finite values"
        )

        # Verify batched computation matches single-vector computation
        for i in range(n_vectors):
            ekfac_model.config.run_config.use_eigenvalue_correction = True
            ihvp_single = ekfac_model.compute_ihvp(
                model=model,
                params=params,
                training_data=jnp.asarray(x_train),
                training_targets=jnp.asarray(y_train),
                loss_fn=loss_fn,
                vector=test_vectors[i],
            )
            assert jnp.allclose(ihvp_batched[i], ihvp_single, rtol=1e-6, atol=1e-4), (
                f"Batched IHVP doesn't match single-vector IHVP for vector {i} (EKFAC)"
            )

            ekfac_model.config.run_config.use_eigenvalue_correction = False
            ihvp_single_kfac = ekfac_model.compute_ihvp(
                model=model,
                params=params,
                training_data=jnp.asarray(x_train),
                training_targets=jnp.asarray(y_train),
                loss_fn=loss_fn,
                vector=test_vectors[i],
            )
            assert jnp.allclose(
                ihvp_batched_kfac[i], ihvp_single_kfac, rtol=1e-6, atol=1e-4
            ), f"Batched IHVP doesn't match single-vector IHVP for vector {i} (KFAC)"
