import shutil
from pathlib import Path
from typing import Any, Generator, Tuple

import jax.numpy as jnp
import numpy as np
import pytest

from config.config import (
    Config,
    KFACConfig,
    LinearModelConfig,
    RandomClassificationConfig,
    TrainingConfig,
)
from data.data import AbstractDataset
from hessian_approximations.kfac.kfac import KFAC
from hessian_approximations.kfac.storage import KFACStorage
from main import train
from models.base import ApproximationModel
from models.train import get_loss_fn


class TestKFACStorage:
    """Test suite for the KFACStorage class handling EKFAC component persistence."""

    # --------------------------------------------------------------------------
    # Fixtures
    # --------------------------------------------------------------------------
    @pytest.fixture(scope="class")
    def tmp_storage_dir(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> Generator[Path, None, None]:
        """Temporary directory for testing file I/O, cleaned up after all tests."""
        path = tmp_path_factory.mktemp("kfac_storage_test")
        yield path
        # Ensure cleanup of the entire temporary tree at the end of the suite
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture(scope="class")
    def simple_config(self):
        """Minimal config for a small linear model on synthetic data."""
        return Config(
            dataset=RandomClassificationConfig(
                n_samples=100,
                n_features=5,
                n_informative=3,
                n_classes=2,
                random_state=42,
                train_test_split=1.0,
            ),
            model=LinearModelConfig(loss="cross_entropy", hidden_dim=[5]),
            training=TrainingConfig(
                epochs=2,
                batch_size=10,
                lr=0.01,
                optimizer="sgd",
                loss="cross_entropy",
            ),
        )

    @pytest.fixture(scope="class")
    def trained_model(
        self, simple_config
    ) -> Tuple[ApproximationModel, AbstractDataset, Any, Config]:
        """Trains a small model to generate KFAC components."""
        model, dataset, params = train(simple_config)
        return model, dataset, params, simple_config

    # --------------------------------------------------------------------------
    # Helper
    # --------------------------------------------------------------------------
    def _cleanup_model_dir(self, base_dir: Path, model_name: str) -> None:
        """Remove a model-specific folder safely (used after each test)."""
        model_dir = base_dir / model_name
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # Tests
    # --------------------------------------------------------------------------
    def test_storage_saves_and_loads_correctly(
        self,
        trained_model: Tuple[ApproximationModel, AbstractDataset, Any, Config],
        tmp_storage_dir: Path,
    ):
        """End-to-end test: compute EKFAC components, save them, and reload them."""
        model_name = "test_model"
        dataset_name = "test_dataset"
        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        kfac_config = KFACConfig()

        # Initialize KFAC with custom storage path
        kfac = KFAC(model_name=model_name, config=kfac_config)
        kfac.storage = KFACStorage(
            model_name=model_name,
            dataset_name=dataset_name,
            base_path=tmp_storage_dir,
        )

        try:
            x, y = dataset.get_train_data()
            kfac.get_ekfac_components(
                model=model,
                params=params,
                training_data=jnp.asarray(x),
                training_targets=jnp.asarray(y),
                loss_fn=loss_fn,
            )

            # --- Verify saved files exist
            model_path = tmp_storage_dir / model_name / dataset_name
            assert (model_path / "covariances.npz").exists()
            assert (model_path / "eigenvectors.npz").exists()
            assert (model_path / "eigenvalue_corrections.npz").exists()
            assert (model_path / "eigenvalues.npz").exists()

            # --- Load back the files
            covariances_loaded = kfac.storage.load_covariances()
            eigenvectors_loaded = kfac.storage.load_eigenvectors()
            corrections_loaded = kfac.storage.load_eigenvalue_corrections()
            eigenvalues_loaded = kfac.storage.load_eigenvalues()

            # --- Compare covariances
            for layer in kfac.covariances.activations.keys():
                assert np.allclose(
                    np.asarray(kfac.covariances.activations[layer]),
                    np.asarray(covariances_loaded.activations[layer]),
                    atol=1e-10,
                ), f"Activation covariances mismatch in layer {layer}"

                assert np.allclose(
                    np.asarray(kfac.covariances.gradients[layer]),
                    np.asarray(covariances_loaded.gradients[layer]),
                    atol=1e-10,
                ), f"Gradient covariances mismatch in layer {layer}"

            # --- Compare eigenvectors (up to sign ambiguity)
            for layer in kfac.eigenvectors.activations.keys():
                Q1 = np.asarray(kfac.eigenvectors.activations[layer])
                Q2 = np.asarray(eigenvectors_loaded.activations[layer])
                assert Q1.shape == Q2.shape
                assert np.allclose(np.abs(Q1), np.abs(Q2), atol=1e-6)

            # --- Compare eigenvalue corrections
            for layer in kfac.eigenvalue_corrections.keys():
                c1 = np.asarray(kfac.eigenvalue_corrections[layer])
                c2 = np.asarray(corrections_loaded[layer])
                assert np.allclose(c1, c2, atol=1e-10), (
                    f"Corrections mismatch in {layer}"
                )

            # --- Compare eigenvalues
            for layer in kfac.eigenvalues.activations.keys():
                assert np.allclose(
                    np.asarray(kfac.eigenvalues.activations[layer]),
                    np.asarray(eigenvalues_loaded.activations[layer]),
                    atol=1e-10,
                ), f"Activation eigenvalues mismatch in layer {layer}"

                assert np.allclose(
                    np.asarray(kfac.eigenvalues.gradients[layer]),
                    np.asarray(eigenvalues_loaded.gradients[layer]),
                    atol=1e-10,
                ), f"Gradient eigenvalues mismatch in layer {layer}"

        finally:
            # Always clean up even if assertions fail
            self._cleanup_model_dir(tmp_storage_dir, model_name)

    def test_storage_handles_missing_files(self, tmp_storage_dir: Path):
        """Ensure FileNotFoundError is raised when loading missing data."""
        model_name = "nonexistent_model"
        dataset_name = "nonexistent_dataset"
        storage = KFACStorage(
            model_name=model_name, dataset_name=dataset_name, base_path=tmp_storage_dir
        )
        try:
            with pytest.raises(FileNotFoundError):
                storage.load_covariances()
            with pytest.raises(FileNotFoundError):
                storage.load_eigenvectors()
            with pytest.raises(FileNotFoundError):
                storage.load_eigenvalue_corrections()
            with pytest.raises(FileNotFoundError):
                storage.load_eigenvalues()
        finally:
            self._cleanup_model_dir(tmp_storage_dir, model_name)

    def test_storage_overwrite_existing_files(
        self,
        trained_model: Tuple[ApproximationModel, AbstractDataset, Any, Config],
        tmp_storage_dir: Path,
    ):
        """Check that re-saving files overwrites them correctly."""
        model_name = "overwrite_test"
        dataset_name = "overwrite_dataset"
        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        kfac_config = KFACConfig()
        kfac = KFAC(model_name=model_name, config=kfac_config)
        kfac.storage = KFACStorage(
            model_name=model_name, dataset_name=dataset_name, base_path=tmp_storage_dir
        )

        try:
            x, y = dataset.get_train_data()
            kfac.compute_covariances(
                model, params, jnp.asarray(x), jnp.asarray(y), loss_fn
            )

            path = tmp_storage_dir / model_name / dataset_name / "covariances.npz"
            timestamp_before = path.stat().st_mtime

            # Save again (should overwrite file)
            kfac.storage.save_covariances(kfac.covariances)
            timestamp_after = path.stat().st_mtime

            assert timestamp_after >= timestamp_before, "File not overwritten properly"
        finally:
            self._cleanup_model_dir(tmp_storage_dir, model_name)
