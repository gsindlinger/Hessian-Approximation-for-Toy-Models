import copy
import shutil
from pathlib import Path
from typing import Any, Generator, Tuple

import numpy as np
import pytest

from src.config.config import Config
from src.config.dataset_config import RandomClassificationConfig
from src.config.hessian_approximation_config import (
    KFACBuildConfig,
    KFACConfig,
)
from src.config.model_config import LinearModelConfig
from src.config.training_config import TrainingConfig
from src.data.data import AbstractDataset
from src.hessian_approximations.kfac.kfac_service import KFAC
from src.hessian_approximations.kfac.kfac_storage import KFACStorage
from src.models.base import ApproximationModel
from src.models.train import get_loss_fn, train_or_load


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
    def simple_config(self, tmp_storage_dir: Path):
        """Minimal config for a small linear model on synthetic data."""
        kfac_build_config = KFACBuildConfig(storage_dir=str(tmp_storage_dir))

        config = Config(
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
            hessian_approximation=KFACConfig(build_config=kfac_build_config),
        )

        return config

    @pytest.fixture(scope="class")
    def trained_model(
        self, simple_config
    ) -> Tuple[ApproximationModel, AbstractDataset, Any, Config]:
        """Trains a small model to generate KFAC components."""
        model_data = train_or_load(simple_config)
        return model_data.model, model_data.dataset, model_data.params, simple_config

    # --------------------------------------------------------------------------
    # Helper
    # --------------------------------------------------------------------------
    def _cleanup_storage_dir(self, storage: KFACStorage) -> None:
        """Remove the storage directory safely (used after each test)."""
        if storage.base_directory.exists():
            shutil.rmtree(storage.base_directory, ignore_errors=True)

    # --------------------------------------------------------------------------
    # Tests
    # --------------------------------------------------------------------------
    def test_storage_saves_and_loads_correctly(
        self,
        trained_model: Tuple[ApproximationModel, AbstractDataset, Any, Config],
        tmp_storage_dir: Path,
    ):
        """End-to-end test: compute EKFAC components, save them, and reload them."""
        model, dataset, params, config = trained_model

        # Create a copy of config with custom storage path

        test_config = copy.deepcopy(config)

        # Initialize KFAC with custom storage path
        kfac = KFAC(full_config=test_config)

        try:
            kfac.provider.get_ekfac_components()

            # --- Verify saved files exist
            storage_path = kfac.provider.storage.base_directory
            assert (storage_path / "covariances.npz").exists()
            assert (storage_path / "eigenvectors.npz").exists()
            assert (storage_path / "eigenvalue_corrections.npz").exists()
            assert (storage_path / "eigenvalues.npz").exists()
            assert (storage_path / "mean_eigenvalues_and_corrections.json").exists()

            # --- Load back the files
            covariances_loaded = kfac.provider.storage.load_covariances()
            eigenvectors_loaded = kfac.provider.storage.load_eigenvectors()
            corrections_loaded = kfac.provider.storage.load_eigenvalue_corrections()
            eigenvalues_loaded = kfac.provider.storage.load_eigenvalues()

            # --- Compare covariances
            for layer in kfac.provider.data.covariances.activations.keys():
                assert np.allclose(
                    np.asarray(kfac.provider.data.covariances.activations[layer]),
                    np.asarray(covariances_loaded.activations[layer]),
                    atol=1e-10,
                ), f"Activation covariances mismatch in layer {layer}"

                assert np.allclose(
                    np.asarray(kfac.provider.data.covariances.gradients[layer]),
                    np.asarray(covariances_loaded.gradients[layer]),
                    atol=1e-10,
                ), f"Gradient covariances mismatch in layer {layer}"

            # --- Compare eigenvectors (up to sign ambiguity)
            for layer in kfac.provider.data.eigenvectors.activations.keys():
                Q1 = np.asarray(kfac.provider.data.eigenvectors.activations[layer])
                Q2 = np.asarray(eigenvectors_loaded.activations[layer])
                assert Q1.shape == Q2.shape
                assert np.allclose(np.abs(Q1), np.abs(Q2), atol=1e-6)

            # --- Compare eigenvalue corrections
            for layer in kfac.provider.data.eigenvalue_corrections.keys():
                c1 = np.asarray(kfac.provider.data.eigenvalue_corrections[layer])
                c2 = np.asarray(corrections_loaded[layer])
                assert np.allclose(c1, c2, atol=1e-10), (
                    f"Corrections mismatch in {layer}"
                )

            # --- Compare eigenvalues
            for layer in kfac.provider.data.eigenvalues.activations.keys():
                assert np.allclose(
                    np.asarray(kfac.provider.data.eigenvalues.activations[layer]),
                    np.asarray(eigenvalues_loaded.activations[layer]),
                    atol=1e-10,
                ), f"Activation eigenvalues mismatch in layer {layer}"

                assert np.allclose(
                    np.asarray(kfac.provider.data.eigenvalues.gradients[layer]),
                    np.asarray(eigenvalues_loaded.gradients[layer]),
                    atol=1e-10,
                ), f"Gradient eigenvalues mismatch in layer {layer}"

        finally:
            # Always clean up even if assertions fail
            self._cleanup_storage_dir(kfac.provider.storage)

    def test_storage_handles_missing_files(self, simple_config):
        """Ensure FileNotFoundError is raised when loading missing data."""

        test_config = copy.deepcopy(simple_config)
        kfac_build_config = KFACBuildConfig(
            storage_dir=str("data/tests/test123/test123")
        )

        storage = KFACStorage(
            configs=(
                test_config.model,
                test_config.dataset,
                test_config.training,
                kfac_build_config,
            )
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
            with pytest.raises(FileNotFoundError):
                storage.load_mean_eigenvalues_and_corrections()
        finally:
            self._cleanup_storage_dir(storage)

    def test_storage_overwrite_existing_files(
        self,
        trained_model: Tuple[ApproximationModel, AbstractDataset, Any, Config],
    ):
        """Check that re-saving files overwrites them correctly."""
        model, dataset, params, config = trained_model
        loss_fn = get_loss_fn(config.model.loss)

        test_config = copy.deepcopy(config)

        assert isinstance(test_config.hessian_approximation, KFACConfig)

        kfac = KFAC(full_config=test_config)
        kfac.provider.storage = KFACStorage(
            configs=(
                test_config.model,
                test_config.dataset,
                test_config.training,
                test_config.hessian_approximation.build_config,
            ),
        )

        try:
            x, y = dataset.get_train_data()
            kfac.provider.compute_covariances()

            path = kfac.provider.storage.base_directory / "covariances.npz"
            timestamp_before = path.stat().st_mtime

            # Save again (should overwrite file)
            kfac.provider.storage.save_covariances(kfac.provider.data.covariances)
            timestamp_after = path.stat().st_mtime

            assert timestamp_after >= timestamp_before, "File not overwritten properly"
        finally:
            self._cleanup_storage_dir(kfac.provider.storage)

    def test_check_storage_with_matching_config(
        self,
        trained_model: Tuple[ApproximationModel, AbstractDataset, Any, Config],
        tmp_storage_dir: Path,
    ):
        """Test that check_storage returns True when config matches."""
        model, dataset, params, config = trained_model

        test_config = copy.deepcopy(config)

        assert isinstance(test_config.hessian_approximation, KFACConfig)

        kfac = KFAC(full_config=test_config)
        kfac.provider.storage = KFACStorage(
            configs=(
                test_config.model,
                test_config.dataset,
                test_config.training,
                test_config.hessian_approximation.build_config,
            )
        )

        try:
            kfac.provider.get_ekfac_components()

            # Check with same config should return True
            assert kfac.provider.storage.check_storage()

        finally:
            self._cleanup_storage_dir(kfac.provider.storage)

    def test_check_storage_with_different_config(
        self,
        trained_model: Tuple[ApproximationModel, AbstractDataset, Any, Config],
        tmp_storage_dir: Path,
    ):
        """Test that check_storage returns False when config differs."""
        model, dataset, params, config = trained_model

        test_config = copy.deepcopy(config)

        assert isinstance(test_config.hessian_approximation, KFACConfig)

        kfac = KFAC(full_config=test_config)
        kfac.provider.storage = KFACStorage(
            configs=(
                test_config.model,
                test_config.dataset,
                test_config.training,
                test_config.hessian_approximation.build_config,
            ),
        )

        try:
            kfac.provider.get_ekfac_components()

            # Create storage with different config
            modified_config = copy.deepcopy(test_config)
            modified_config.training.epochs = 999  # Different value

            assert isinstance(modified_config.hessian_approximation, KFACConfig)

            storage2 = KFACStorage(
                configs=(
                    modified_config.model,
                    modified_config.dataset,
                    modified_config.training,
                    modified_config.hessian_approximation.build_config,
                ),
            )

            # Check should return False due to config mismatch
            assert not storage2.check_storage()

        finally:
            self._cleanup_storage_dir(kfac.provider.storage)

    def test_delete_storage(
        self,
        trained_model: Tuple[ApproximationModel, AbstractDataset, Any, Config],
        tmp_storage_dir: Path,
    ):
        """Test that delete_storage removes all files and directory."""
        model, dataset, params, config = trained_model

        test_config = copy.deepcopy(config)

        assert isinstance(test_config.hessian_approximation, KFACConfig)

        kfac = KFAC(full_config=test_config)
        kfac.provider.storage = KFACStorage(
            configs=(
                test_config.model,
                test_config.dataset,
                test_config.training,
                test_config.hessian_approximation.build_config,
            ),
        )

        kfac.provider.get_ekfac_components()

        storage_path = kfac.provider.storage.base_directory
        assert storage_path.exists()

        # Delete storage
        kfac.provider.storage.delete_storage()
        # Verify directory and files are gone
        assert not storage_path.exists()
