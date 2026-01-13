import tempfile
from pathlib import Path

import pytest
from jax import numpy as jnp

from src.config import DatasetEnum
from src.utils.data.data import (
    Dataset,
    DownloadableDataset,
)

# All downloadable datasets to test
DOWNLOADABLE_DATASETS = [
    DatasetEnum.ENERGY,
    DatasetEnum.CONCRETE,
    DatasetEnum.CANCER,
    DatasetEnum.DIGITS,
    DatasetEnum.SKLEARN_DIGITS,
    DatasetEnum.MNIST,
    DatasetEnum.FASHION_MNIST,
    DatasetEnum.CIFAR10,
]


class TestDatasetDownloadAndCache:
    """Test suite for downloadable datasets."""

    @pytest.mark.parametrize("dataset_enum", DOWNLOADABLE_DATASETS)
    def test_download(self, dataset_enum: DatasetEnum):
        """Test that dataset can be downloaded."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=dataset_enum, directory=None, store_on_disk=False
        )

        # Check that dataset has data
        assert len(dataset) > 0, f"{dataset_enum.value} has no samples"
        assert dataset.inputs.shape[0] > 0, f"{dataset_enum.value} has no inputs"
        assert dataset.targets.shape[0] > 0, f"{dataset_enum.value} has no targets"

        # Check that inputs and targets have same number of samples
        assert dataset.inputs.shape[0] == dataset.targets.shape[0], (
            f"{dataset_enum.value}:  Mismatch between inputs "
            f"({dataset.inputs.shape[0]}) and targets ({dataset.targets.shape[0]})"
        )

        # Check dimensions are positive
        assert dataset.input_dim() > 0, f"{dataset_enum.value} has invalid input_dim"
        assert dataset.output_dim() > 0, f"{dataset_enum.value} has invalid output_dim"

    @pytest.mark.parametrize("dataset_enum", DOWNLOADABLE_DATASETS)
    def test_save_and_load(self, dataset_enum: DatasetEnum):
        """Test that dataset can be saved to disk and reloaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / dataset_enum.value

            # First load:  download and save to disk
            dataset1: Dataset = DownloadableDataset.load(
                dataset=dataset_enum, directory=str(cache_dir), store_on_disk=True
            )

            # Check that files were created
            assert cache_dir.exists(), (
                f"Cache directory not created for {dataset_enum.value}"
            )
            assert (cache_dir / "X.csv").exists(), (
                f"X.csv not saved for {dataset_enum.value}"
            )
            assert (cache_dir / "Y.csv").exists(), (
                f"Y. csv not saved for {dataset_enum.value}"
            )

            # Second load: load from disk
            dataset2: Dataset = DownloadableDataset.load(
                dataset=dataset_enum, directory=str(cache_dir), store_on_disk=False
            )

            # Check that datasets are identical
            assert dataset1.inputs.shape == dataset2.inputs.shape, (
                f"{dataset_enum.value}: Input shapes don't match after reload"
            )
            assert dataset1.targets.shape == dataset2.targets.shape, (
                f"{dataset_enum.value}:  Target shapes don't match after reload"
            )
            assert jnp.allclose(dataset1.inputs, dataset2.inputs, rtol=1e-5), (
                f"{dataset_enum.value}: Input values don't match after reload"
            )
            assert jnp.allclose(dataset1.targets, dataset2.targets, rtol=1e-5), (
                f"{dataset_enum.value}:  Target values don't match after reload"
            )

    @pytest.mark.parametrize("dataset_enum", DOWNLOADABLE_DATASETS)
    def test_dataloader(self, dataset_enum: DatasetEnum):
        """Test that dataloader can be created and works correctly."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=dataset_enum, directory=None, store_on_disk=False
        )

        # Test with batch size
        batch_size = min(32, len(dataset))
        dataloader = dataset.get_dataloader(
            batch_size=batch_size, shuffle=True, seed=42
        )

        # Get first batch
        batch = next(iter(dataloader))
        batch_inputs, batch_targets = batch

        # Check batch shapes
        assert batch_inputs.shape[0] <= batch_size, (
            f"{dataset_enum.value}:  Batch size exceeds requested size"
        )
        assert batch_inputs.shape[1] == dataset.input_dim(), (
            f"{dataset_enum.value}:  Batch input dimension doesn't match dataset"
        )

        # Test without batch size (full dataset)
        dataloader_full = dataset.get_dataloader(
            batch_size=len(dataset), shuffle=False, seed=42
        )
        full_batch = next(iter(dataloader_full))
        full_inputs, full_targets = full_batch

        assert full_inputs.shape[0] == len(dataset), (
            f"{dataset_enum.value}:  Full batch doesn't contain all samples"
        )


class TestDatasetTypes:
    """Test specific properties of different dataset types."""

    def test_regression_datasets_have_float_targets(self):
        """Test that regression datasets (UCI) have float targets."""
        regression_datasets = [DatasetEnum.ENERGY, DatasetEnum.CONCRETE]

        for dataset_enum in regression_datasets:
            dataset = DownloadableDataset.load(
                dataset=dataset_enum, directory=None, store_on_disk=False
            )
            assert dataset.targets.dtype == jnp.float32, (
                f"{dataset_enum.value} should have float32 targets for regression"
            )

    def test_classification_datasets_have_int_targets(self):
        """Test that classification datasets have integer targets."""
        classification_datasets = [
            DatasetEnum.CANCER,
            DatasetEnum.DIGITS,
            DatasetEnum.SKLEARN_DIGITS,
            DatasetEnum.MNIST,
            DatasetEnum.FASHION_MNIST,
            DatasetEnum.CIFAR10,
        ]

        for dataset_enum in classification_datasets:
            dataset = DownloadableDataset.load(
                dataset=dataset_enum, directory=None, store_on_disk=False
            )
            assert dataset.targets.dtype == jnp.int32, (
                f"{dataset_enum.value} should have int32 targets for classification"
            )

    def test_image_datasets_normalized(self):
        """Test that image datasets have normalized pixel values in [0, 1]."""
        image_datasets = [
            DatasetEnum.DIGITS,
            DatasetEnum.SKLEARN_DIGITS,
            DatasetEnum.MNIST,
            DatasetEnum.FASHION_MNIST,
            DatasetEnum.CIFAR10,
        ]

        for dataset_enum in image_datasets:
            dataset = DownloadableDataset.load(
                dataset=dataset_enum, directory=None, store_on_disk=False
            )
            assert jnp.min(dataset.inputs) >= 0.0, (
                f"{dataset_enum.value}:  Image values should be >= 0"
            )
            assert jnp.max(dataset.inputs) <= 1.0, (
                f"{dataset_enum.value}: Image values should be <= 1"
            )

    def test_mnist_dimensions(self):
        """Test MNIST has correct flattened dimensions."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=DatasetEnum.MNIST, directory=None, store_on_disk=False
        )
        assert dataset.input_dim() == 784, "MNIST should have 784 features (28x28)"
        assert dataset.output_dim() == 10, "MNIST should have 10 classes"

    def test_fashion_mnist_dimensions(self):
        """Test Fashion-MNIST has correct flattened dimensions."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=DatasetEnum.FASHION_MNIST, directory=None, store_on_disk=False
        )
        assert dataset.input_dim() == 784, (
            "Fashion-MNIST should have 784 features (28x28)"
        )
        assert dataset.output_dim() == 10, "Fashion-MNIST should have 10 classes"

    def test_cifar10_dimensions(self):
        """Test CIFAR-10 has correct flattened dimensions."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=DatasetEnum.CIFAR10, directory=None, store_on_disk=False
        )
        assert dataset.input_dim() == 3072, (
            "CIFAR-10 should have 3072 features (32x32x3)"
        )
        assert dataset.output_dim() == 10, "CIFAR-10 should have 10 classes"

    def test_digits_dimensions(self):
        """Test UCI Digits has correct dimensions."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=DatasetEnum.DIGITS, directory=None, store_on_disk=False
        )
        assert dataset.input_dim() == 64, "UCI Digits should have 64 features (8x8)"
        assert dataset.output_dim() == 10, "UCI Digits should have 10 classes"

    def test_sklearn_digits_dimensions(self):
        """Test sklearn Digits has correct dimensions."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=DatasetEnum.SKLEARN_DIGITS, directory=None, store_on_disk=False
        )
        assert dataset.input_dim() == 64, "sklearn Digits should have 64 features (8x8)"
        assert dataset.output_dim() == 10, "sklearn Digits should have 10 classes"

    def test_energy_dimensions(self):
        """Test Energy dataset has correct dimensions."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=DatasetEnum.ENERGY, directory=None, store_on_disk=False
        )
        assert dataset.input_dim() == 8, "Energy should have 8 features"
        assert dataset.output_dim() == 2, "Energy should have 2 target variables"

    def test_concrete_dimensions(self):
        """Test Concrete dataset has correct dimensions."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=DatasetEnum.CONCRETE, directory=None, store_on_disk=False
        )
        assert dataset.input_dim() == 8, "Concrete should have 8 features"
        assert dataset.output_dim() == 1, "Concrete should have 1 target variable"

    def test_cancer_dimensions(self):
        """Test Cancer dataset has correct dimensions."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=DatasetEnum.CANCER, directory=None, store_on_disk=False
        )
        assert dataset.input_dim() == 30, "Cancer should have 30 features"
        assert dataset.output_dim() == 2, "Cancer should have 2 classes"


class TestDatasetReplaceTargets:
    """Test the replace_targets functionality."""

    @pytest.mark.parametrize("dataset_enum", DOWNLOADABLE_DATASETS)
    def test_replace_targets(self, dataset_enum: DatasetEnum):
        """Test that targets can be replaced for all datasets."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=dataset_enum, directory=None, store_on_disk=False
        )

        # Create new targets with same shape
        new_targets = jnp.ones_like(dataset.targets)
        new_dataset = dataset.replace_targets(new_targets)

        # Check that targets were replaced
        assert jnp.allclose(new_dataset.targets, new_targets), (
            f"{dataset_enum.value}: Targets not replaced correctly"
        )

        # Check that inputs remain the same
        assert jnp.allclose(new_dataset.inputs, dataset.inputs), (
            f"{dataset_enum.value}: Inputs should not change when replacing targets"
        )

        # Check that shape is preserved
        assert new_dataset.targets.shape == dataset.targets.shape, (
            f"{dataset_enum.value}: Target shape changed after replacement"
        )

    def test_replace_targets_shape_mismatch(self):
        """Test that replacing targets with wrong shape raises error."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=DatasetEnum.CANCER, directory=None, store_on_disk=False
        )

        # Try to replace with wrong shape - should fail
        wrong_shape_targets = jnp.ones((len(dataset) + 10, dataset.output_dim()))

        with pytest.raises((ValueError, AssertionError)):
            dataset.replace_targets(wrong_shape_targets)


class TestDatasetSplit:
    """Test dataset splitting functionality."""

    @pytest.mark.parametrize("dataset_enum", DOWNLOADABLE_DATASETS)
    def test_train_test_split(self, dataset_enum: DatasetEnum):
        """Test that train/test split works correctly."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=dataset_enum, directory=None, store_on_disk=False
        )

        test_size = 0.2
        seed = 42

        train_ds, test_ds = dataset.train_test_split(test_size=test_size, seed=seed)

        # Check sizes (approximate due to class balancing in classification)
        total_size = len(train_ds) + len(test_ds)
        assert total_size == len(dataset), (
            f"{dataset_enum.value}:  Train + test size doesn't equal original size"
        )

        # Check that dimensions are preserved
        assert train_ds.input_dim() == dataset.input_dim(), (
            f"{dataset_enum.value}:  Train set input_dim doesn't match"
        )
        assert test_ds.input_dim() == dataset.input_dim(), (
            f"{dataset_enum.value}: Test set input_dim doesn't match"
        )
        assert train_ds.output_dim() == dataset.output_dim(), (
            f"{dataset_enum.value}: Train set output_dim doesn't match"
        )
        assert test_ds.output_dim() == dataset.output_dim(), (
            f"{dataset_enum.value}: Test set output_dim doesn't match"
        )

        # Check reproducibility
        train_ds2, test_ds2 = dataset.train_test_split(test_size=test_size, seed=seed)
        assert jnp.allclose(train_ds.inputs, train_ds2.inputs), (
            f"{dataset_enum.value}: Split not reproducible"
        )

    @pytest.mark.parametrize("dataset_enum", DOWNLOADABLE_DATASETS)
    def test_k_fold_splits(self, dataset_enum: DatasetEnum):
        """Test that k-fold cross-validation splits work correctly."""
        dataset: Dataset = DownloadableDataset.load(
            dataset=dataset_enum, directory=None, store_on_disk=False
        )

        n_splits = 5
        seed = 42

        splits = list(
            dataset.get_k_fold_splits(n_splits=n_splits, shuffle=True, seed=seed)
        )

        # Check we got correct number of splits
        assert len(splits) == n_splits, (
            f"{dataset_enum.value}: Expected {n_splits} splits, got {len(splits)}"
        )

        # Check that each split has train and test data
        for i, ((train_X, train_Y), (test_X, test_Y)) in enumerate(splits):
            assert train_X.shape[0] > 0, (
                f"{dataset_enum.value}:  Fold {i} has empty train set"
            )
            assert test_X.shape[0] > 0, (
                f"{dataset_enum.value}: Fold {i} has empty test set"
            )

            # Check dimensions match
            assert train_X.shape[1] == dataset.input_dim(), (
                f"{dataset_enum.value}: Fold {i} train input_dim doesn't match"
            )
            assert test_X.shape[1] == dataset.input_dim(), (
                f"{dataset_enum.value}: Fold {i} test input_dim doesn't match"
            )

            # Check that train + test = total dataset size
            total = train_X.shape[0] + test_X.shape[0]
            assert total == len(dataset), (
                f"{dataset_enum.value}: Fold {i} train+test size doesn't match dataset size"
            )
