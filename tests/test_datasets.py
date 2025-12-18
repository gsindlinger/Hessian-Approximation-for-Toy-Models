import tempfile
from pathlib import Path

import pytest
from jax import numpy as jnp

from src.utils.data.data import (
    CancerDataset,
    CIFAR10Dataset,
    ConcreteDataset,
    Dataset,
    DigitsDataset,
    EnergyDataset,
    FashionMNISTDataset,
    MNISTDataset,
)

# All downloadable datasets to test
DOWNLOADABLE_DATASETS = [
    EnergyDataset,
    ConcreteDataset,
    CancerDataset,
    DigitsDataset,
    MNISTDataset,
    FashionMNISTDataset,
    CIFAR10Dataset,
]


class TestDatasetDownloadAndCache:
    """Test suite for downloadable datasets."""

    @pytest.mark.parametrize("dataset_class", DOWNLOADABLE_DATASETS)
    def test_download(self, dataset_class):
        """Test that dataset can be downloaded."""
        dataset: Dataset = dataset_class.load(directory=None, store_on_disk=False)

        # Check that dataset has data
        assert len(dataset) > 0, f"{dataset_class.__name__} has no samples"
        assert dataset.inputs.shape[0] > 0, f"{dataset_class.__name__} has no inputs"
        assert dataset.targets.shape[0] > 0, f"{dataset_class.__name__} has no targets"

        # Check that inputs and targets have same number of samples
        assert dataset.inputs.shape[0] == dataset.targets.shape[0], (
            f"{dataset_class.__name__}: Mismatch between inputs "
            f"({dataset.inputs.shape[0]}) and targets ({dataset.targets.shape[0]})"
        )

        # Check dimensions are positive
        assert dataset.input_dim() > 0, (
            f"{dataset_class.__name__} has invalid input_dim"
        )
        assert dataset.output_dim() > 0, (
            f"{dataset_class.__name__} has invalid output_dim"
        )

    @pytest.mark.parametrize("dataset_class", DOWNLOADABLE_DATASETS)
    def test_save_and_load(self, dataset_class):
        """Test that dataset can be saved to disk and reloaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / dataset_class.__name__

            # First load: download and save to disk
            dataset1: Dataset = dataset_class.load(
                directory=str(cache_dir), store_on_disk=True
            )

            # Check that files were created
            assert cache_dir.exists(), (
                f"Cache directory not created for {dataset_class.__name__}"
            )
            assert (cache_dir / "X.csv").exists(), (
                f"X.csv not saved for {dataset_class.__name__}"
            )
            assert (cache_dir / "Y.csv").exists(), (
                f"Y.csv not saved for {dataset_class.__name__}"
            )

            # Second load: load from disk
            dataset2: Dataset = dataset_class.load(directory=str(cache_dir))

            # Check that datasets are identical
            assert dataset1.inputs.shape == dataset2.inputs.shape, (
                f"{dataset_class.__name__}: Input shapes don't match after reload"
            )
            assert dataset1.targets.shape == dataset2.targets.shape, (
                f"{dataset_class.__name__}: Target shapes don't match after reload"
            )
            assert jnp.allclose(dataset1.inputs, dataset2.inputs, rtol=1e-5), (
                f"{dataset_class.__name__}: Input values don't match after reload"
            )
            assert jnp.allclose(dataset1.targets, dataset2.targets, rtol=1e-5), (
                f"{dataset_class.__name__}: Target values don't match after reload"
            )

    @pytest.mark.parametrize("dataset_class", DOWNLOADABLE_DATASETS)
    def test_dataloader(self, dataset_class):
        """Test that dataloader can be created and works correctly."""
        dataset: Dataset = dataset_class.load(directory=None, store_on_disk=False)

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
            f"{dataset_class.__name__}: Batch size exceeds requested size"
        )
        assert batch_inputs.shape[1] == dataset.input_dim(), (
            f"{dataset_class.__name__}: Batch input dimension doesn't match dataset"
        )

        # Test without batch size (full dataset)
        dataloader_full = dataset.get_dataloader(
            batch_size=len(dataset), shuffle=False, seed=42
        )
        full_batch = next(iter(dataloader_full))
        full_inputs, full_targets = full_batch

        assert full_inputs.shape[0] == len(dataset), (
            f"{dataset_class.__name__}: Full batch doesn't contain all samples"
        )


class TestDatasetTypes:
    """Test specific properties of different dataset types."""

    def test_regression_datasets_have_float_targets(self):
        """Test that regression datasets (UCI) have float targets."""
        regression_datasets = [EnergyDataset, ConcreteDataset]

        for dataset_class in regression_datasets:
            dataset = dataset_class.load(directory=None, store_on_disk=False)
            assert dataset.targets.dtype == jnp.float32, (
                f"{dataset_class.__name__} should have float32 targets for regression"
            )

    def test_classification_datasets_have_int_targets(self):
        """Test that classification datasets have integer targets."""
        classification_datasets = [
            CancerDataset,
            DigitsDataset,
            MNISTDataset,
            CIFAR10Dataset,
        ]

        for dataset_class in classification_datasets:
            dataset = dataset_class.load(directory=None, store_on_disk=False)
            assert dataset.targets.dtype == jnp.int32, (
                f"{dataset_class.__name__} should have int32 targets for classification"
            )

    def test_image_datasets_normalized(self):
        """Test that image datasets have normalized pixel values in [0, 1]."""
        image_datasets = [
            DigitsDataset,
            MNISTDataset,
            CIFAR10Dataset,
        ]

        for dataset_class in image_datasets:
            dataset = dataset_class.load(directory=None, store_on_disk=False)
            assert jnp.min(dataset.inputs) >= 0.0, (
                f"{dataset_class.__name__}: Image values should be >= 0"
            )
            assert jnp.max(dataset.inputs) <= 1.0, (
                f"{dataset_class.__name__}: Image values should be <= 1"
            )

    def test_mnist_dimensions(self):
        """Test MNIST has correct flattened dimensions."""
        dataset: Dataset = MNISTDataset.load(directory=None, store_on_disk=False)
        assert dataset.input_dim() == 784, "MNIST should have 784 features (28x28)"
        assert dataset.output_dim() == 10, "MNIST should have 10 classes"

    def test_cifar10_dimensions(self):
        """Test CIFAR-10 has correct flattened dimensions."""
        dataset: Dataset = CIFAR10Dataset.load(directory=None, store_on_disk=False)
        assert dataset.input_dim() == 3072, (
            "CIFAR-10 should have 3072 features (32x32x3)"
        )
        assert dataset.output_dim() == 10, "CIFAR-10 should have 10 classes"

    def test_digits_dimensions(self):
        """Test Digits has correct dimensions."""
        dataset: Dataset = DigitsDataset.load(directory=None, store_on_disk=False)
        assert dataset.input_dim() == 64, "Digits should have 64 features (8x8)"
        assert dataset.output_dim() == 10, "Digits should have 10 classes"


class TestDatasetReplaceTargets:
    """Test the replace_targets functionality."""

    def test_replace_targets(self):
        """Test that targets can be replaced."""
        dataset: Dataset = CancerDataset.load(directory=None, store_on_disk=False)

        # Create new targets
        new_targets = jnp.ones_like(dataset.targets)
        new_dataset = dataset.replace_targets(new_targets)

        # Check that targets were replaced
        assert jnp.allclose(new_dataset.targets, new_targets), (
            "Targets not replaced correctly"
        )

        # Check that inputs remain the same
        assert jnp.allclose(new_dataset.inputs, dataset.inputs), (
            "Inputs should not change"
        )
