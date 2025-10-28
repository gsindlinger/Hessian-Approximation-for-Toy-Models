from abc import ABC, abstractmethod

import numpy as np
from sklearn.datasets import fetch_openml, make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from config.config import (
    CIFAR10DatasetConfig,
    DatasetConfig,
    MNISTDatasetConfig,
    RandomClassificationConfig,
    RandomRegressionConfig,
    UCIDatasetConfig,
)
from data.jax_dataloader import JAXDataLoader


class AbstractDataset(ABC):
    """Abstract base class for datasets in JAX."""

    data: np.ndarray
    targets: np.ndarray
    train_data: np.ndarray | None = None
    train_targets: np.ndarray | None = None
    test_data: np.ndarray | None = None
    test_targets: np.ndarray | None = None

    def __init__(self, train_test_split=0.8, transform=None):
        super().__init__()
        self.train_test_split = train_test_split
        self.transform = transform

    def get_train_data(self):
        if self.train_data is None or self.train_targets is None:
            raise ValueError("Train data not set. Please call split_dataset() first.")
        return self.train_data, self.train_targets

    def get_test_data(self):
        if self.test_data is None or self.test_targets is None:
            raise ValueError("Test data not set. Please call split_dataset() first.")
        return self.test_data, self.test_targets

    def split_dataset(self):
        """Split dataset into train and test sets. Assumes data to be normalized."""
        split_idx = int(len(self) * self.train_test_split)

        self.train_data = self.data[:split_idx]
        self.train_targets = self.targets[:split_idx]

        if self.train_test_split < 1.0:
            self.test_data = self.data[split_idx:]
            self.test_targets = self.targets[split_idx:]
        else:
            self.test_data = None
            self.test_targets = None

        return (self.train_data, self.train_targets), (
            self.test_data,
            self.test_targets,
        )

    def get_dataloaders(self, batch_size=32, shuffle=True):
        """
        Split dataset into train and test sets and return data iterators.
        Returns tuple of (train_loader, test_loader).
        """
        if self.train_data is None or self.train_targets is None:
            self.split_dataset()

        train_loader = JAXDataLoader(
            self.train_data, self.train_targets, batch_size=batch_size, shuffle=shuffle
        )

        if self.train_test_split < 1.0 and self.test_data is not None:
            test_loader = JAXDataLoader(
                self.test_data,
                self.test_targets,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        else:
            test_loader = None

        return train_loader, test_loader

    @abstractmethod
    def input_dim(self) -> int:
        pass

    @abstractmethod
    def output_dim(self) -> int:
        pass

    def __len__(self) -> int:
        return len(self.data)


def create_dataset(config: DatasetConfig) -> AbstractDataset:
    """Create dataset from config."""
    dataset_map = {
        RandomRegressionConfig: RandomRegressionDataset,
        RandomClassificationConfig: RandomClassificationDataset,
        UCIDatasetConfig: UCIDataset,
        MNISTDatasetConfig: MNISTDataset,
        CIFAR10DatasetConfig: CIFAR10Dataset,
    }

    dataset_cls = dataset_map.get(type(config))
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset: {type(config).__name__}")

    dataset_kwargs = vars(config).copy()
    return dataset_cls(**dataset_kwargs)


class RandomRegressionDataset(AbstractDataset):
    def __init__(
        self, n_samples, n_features, n_targets, noise, random_state, train_test_split
    ):
        super().__init__(train_test_split=train_test_split)
        data, targets = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_targets=n_targets,
            noise=noise,
            random_state=random_state,
        )[:2]

        # Store as numpy arrays first
        self.data = data.astype(np.float32)
        self.targets = targets.reshape(-1, 1).astype(np.float32)

    @override
    def input_dim(self):
        return self.data.shape[1]

    @override
    def output_dim(self):
        return self.targets.shape[1]


class RandomClassificationDataset(AbstractDataset):
    def __init__(
        self,
        n_samples,
        n_features,
        n_informative,
        n_classes,
        random_state,
        train_test_split,
    ):
        super().__init__(train_test_split=train_test_split)
        data, targets = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_classes=n_classes,
            random_state=random_state,
        )[:2]

        # Store as numpy arrays
        self.data = data.astype(np.float32)
        self.targets = targets.astype(np.int32)

    @override
    def input_dim(self):
        return self.data.shape[1]

    @override
    def output_dim(self) -> int:
        return int(self.targets.max() + 1)


class UCIDataset(AbstractDataset):
    def __init__(self, name: str, train_test_split: float = 0.8):
        super().__init__(train_test_split=train_test_split)

        match name:
            case "energy":
                id = 242
            case _:
                raise ValueError(f"Unknown UCI dataset: {name}")

        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=id)
        X = dataset.data.features  # type: ignore
        Y = dataset.data.targets  # type: ignore

        # Normalize
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)
        scaler_Y = StandardScaler()
        Y = scaler_Y.fit_transform(Y)

        # Store as numpy arrays
        self.data = X.astype(np.float32)
        self.targets = Y.astype(np.float32)

    @override
    def input_dim(self):
        return self.data.shape[1]

    @override
    def output_dim(self):
        return self.targets.shape[1] if len(self.targets.shape) > 1 else 1


class MNISTDataset(AbstractDataset):
    def __init__(self, train_test_split: float = 0.8):
        super().__init__(train_test_split=train_test_split)

        mnist = fetch_openml("mnist_784", version=1)
        X = mnist.data.values
        Y = mnist.target.values.astype(np.int32)

        # Normalize
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)

        # Store as numpy arrays
        self.data = X.astype(np.float32)
        self.targets = Y.astype(np.int32)

    @override
    def input_dim(self):
        return self.data.shape[1]

    @override
    def output_dim(self) -> int:
        return int(self.targets.max() + 1)


class CIFAR10Dataset(AbstractDataset):
    def __init__(self, train_test_split: float = 0.8):
        super().__init__(train_test_split=train_test_split)

        cifar10 = fetch_openml("CIFAR_10_small", version=1)
        X = cifar10.data.values
        Y = cifar10.target.values.astype(np.int32)

        # Normalize
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)

        # Store as numpy arrays
        self.data = X.astype(np.float32)
        self.targets = Y.astype(np.int32)

    @override
    def input_dim(self):
        return self.data.shape[1]

    @override
    def output_dim(self) -> int:
        return int(self.targets.max() + 1)
