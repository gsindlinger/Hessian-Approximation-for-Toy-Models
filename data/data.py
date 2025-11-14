from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from jax import numpy as jnp
from jaxtyping import Array
from sklearn.datasets import fetch_openml, make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from typing_extensions import override
from ucimlrepo import fetch_ucirepo

from config.config import DatasetConfig
from config.dataset_config import (
    CIFAR10DatasetConfig,
    MNISTDatasetConfig,
    RandomClassificationConfig,
    RandomRegressionConfig,
    UCIDatasetConfig,
)
from data.jax_dataloader import JAXDataLoader


@dataclass
class AbstractDataset(ABC):
    """Abstract base class for datasets in JAX."""

    train_test_split: float = 0.8
    transform: Optional[Callable] = None
    data: Array = field(init=False)
    targets: Array = field(init=False)
    train_data: Optional[Array] = field(default=None, init=False)
    train_targets: Optional[Array] = field(default=None, init=False)
    test_data: Optional[Array] = field(default=None, init=False)
    test_targets: Optional[Array] = field(default=None, init=False)

    def get_train_data(self) -> Tuple[Array, Array]:
        if self.train_data is None or self.train_targets is None:
            self.split_dataset()

        if self.train_data is None or self.train_targets is None:
            raise ValueError("Training data or targets are not available.")

        assert self.train_data is not None, "Train data should not be None"
        assert self.train_targets is not None, "Train targets should not be None"

        return self.train_data, self.train_targets

    def get_test_data(self) -> Tuple[Array, Array]:
        if self.test_data is None or self.test_targets is None:
            self.split_dataset()
        if self.test_data is None or self.test_targets is None:
            raise ValueError("Test data or targets are not available.")

        assert self.test_data is not None, "Test data should not be None"
        assert self.test_targets is not None, "Test targets should not be None"

        return self.test_data, self.test_targets

    def split_dataset(
        self,
    ) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
        """Split dataset into train and test sets. Assumes data to be normalized."""
        split_idx = int(len(self) * self.train_test_split)

        self.train_data = self.data[:split_idx]
        self.train_targets = self.targets[:split_idx]

        if self.train_test_split < 1.0:
            self.test_data = self.data[split_idx:]
            self.test_targets = self.targets[split_idx:]
        else:
            self.test_data = jnp.array([])
            self.test_targets = jnp.array([])

        return (self.train_data, self.train_targets), (
            self.test_data,
            self.test_targets,
        )

    def get_dataloaders(self, batch_size: int | None = None, shuffle: bool = False):
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


@dataclass
class RandomRegressionDataset(AbstractDataset):
    n_samples: int = field(default=100)
    n_features: int = field(default=10)
    n_targets: int = field(default=1)
    noise: float = field(default=0.1)
    random_state: Optional[int] = field(default=None)

    def __post_init__(self):
        data, targets = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_targets=self.n_targets,
            noise=self.noise,
            random_state=self.random_state,
        )[:2]

        self.data = jnp.array(data, dtype=jnp.float32)
        self.targets = jnp.array(targets.reshape(-1, 1), dtype=jnp.float32)

    @override
    def input_dim(self) -> int:
        return self.data.shape[1]

    @override
    def output_dim(self) -> int:
        return self.targets.shape[1]


@dataclass
class RandomClassificationDataset(AbstractDataset):
    n_samples: int = field(default=100)
    n_features: int = field(default=10)
    n_informative: int = field(default=5)
    n_classes: int = field(default=2)
    random_state: Optional[int] = field(default=42)

    def __post_init__(self):
        data, targets = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_classes=self.n_classes,
            random_state=self.random_state,
        )[:2]

        self.data = jnp.array(data, dtype=jnp.float32)
        self.targets = jnp.array(targets, dtype=jnp.int32)

    @override
    def input_dim(self) -> int:
        return self.data.shape[1]

    @override
    def output_dim(self) -> int:
        return int(self.targets.max() + 1)


@dataclass
class UCIDataset(AbstractDataset):
    name: str = field(default="energy")

    def __post_init__(self):
        match self.name:
            case "energy":
                id = 242
            case _:
                raise ValueError(f"Unknown UCI dataset: {self.name}")

        dataset = fetch_ucirepo(id=id)
        X = dataset.data.features  # type: ignore
        Y = dataset.data.targets  # type: ignore

        # Normalize
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)
        scaler_Y = StandardScaler()
        Y = scaler_Y.fit_transform(Y)

        self.data = jnp.array(X, dtype=jnp.float32)
        self.targets = jnp.array(Y, dtype=jnp.float32)

    @override
    def input_dim(self) -> int:
        return self.data.shape[1]

    @override
    def output_dim(self) -> int:
        return self.targets.shape[1] if len(self.targets.shape) > 1 else 1


@dataclass
class MNISTDataset(AbstractDataset):
    def __post_init__(self):
        mnist = fetch_openml("mnist_784", version=1)
        X = mnist.data.values
        Y = mnist.target.values.astype(jnp.int32)

        # Normalize
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)

        self.data = jnp.array(X, dtype=jnp.float32)
        self.targets = jnp.array(Y, dtype=jnp.int32)

    @override
    def input_dim(self) -> int:
        return self.data.shape[1]

    @override
    def output_dim(self) -> int:
        return int(self.targets.max() + 1)


@dataclass
class CIFAR10Dataset(AbstractDataset):
    def __post_init__(self):
        cifar10 = fetch_openml("CIFAR_10_small", version=1)
        X = cifar10.data.values
        Y = cifar10.target.values.astype(jnp.int32)

        # Normalize
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)

        self.data = jnp.array(X, dtype=jnp.float32)
        self.targets = jnp.array(Y, dtype=jnp.int32)

    @override
    def input_dim(self) -> int:
        return self.data.shape[1]

    @override
    def output_dim(self) -> int:
        return int(self.targets.max() + 1)
