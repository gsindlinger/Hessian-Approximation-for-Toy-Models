from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Optional, Self, Tuple, Type, TypeVar

import jax
import numpy as np
import pandas as pd
from jax import numpy as jnp
from jaxtyping import Array
from sklearn.datasets import (
    fetch_openml,
    load_digits,
    make_classification,
    make_regression,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing_extensions import override
from ucimlrepo import fetch_ucirepo

from src.config import DatasetEnum
from src.utils.data.jax_dataloader import JAXDataLoader

T = TypeVar("T", bound="DownloadableDataset")


@dataclass
class Dataset:
    """Base class for datasets."""

    inputs: Array = field(default_factory=lambda: jnp.array([]))
    targets: Array = field(default_factory=lambda: jnp.array([]))

    def get_dataloader(
        self, batch_size: int | None = None, shuffle: bool = False, seed: int = 42
    ) -> JAXDataLoader:
        """Get JAX dataloaders for training and testing."""
        data_loader = JAXDataLoader(
            self.inputs,
            self.targets,
            batch_size=batch_size,
            shuffle=shuffle,
            rng_key=jax.random.PRNGKey(seed=seed),
        )
        return data_loader

    def replace_targets(self, new_targets: Array) -> Dataset:
        """Replace the targets of the dataset with new targets."""
        assert new_targets.shape == self.targets.shape, (
            "New targets must have the same shape as existing targets."
        )
        return Dataset(
            inputs=self.inputs,
            targets=new_targets,
        )

    def __len__(self) -> int:
        return len(self.inputs)

    @abstractmethod
    def input_dim(self) -> int:
        pass

    @abstractmethod
    def output_dim(self) -> int:
        pass

    @abstractmethod
    def train_test_split(
        self, test_size: float = 0.2, seed: int = 42
    ) -> tuple[Dataset, Dataset]:
        pass

    @staticmethod
    def normalize_data(train_data: Array, test_data: Array) -> Tuple[Array, Array]:
        """Standardize inputs using StandardScaler."""

        scaler = StandardScaler()
        normalized_train_data = scaler.fit_transform(train_data)  # type: ignore
        normalized_test_data = scaler.transform(test_data)

        return (
            jnp.asarray(normalized_train_data, dtype=train_data.dtype),
            jnp.asarray(normalized_test_data, dtype=test_data.dtype),
        )

    def get_k_fold_splits(
        self,
        n_splits: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Iterator[
        tuple[
            tuple[Array, Array],
            tuple[Array, Array],
        ]
    ]:
        """
        Yield K-fold (train, test) splits.

        Each yield returns:
        ((train_inputs, train_targets), (test_inputs, test_targets))
        """
        num_samples = len(self)
        indices = jnp.arange(num_samples)

        if shuffle:
            key = jax.random.PRNGKey(seed)
            indices = jax.random.permutation(key, indices)

        folds = jnp.array_split(indices, n_splits)

        for i in range(n_splits):
            test_idx = folds[i]
            train_idx = jnp.concatenate([folds[j] for j in range(n_splits) if j != i])

            yield (
                (self.inputs[train_idx], self.targets[train_idx]),
                (self.inputs[test_idx], self.targets[test_idx]),
            )


class ClassificationDataset(Dataset):
    """Dataset for classification tasks."""

    def train_test_split(self, test_size: float = 0.2, seed: int = 42):
        y = self.targets.reshape(-1)
        num_classes = self.output_dim()

        key = jax.random.PRNGKey(seed)
        train_indices = []
        test_indices = []

        for cls in range(num_classes):
            cls_indices = jnp.where(y == cls)[0]

            key, subkey = jax.random.split(key)
            cls_indices = jax.random.permutation(subkey, cls_indices)

            split_idx = int(len(cls_indices) * (1 - test_size))
            train_indices.append(cls_indices[:split_idx])
            test_indices.append(cls_indices[split_idx:])

        train_indices = jnp.concatenate(train_indices)
        test_indices = jnp.concatenate(test_indices)

        # Final shuffle so classes are mixed
        key, subkey = jax.random.split(key)
        train_indices = jax.random.permutation(subkey, train_indices)

        key, subkey = jax.random.split(key)
        test_indices = jax.random.permutation(subkey, test_indices)

        train_dataset = replace(
            self,
            inputs=self.inputs[train_indices],
            targets=self.targets[train_indices],
        )
        test_dataset = replace(
            self,
            inputs=self.inputs[test_indices],
            targets=self.targets[test_indices],
        )

        return train_dataset, test_dataset

    def get_k_fold_splits(
        self,
        n_splits: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        y = self.targets.reshape(-1)
        num_classes = self.output_dim()

        key = jax.random.PRNGKey(seed)

        # collect per-class folds
        class_folds = [[] for _ in range(n_splits)]

        for cls in range(num_classes):
            cls_indices = jnp.where(y == cls)[0]

            if shuffle:
                key, subkey = jax.random.split(key)
                cls_indices = jax.random.permutation(subkey, cls_indices)

            splits = jnp.array_split(cls_indices, n_splits)
            for i in range(n_splits):
                class_folds[i].append(splits[i])

        folds = [jnp.concatenate(class_folds[i]) for i in range(n_splits)]

        for i in range(n_splits):
            test_idx = folds[i]
            train_idx = jnp.concatenate([folds[j] for j in range(n_splits) if j != i])

            yield (
                (self.inputs[train_idx], self.targets[train_idx]),
                (self.inputs[test_idx], self.targets[test_idx]),
            )


class RegressionDataset(Dataset):
    """Dataset for regression tasks."""

    def train_test_split(
        self, test_size: float = 0.2, seed: int = 42
    ) -> tuple[Dataset, Dataset]:
        num_samples = self.inputs.shape[0]
        indices = jnp.arange(num_samples)

        key = jax.random.PRNGKey(seed)
        shuffled_indices = jax.random.permutation(key, indices)

        split_idx = int(num_samples * (1 - test_size))
        train_indices = shuffled_indices[:split_idx]
        test_indices = shuffled_indices[split_idx:]

        train_dataset = replace(
            self,
            inputs=self.inputs[train_indices],
            targets=self.targets[train_indices],
        )
        test_dataset = replace(
            self,
            inputs=self.inputs[test_indices],
            targets=self.targets[test_indices],
        )

        return train_dataset, test_dataset


@dataclass(frozen=True)
class RawDataset:
    X: Any
    Y: Any
    metadata: dict | None = None


class DownloadableDataset(Dataset, ABC):
    """Dataset that can be downloaded and cached."""

    @classmethod
    @abstractmethod
    def download(cls) -> RawDataset: ...

    @staticmethod
    @abstractmethod
    def save(directory: Path, raw: RawDataset) -> None: ...

    @staticmethod
    @abstractmethod
    def load_from_disk(directory: Path) -> RawDataset: ...

    @classmethod
    @abstractmethod
    def create_dataset_from_raw(cls, raw: RawDataset) -> Self: ...

    @staticmethod
    def load(
        dataset: DatasetEnum,
        directory: Optional[str] = None,
        store_on_disk: bool = True,
    ) -> Dataset:
        """
        Load a dataset from disk or download it.

        If a directory is provided and exists, load the dataset from disk.
        Otherwise, download the dataset and optionally save it to disk.

        Args:
            directory: Optional path to directory containing the dataset / also the directory to save the dataset.
            store_on_disk: Whether to save the downloaded dataset to disk.

        Returns:
            A Dataset object created from the loaded or downloaded data.
        """
        dataset_cls = DatasetRegistry.get_dataset(dataset)
        if dataset_cls is None:
            raise ValueError(f"Dataset '{dataset}' not found in registry.")
        if directory is not None:
            path = Path(directory)
            if path.exists():
                raw = dataset_cls.load_from_disk(path)
            else:
                raw = dataset_cls.download()
                if store_on_disk:
                    path.mkdir(parents=True, exist_ok=True)
                    dataset_cls.save(path, raw)
        else:
            raw = dataset_cls.download()

        return dataset_cls.create_dataset_from_raw(raw)


@dataclass
class RandomRegressionDataset(RegressionDataset):
    n_samples: int = field(default=1000)
    n_features: int = field(default=10)
    n_targets: int = field(default=1)
    noise: float = field(default=0.1)
    seed: int = field(default=42)

    def __post_init__(self):
        data, targets = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_targets=self.n_targets,
            noise=self.noise,
            random_state=self.seed,
        )[:2]

        # Standardize inputs (common for regression)
        data = StandardScaler().fit_transform(data)

        # Keep targets at original scale for regression
        self.inputs = jnp.array(data, dtype=jnp.float32)
        self.targets = jnp.array(targets, dtype=jnp.float32)

        # Reshape targets if single output
        if self.n_targets == 1 and len(self.targets.shape) == 1:
            self.targets = self.targets.reshape(-1, 1)

    def input_dim(self) -> int:
        return self.n_features

    def output_dim(self) -> int:
        return self.n_targets


@dataclass
class RandomClassificationDataset(ClassificationDataset):
    n_samples: int = field(default=100)
    n_features: int = field(default=10)
    n_informative: int = field(default=5)
    n_classes: int = field(default=2)
    seed: int = field(default=42)

    def __post_init__(self):
        data, targets = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_classes=self.n_classes,
            random_state=self.seed,
        )[:2]

        # Standardize inputs (common for classification)
        data = StandardScaler().fit_transform(data)

        # Keep targets as class indices (integers)
        self.inputs = jnp.array(data, dtype=jnp.float32)
        self.targets = jnp.array(targets, dtype=jnp.int32)

    @override
    def input_dim(self) -> int:
        return self.n_features

    @override
    def output_dim(self) -> int:
        return self.n_classes


@dataclass
class SklearnDigitsDataset(DownloadableDataset, ClassificationDataset):
    """
    sklearn.datasets.load_digits dataset.
    8x8 images (64 features) with pixel values in range 0-16, already flattened.
    """

    @classmethod
    def download(cls) -> RawDataset:
        digits = load_digits()

        X = digits.data  # type: ignore
        Y = digits.target  # type: ignore

        return RawDataset(X=X, Y=Y)

    @staticmethod
    def save(directory: Path, raw: RawDataset) -> None:
        pd.DataFrame(raw.X).to_csv(directory / "X.csv", index=False)
        pd.DataFrame(raw.Y).to_csv(directory / "Y.csv", index=False)

    @staticmethod
    def load_from_disk(directory: Path) -> RawDataset:
        X = pd.read_csv(directory / "X.csv").values
        Y = pd.read_csv(directory / "Y.csv").values.ravel()
        return RawDataset(X=X, Y=Y)

    @classmethod
    def create_dataset_from_raw(cls, raw: RawDataset) -> Self:
        X = raw.X.astype(jnp.float32)
        Y = raw.Y.astype(jnp.int32)

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Normalize pixels from [0, 16] to [0, 1] range

        if X.max() > 1.0:
            X = X / 16.0

        return cls(
            inputs=jnp.asarray(X, dtype=jnp.float32),
            targets=jnp.asarray(Y, dtype=jnp.int32),
        )

    def input_dim(self) -> int:
        return 64

    def output_dim(self) -> int:
        return 10


@dataclass
class UCIRegressionDataset(DownloadableDataset, RegressionDataset):
    """Base class for UCI datasets - handles regression datasets."""

    id: int = 0

    @classmethod
    def download(cls) -> RawDataset:
        dataset = fetch_ucirepo(id=cls.id)
        assert dataset.data is not None, "Fetched dataset has no data."
        return RawDataset(
            X=dataset.data.features,
            Y=dataset.data.targets,
            metadata=dataset.metadata,
        )

    @staticmethod
    def save(directory: Path, raw: RawDataset) -> None:
        raw.X.to_csv(directory / "X.csv", index=False)
        raw.Y.to_csv(directory / "Y.csv", index=False)
        with open(directory / "metadata.json", "w") as f:
            json.dump(raw.metadata, f, indent=2)

    @staticmethod
    def load_from_disk(directory: Path) -> RawDataset:
        X = pd.read_csv(directory / "X.csv")
        Y = pd.read_csv(directory / "Y.csv")
        with open(directory / "metadata.json") as f:
            metadata = json.load(f)
        return RawDataset(X=X, Y=Y, metadata=metadata)

    @classmethod
    def create_dataset_from_raw(cls, raw: RawDataset) -> Dataset:
        X = raw.X.values
        Y = raw.Y.values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)

        # Ensure Y is 2D
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        return cls(
            inputs=jnp.asarray(X, dtype=jnp.float32),
            targets=jnp.asarray(Y, dtype=jnp.float32),
        )

    def input_dim(self) -> int:
        return self.inputs.shape[1]

    def output_dim(self) -> int:
        return self.targets.shape[1]


class EnergyDataset(UCIRegressionDataset):
    """Energy efficiency dataset - regression task."""

    id = 242


class ConcreteDataset(UCIRegressionDataset):
    """Concrete compressive strength - regression task."""

    id = 165


@dataclass
class UCIClassificationDataset(DownloadableDataset, ClassificationDataset):
    """Base class for UCI classification datasets."""

    id: int = 0

    @classmethod
    def download(cls) -> RawDataset:
        dataset = fetch_ucirepo(id=cls.id)
        assert dataset.data is not None, "Fetched dataset has no data."
        return RawDataset(
            X=dataset.data.features,
            Y=dataset.data.targets,
            metadata=dataset.metadata,
        )

    @staticmethod
    def save(directory: Path, raw: RawDataset) -> None:
        raw.X.to_csv(directory / "X.csv", index=False)
        raw.Y.to_csv(directory / "Y.csv", index=False)
        with open(directory / "metadata.json", "w") as f:
            json.dump(raw.metadata, f, indent=2)

    @staticmethod
    def load_from_disk(directory: Path) -> RawDataset:
        X = pd.read_csv(directory / "X.csv")
        Y = pd.read_csv(directory / "Y.csv")
        with open(directory / "metadata.json") as f:
            metadata = json.load(f)
        return RawDataset(X=X, Y=Y, metadata=metadata)

    @classmethod
    def create_dataset_from_raw(cls, raw: RawDataset) -> Dataset:
        X = raw.X.values
        Y = raw.Y.values.ravel()  # Flatten to 1D

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Encode labels as integers starting from 0
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)

        return cls(
            inputs=jnp.asarray(X, dtype=jnp.float32),
            targets=jnp.asarray(Y, dtype=jnp.int32),
        )

    def input_dim(self) -> int:
        return self.inputs.shape[1]

    def output_dim(self) -> int:
        return int(jnp.max(self.targets)) + 1


class CancerDataset(UCIClassificationDataset):
    """Breast cancer Wisconsin - binary classification."""

    id = 17


class DigitsDataset(UCIClassificationDataset):
    """Optical recognition of handwritten digits - 10-class classification.
    8x8 images (64 features) with pixel values in range 0-16, already flattened.
    """

    id = 80

    @classmethod
    def create_dataset_from_raw(cls, raw: RawDataset) -> Dataset:
        X = raw.X.values
        Y = raw.Y.values.ravel()

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Normalize pixels from [0, 16] to [0, 1] range
        # (digits dataset uses 0-16 range, not 0-255)
        if X.max() > 1.0:
            X = X / 16.0

        # Encode labels as integers starting from 0
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)

        return cls(
            inputs=jnp.asarray(X, dtype=jnp.float32),
            targets=jnp.asarray(Y, dtype=jnp.int32),
        )


@dataclass
class OpenMLDataset(DownloadableDataset, ClassificationDataset):
    """Base class for OpenML datasets - handles classification tasks."""

    name: str = ""

    @classmethod
    def download(cls) -> RawDataset:
        dataset = fetch_openml(cls.name, as_frame=True, parser="auto")
        return RawDataset(
            X=dataset.data,
            Y=dataset.target,
            metadata={"details": dataset.details},
        )

    @staticmethod
    def save(directory: Path, raw: RawDataset) -> None:
        raw.X.to_csv(directory / "X.csv", index=False)
        raw.Y.to_csv(directory / "Y.csv", index=False)

    @staticmethod
    def load_from_disk(directory: Path) -> RawDataset:
        X = pd.read_csv(directory / "X.csv")
        Y = pd.read_csv(directory / "Y.csv").squeeze()
        return RawDataset(X=X, Y=Y)

    @classmethod
    def create_dataset_from_raw(cls, raw: RawDataset) -> Dataset:
        X = raw.X.values
        Y = raw.Y.values if hasattr(raw.Y, "values") else raw.Y

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # For image datasets (MNIST, CIFAR), normalize to [0, 1] range
        # This is standard practice for image data
        if X.max() > 1.0:
            X = X / 255.0

        # Flatten images if needed (images come pre-flattened from OpenML usually)
        # MNIST: 28*28=784, CIFAR: 32*32*3=3072
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        # No additional standardization for images as pixel values
        # are already on a comparable scale after normalization

        # Encode labels as integers
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)

        return cls(
            inputs=jnp.asarray(X, dtype=jnp.float32),
            targets=jnp.asarray(Y, dtype=jnp.int32),
        )

    def input_dim(self) -> int:
        return self.inputs.shape[1]

    def output_dim(self) -> int:
        return int(jnp.max(self.targets)) + 1


@dataclass
class MNISTDataset(OpenMLDataset):
    """MNIST handwritten digits - 28x28 grayscale images, 10 classes."""

    name = "mnist_784"


@dataclass
class FashionMNISTDataset(OpenMLDataset):
    """Fashion-MNIST - 28x28 grayscale images of clothing items, 10 classes."""

    name = "Fashion-MNIST"


@dataclass
class CIFAR10Dataset(OpenMLDataset):
    """CIFAR-10 - 32x32 color images, 10 classes."""

    name = "CIFAR_10_small"


class DatasetRegistry:
    """Registry for datasets."""

    REGISTRY: dict[DatasetEnum, Type[DownloadableDataset]] = {
        DatasetEnum.MNIST: MNISTDataset,
        DatasetEnum.FASHION_MNIST: FashionMNISTDataset,
        DatasetEnum.CIFAR10: CIFAR10Dataset,
        DatasetEnum.SKLEARN_DIGITS: SklearnDigitsDataset,
        DatasetEnum.ENERGY: EnergyDataset,
        DatasetEnum.CONCRETE: ConcreteDataset,
        DatasetEnum.CANCER: CancerDataset,
        DatasetEnum.DIGITS: DigitsDataset,
    }

    @staticmethod
    def get_dataset(name: DatasetEnum, *args, **kwargs) -> DownloadableDataset:
        dataset_cls = DatasetRegistry.REGISTRY.get(name)
        if dataset_cls is None:
            raise ValueError(f"Unknown dataset: {name}")
        return dataset_cls(*args, **kwargs)
