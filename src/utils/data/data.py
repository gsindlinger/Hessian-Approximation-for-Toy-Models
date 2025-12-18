from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import jax
import numpy as np
import pandas as pd
from jax import numpy as jnp
from jaxtyping import Array
from sklearn.datasets import fetch_openml, make_classification, make_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing_extensions import override
from ucimlrepo import fetch_ucirepo

from src.utils.data.jax_dataloader import JAXDataLoader


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

    @staticmethod
    @abstractmethod
    def create_dataset_from_raw(raw: RawDataset) -> Dataset: ...

    @classmethod
    def load(
        cls,
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
        if directory is not None:
            path = Path(directory)
            if path.exists():
                raw = cls.load_from_disk(path)
            else:
                raw = cls.download()
                if store_on_disk:
                    path.mkdir(parents=True, exist_ok=True)
                    cls.save(path, raw)
        else:
            raw = cls.download()

        return cls.create_dataset_from_raw(raw)


@dataclass
class RandomRegressionDataset(Dataset):
    n_samples: int = field(default=100)
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

    @override
    def input_dim(self) -> int:
        return self.n_features

    @override
    def output_dim(self) -> int:
        return self.n_targets


@dataclass
class RandomClassificationDataset(Dataset):
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
class UCIDataset(DownloadableDataset):
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

        # Standardize inputs (standard for regression)
        X = StandardScaler().fit_transform(X)

        # Keep targets at original scale for regression tasks
        # This preserves interpretability and is standard practice

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


class EnergyDataset(UCIDataset):
    """Energy efficiency dataset - regression task."""

    id = 242


class ConcreteDataset(UCIDataset):
    """Concrete compressive strength - regression task."""

    id = 165


@dataclass
class UCIClassificationDataset(DownloadableDataset):
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

        # Standardize inputs (standard for classification)
        X = StandardScaler().fit_transform(X)

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
class OpenMLDataset(DownloadableDataset):
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
