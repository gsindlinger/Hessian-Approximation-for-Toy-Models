from abc import abstractmethod
from dataclasses import dataclass, field

import jax
from jax import numpy as jnp
from jaxtyping import Array
from sklearn.datasets import fetch_openml, make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from typing_extensions import override
from ucimlrepo import fetch_ucirepo

from deleuze.utils.data.jax_dataloader import JAXDataLoader


@dataclass
class Dataset:
    """Base class for datasets."""

    inputs: Array = field(init=False)
    targets: Array = field(init=False)

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

    def __len__(self) -> int:
        return len(self.inputs)

    @abstractmethod
    def input_dim(self) -> int:
        pass

    @abstractmethod
    def output_dim(self) -> int:
        pass


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

        self.inputs = jnp.array(data, dtype=jnp.float32)
        self.targets = jnp.array(targets.reshape(-1, 1), dtype=jnp.float32)

    @override
    def input_dim(self) -> int:
        return self.inputs.shape[1]

    @override
    def output_dim(self) -> int:
        return self.targets.shape[1]


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

        self.inputs = jnp.array(data, dtype=jnp.float32)
        self.targets = jnp.array(targets, dtype=jnp.int32)

    @override
    def input_dim(self) -> int:
        return self.inputs.shape[1]

    @override
    def output_dim(self) -> int:
        return int(self.targets.max() + 1)


@dataclass
class UCIDataset(Dataset):
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

        self.inputs = jnp.array(X, dtype=jnp.float32)
        self.targets = jnp.array(Y, dtype=jnp.float32)

    def input_dim(self) -> int:
        return self.inputs.shape[1]

    def output_dim(self) -> int:
        return self.targets.shape[1] if len(self.targets.shape) > 1 else 1


@dataclass
class MNISTDataset(Dataset):
    def __post_init__(self):
        mnist = fetch_openml("mnist_784", version=1)
        X = mnist.data.values
        Y = mnist.target.values.astype(jnp.int32)

        # Normalize
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)

        self.inputs = jnp.array(X, dtype=jnp.float32)
        self.targets = jnp.array(Y, dtype=jnp.int32)

    @override
    def input_dim(self) -> int:
        return self.inputs.shape[1]

    @override
    def output_dim(self) -> int:
        return int(self.targets.max() + 1)


@dataclass
class CIFAR10Dataset(Dataset):
    def __post_init__(self):
        cifar10 = fetch_openml("CIFAR_10_small", version=1)
        X = cifar10.data.values
        Y = cifar10.target.values.astype(jnp.int32)

        # Normalize
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)

        self.inputs = jnp.array(X, dtype=jnp.float32)
        self.targets = jnp.array(Y, dtype=jnp.int32)

    @override
    def input_dim(self) -> int:
        return self.inputs.shape[1]

    @override
    def output_dim(self) -> int:
        return int(self.targets.max() + 1)
