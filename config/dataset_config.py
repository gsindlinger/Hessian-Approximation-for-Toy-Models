from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field


@dataclass
class DatasetConfig(ABC):
    name: str
    train_test_split: float = 0.8


@dataclass
class RandomRegressionConfig(DatasetConfig):
    name: str = field(default="random_regression", init=False)
    n_samples: int = 1000
    n_features: int = 20
    n_targets: int = 1
    noise: float = 0.1
    random_state: int = 42


@dataclass
class RandomClassificationConfig(DatasetConfig):
    name: str = field(default="random_classification", init=False)
    n_samples: int = 1000
    n_features: int = 40
    n_informative: int = 10
    n_classes: int = 10
    random_state: int = 42


@dataclass
class UCIDatasetConfig(DatasetConfig):
    name: str = field(default="uci", init=False)


@dataclass
class MNISTDatasetConfig(DatasetConfig):
    name: str = field(default="mnist", init=False)


@dataclass
class CIFAR10DatasetConfig(DatasetConfig):
    name: str = field(default="cifar10", init=False)
