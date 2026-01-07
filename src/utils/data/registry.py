from typing import Type

from src.utils.data.data import (
    CancerDataset,
    CIFAR10Dataset,
    ConcreteDataset,
    DigitsDataset,
    DownloadableDataset,
    EnergyDataset,
    FashionMNISTDataset,
    MNISTDataset,
    SklearnDigitsDataset,
)

DATASET_REGISTRY: dict[str, Type[DownloadableDataset]] = {
    # sklearn
    "sklearn_digits": SklearnDigitsDataset,
    # UCI
    "energy": EnergyDataset,
    "concrete": ConcreteDataset,
    "cancer": CancerDataset,
    "uci_digits": DigitsDataset,
    # OpenML
    "mnist": MNISTDataset,
    "fashion_mnist": FashionMNISTDataset,
    "cifar10": CIFAR10Dataset,
}
