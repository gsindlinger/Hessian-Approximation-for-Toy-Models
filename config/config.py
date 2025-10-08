from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field
from typing import Literal, List, Dict
from simple_parsing import ArgumentParser


@dataclass
class DatasetConfig(ABC):
    """Configuration for dataset."""


@dataclass
class RandomRegressionConfig(DatasetConfig):
    n_samples: int = 1000
    n_features: int = 20
    n_targets: int = 1
    noise: float = 0.1
    random_state: int = 42
    train_test_split: float = 0.8


@dataclass
class RandomClassificationConfig(DatasetConfig):
    name = "random_classification"
    n_samples: int = 1000
    n_features: int = 40
    n_informative: int = 10
    n_classes: int = 10
    random_state: int = 42
    train_test_split: float = 0.8


@dataclass
class UCIDatasetConfig(DatasetConfig):
    name: Literal["energy"] = "energy"
    train_test_split: float = 0.8


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    name: Literal["linear"] = "linear"
    loss: Literal["mse", "cross_entropy"] = "mse"


@dataclass
class LinearModelConfig(ModelConfig):
    name: Literal["linear"] = "linear"
    loss: Literal["mse", "cross_entropy"] = "mse"
    hidden_dim: List[int] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    epochs: int = 20  # Number of training epochs
    lr: float = 0.01  # Learning rate
    save_checkpoint: bool = True  # Whether to save model checkpoints
    batch_size: int = 32  # Batch size for training
    optimizer: Literal["sgd", "adam"] = "sgd"  # Optimizer type
    loss: Literal["mse", "cross_entropy"] = "mse"  # Loss function


@dataclass
class HessianApproximationConfig(ABC):
    """Configuration for Hessian approximation."""

    name: Literal["exact-hessian-regression", "hessian", "fim", "gauss-newton"] = (
        "hessian"
    )


@dataclass
class Config:
    """Main configuration for the project."""

    dataset: DatasetConfig = field(
        default_factory=DatasetConfig
    )  # Which dataset to use
    model: ModelConfig = field(default_factory=ModelConfig)  # Which model to use
    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )  # Training parameters
    hessian_approximation: HessianApproximationConfig = field(
        default_factory=HessianApproximationConfig
    )

    @staticmethod
    def parse_args() -> Config:
        parser = ArgumentParser()

        # Add preset argument for predefined configs
        parser.add_argument(
            "--preset",
            type=str,
            default=None,
            choices=list_configs(),
            help="Use a predefined configuration preset",
        )
        parser.add_argument(
            "--list-presets",
            action="store_true",
            help="List all available configuration presets and exit",
        )

        parser.add_arguments(Config, dest="config")
        args = parser.parse_args()

        # List presets if requested
        if args.list_presets:
            print("Available configuration presets:")
            for preset_name in list_configs():
                print(f"  - {preset_name}")
            exit(0)

        # Use preset or command-line config
        if args.preset:
            print(f"Loading preset: {args.preset}")
            config = get_config(args.preset)
        else:
            config = args.config

        return config


# Predefined configurations
CONFIGS: Dict[str, Config] = {
    "random_regression": Config(
        dataset=RandomRegressionConfig(
            n_samples=1000,
            n_features=50,
            n_targets=1,
            noise=20,
            random_state=42,
        ),
        model=ModelConfig(name="linear", loss="mse"),
        training=TrainingConfig(
            epochs=200,
            lr=0.001,
            optimizer="sgd",
            loss="mse",
        ),
        hessian_approximation=HessianApproximationConfig(name="fim"),
    ),
    "random_regression_single_feature": Config(
        dataset=RandomRegressionConfig(
            n_samples=1000,
            n_features=1,
            n_targets=1,
            noise=30,
            random_state=42,
            train_test_split=1,
        ),
        model=ModelConfig(name="linear", loss="mse"),
        training=TrainingConfig(
            epochs=300,
            lr=0.001,
            optimizer="sgd",
            loss="mse",
        ),
        hessian_approximation=HessianApproximationConfig(name="hessian"),
    ),
    "random_classification": Config(
        dataset=RandomClassificationConfig(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_classes=3,
            random_state=42,
        ),
        model=LinearModelConfig(name="linear", loss="cross_entropy", hidden_dim=[]),
        training=TrainingConfig(
            epochs=100,
            lr=0.001,
            optimizer="sgd",
            loss="cross_entropy",
        ),
        hessian_approximation=HessianApproximationConfig(name="hessian"),
    ),
    "energy": Config(
        dataset=UCIDatasetConfig(
            name="energy",
            train_test_split=1,
        ),
        model=LinearModelConfig(name="linear", loss="mse", hidden_dim=[]),
        training=TrainingConfig(
            epochs=200,
            lr=0.001,
            batch_size=768,
            optimizer="sgd",
            loss="mse",
        ),
        hessian_approximation=HessianApproximationConfig(name="gauss-newton"),
    ),
}


def get_config(config_name: str = "default") -> Config:
    """Get a predefined configuration by name."""
    if config_name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    return CONFIGS[config_name]


def list_configs() -> List[str]:
    """List all available configuration names."""
    return list(CONFIGS.keys())
