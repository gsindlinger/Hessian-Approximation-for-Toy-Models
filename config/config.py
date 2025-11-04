from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal

import jax
from simple_parsing import ArgumentParser

jax.config.update("jax_enable_x64", True)


@dataclass
class DatasetConfig(ABC):
    train_test_split: float = 0.8


@dataclass
class RandomRegressionConfig(DatasetConfig):
    name = "random_regression"
    n_samples: int = 1000
    n_features: int = 20
    n_targets: int = 1
    noise: float = 0.1
    random_state: int = 42


@dataclass
class RandomClassificationConfig(DatasetConfig):
    name = "random_classification"
    n_samples: int = 1000
    n_features: int = 40
    n_informative: int = 10
    n_classes: int = 10
    random_state: int = 42


@dataclass
class UCIDatasetConfig(DatasetConfig):
    name: Literal["energy"] = "energy"


@dataclass
class MNISTDatasetConfig(DatasetConfig):
    pass


@dataclass
class CIFAR10DatasetConfig(DatasetConfig):
    pass


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
class MLPModelConfig(ModelConfig):
    name: Literal["mlp"] = "mlp"
    loss: Literal["mse", "cross_entropy"] = "mse"
    hidden_dim: List[int] = field(default_factory=list)
    use_bias: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training."""

    epochs: int = 20  # Number of training epochs
    lr: float = 0.01  # Learning rate
    save_checkpoint: bool = True  # Whether to save model checkpoints
    batch_size: int = 32  # Batch size for training
    optimizer: Literal["sgd", "adam"] = "sgd"  # Optimizer type
    loss: Literal["mse", "cross_entropy"] = "mse"  # Loss function


class HessianName(str, Enum):
    EXACT_HESSIAN_REGRESSION = "exact-hessian-regression"
    HESSIAN = "hessian"
    FIM = "fim"
    GAUSS_NEWTON = "gauss-newton"
    KFAC = "kfac"
    LISSA = "lissa"


@dataclass
class HessianApproximationConfig(ABC):
    """Configuration for Hessian approximation."""

    name: HessianName = HessianName.HESSIAN


@dataclass
class LiSSAConfig(HessianApproximationConfig):
    name: HessianName = HessianName.LISSA
    num_samples: int = 3
    recursion_depth: int = 500
    alpha: float = 0.05
    damping: float = 0.001
    batch_size: int = 128
    seed: int = 42
    convergence_tol: float = 1e-6
    check_convergence_every: int = 50


@dataclass
class KFACBuildConfig:
    use_pseudo_targets: bool = False
    pseudo_target_noise_std: float = 0.1
    collector_batch_size: int | None = None
    recalc_ekfac_components: bool = False


@dataclass
class KFACRunConfig:
    damping_lambda: float = 0.1
    damping_mode: Literal["mean_eigenvalue", "mean_corrections"] = "mean_eigenvalue"
    use_eigenvalue_correction: bool = True
    recalc_kfac_components: bool = False


@dataclass
class KFACConfig(HessianApproximationConfig):
    name: HessianName = HessianName.KFAC
    build_config: KFACBuildConfig = field(default_factory=KFACBuildConfig)
    run_config: KFACRunConfig = field(default_factory=KFACRunConfig)


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
            n_features=20,
            n_targets=1,
            noise=20,
            random_state=42,
        ),
        model=LinearModelConfig(name="linear", loss="mse", hidden_dim=[]),
        training=TrainingConfig(
            epochs=200,
            lr=0.01,
            optimizer="sgd",
            batch_size=100,
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
            n_features=10,
            n_informative=5,
            n_classes=2,
            random_state=42,
            train_test_split=1,
        ),
        model=LinearModelConfig(loss="cross_entropy", hidden_dim=[]),
        training=TrainingConfig(
            epochs=100,
            batch_size=100,
            lr=0.001,
            optimizer="sgd",
            loss="cross_entropy",
        ),
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
