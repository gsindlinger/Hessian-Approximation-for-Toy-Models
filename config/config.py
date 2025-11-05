from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import jax
from simple_parsing import ArgumentParser

from config.dataset_config import (
    DatasetConfig,
    RandomClassificationConfig,
    RandomRegressionConfig,
    UCIDatasetConfig,
)
from config.hessian_approximation_config import HessianApproximationConfig, HessianName
from config.model_config import LinearModelConfig, ModelConfig
from config.training_config import TrainingConfig

jax.config.update("jax_enable_x64", True)


@dataclass
class Config:
    """Main configuration for the project."""

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    hessian_approximation: HessianApproximationConfig = field(
        default_factory=lambda: HessianApproximationConfig(name=HessianName.HESSIAN)
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

    @staticmethod
    def model_training_dataset_hash(
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        length: int = 8,
    ) -> str:
        """Generate a unique hash string for the combination of dataset, model, and training configs."""
        import hashlib
        import json

        # Serialize configurations to JSON strings
        dataset_json = json.dumps(vars(dataset_config), sort_keys=True)
        model_json = json.dumps(vars(model_config), sort_keys=True)
        training_json = json.dumps(vars(training_config), sort_keys=True)

        # Combine all JSON strings
        combined = dataset_json + model_json + training_json

        # Generate SHA256 hash
        hash_object = hashlib.sha256(combined.encode())
        hash_hex = hash_object.hexdigest()[:10]  # Use first 10 characters for brevity

        return hash_hex


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
        model=LinearModelConfig(loss="mse", hidden_dim=[]),
        training=TrainingConfig(
            epochs=200,
            lr=0.01,
            optimizer="sgd",
            batch_size=100,
            loss="mse",
        ),
        hessian_approximation=HessianApproximationConfig(name=HessianName.FIM),
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
        model=LinearModelConfig(loss="mse"),
        training=TrainingConfig(
            epochs=300,
            lr=0.001,
            optimizer="sgd",
            loss="mse",
        ),
        hessian_approximation=HessianApproximationConfig(name=HessianName.HESSIAN),
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
            train_test_split=1,
        ),
        model=LinearModelConfig(loss="mse", hidden_dim=[]),
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
