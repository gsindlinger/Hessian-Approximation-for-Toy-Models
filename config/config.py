from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal

from simple_parsing import ArgumentParser

from config.dataset_config import (
    DatasetConfig,
    RandomClassificationConfig,
    RandomRegressionConfig,
    UCIDatasetConfig,
)
from config.hessian_approximation_config import HessianApproximationConfig, HessianName
from config.model_config import LinearModelConfig, MLPModelConfig, ModelConfig
from config.training_config import TrainingConfig


@dataclass
class Config:
    """Main configuration for the project."""

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    hessian_approximation: HessianApproximationConfig | None = None
    device: Literal["auto", "cpu", "gpu", "tpu"] = "auto"
    seed: int = 42

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
        parser.add_argument(
            "--device",
            type=str,
            default="auto",
            choices=["auto", "cpu", "gpu", "tpu"],
            help="Select device to run on: 'auto' lets JAX decide, 'cpu' forces CPU, 'gpu' forces GPU if available",
        )

        # parser.add_arguments(Config, dest="config")
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

        config.device = args.device
        return config

    @staticmethod
    def model_training_dataset_hash(
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        seed: int,
        hessian_approx_config: HessianApproximationConfig | None = None,
        length: int = 10,
    ) -> str:
        """Generate a unique hash string for the combination of dataset, model, and training configs."""
        import hashlib
        import json

        # Serialize configurations to JSON strings
        dataset_json = json.dumps(asdict(dataset_config), sort_keys=True)
        model_json = json.dumps(asdict(model_config), sort_keys=True)
        training_json = json.dumps(asdict(training_config), sort_keys=True)

        if hessian_approx_config is not None:
            hessian_json = json.dumps(hessian_approx_config.to_dict(), sort_keys=True)
            # Combine all JSON strings including Hessian config
            combined = (
                dataset_json + model_json + training_json + hessian_json + str(seed)
            )
        else:
            # Combine all JSON strings
            combined = dataset_json + model_json + training_json + str(seed)

        # Generate SHA256 hash
        hash_object = hashlib.sha256(combined.encode())
        hash_hex = hash_object.hexdigest()[
            :length
        ]  # Use first length characters for brevity

        return hash_hex


# Predefined configurations
CONFIGS: Dict[str, Config] = {
    "random_regression": Config(
        dataset=RandomRegressionConfig(
            n_samples=1000,
            n_features=20,
            n_targets=1,
            noise=20,
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
            n_samples=500,
            n_features=10,
            n_informative=5,
            n_classes=2,
            train_test_split=1,
        ),
        model=LinearModelConfig(loss="cross_entropy", hidden_dim=[]),
        training=TrainingConfig(
            epochs=30,
            batch_size=100,
            lr=0.001,
            optimizer="sgd",
            loss="cross_entropy",
        ),
    ),
    "random_classification_large": Config(
        dataset=RandomClassificationConfig(
            n_samples=2000,
            n_features=200,
            n_informative=50,
            n_classes=10,
            train_test_split=1,
        ),
        model=MLPModelConfig(loss="cross_entropy", hidden_dim=[100]),
        training=TrainingConfig(
            epochs=200,
            lr=0.001,
            batch_size=100,
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
