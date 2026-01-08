"""
Helper utilities for building experiment configurations programmatically.

This is especially useful when you want to generate sweeps or many model configurations.
"""

from dataclasses import asdict
from typing import List, Optional

import yaml

from src.config import (
    LOSS_TYPE,
    ComputationType,
    DampingStrategy,
    DatasetConfig,
    DatasetEnum,
    ExperimentConfig,
    HessianAnalysisConfig,
    HessianApproximator,
    HessianComputationConfig,
    ModelArchitecture,
    ModelConfig,
    OptimizerType,
    TrainingConfig,
    VectorSamplingMethod,
)
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric


class ModelConfigBuilder:
    """Builder pattern for creating ModelConfig with training settings."""

    def __init__(self, architecture: ModelArchitecture, hidden_dim):
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.loss = LOSS_TYPE.CROSS_ENTROPY
        self.init_seed = 42
        self.skip_existing = True

        # Training defaults
        self.learning_rate = 1e-3
        self.weight_decay = 0.0
        self.optimizer = OptimizerType.ADAMW
        self.epochs = 500
        self.batch_size = 32

    def with_loss(self, loss: LOSS_TYPE):
        self.loss = loss
        return self

    def with_seed(self, seed: int):
        self.init_seed = seed
        return self

    def with_training(
        self,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        optimizer: Optional[OptimizerType] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if weight_decay is not None:
            self.weight_decay = weight_decay
        if optimizer is not None:
            self.optimizer = optimizer
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        return self

    def build(self) -> ModelConfig:
        return ModelConfig(
            architecture=self.architecture,
            hidden_dim=self.hidden_dim,
            loss=self.loss,
            init_seed=self.init_seed,
            skip_existing=self.skip_existing,
            training=TrainingConfig(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                optimizer=self.optimizer,
                epochs=self.epochs,
                batch_size=self.batch_size,
            ),
        )


def create_hyperparameter_sweep(
    architecture: ModelArchitecture,
    hidden_dim,
    learning_rates: List[float],
    weight_decays: List[float],
    **kwargs,
) -> List[ModelConfig]:
    """
    Create a grid of models with different hyperparameters.

    Example:
        models = create_hyperparameter_sweep(
            ModelArchitecture.MLP,
            [64, 64],
            learning_rates=[1e-3, 1e-4],
            weight_decays=[0.0, 1e-3],
            epochs=500,
        )
    """
    models = []
    for lr in learning_rates:
        for wd in weight_decays:
            model = (
                ModelConfigBuilder(architecture, hidden_dim)
                .with_training(learning_rate=lr, weight_decay=wd, **kwargs)
                .build()
            )
            models.append(model)
    return models


def create_architecture_sweep(
    architectures_and_dims: List[tuple],
    training_config: Optional[dict] = None,
) -> List[ModelConfig]:
    """
    Create models for different architectures.

    Example:
        models = create_architecture_sweep([
            (ModelArchitecture.MLP, [64]),
            (ModelArchitecture.MLP, [64, 64]),
            (ModelArchitecture.MLP, [128, 64, 32]),
            (ModelArchitecture.MLPSWIGLU, [[10, 10, 10]]),
        ], training_config={'learning_rate': 0.001, 'epochs': 500})
    """
    training_config = training_config or {}
    models = []
    for arch, dims in architectures_and_dims:
        model = ModelConfigBuilder(arch, dims).with_training(**training_config).build()
        models.append(model)
    return models


def create_depth_sweep(
    architecture: ModelArchitecture, width: int, depths: List[int], **training_kwargs
) -> List[ModelConfig]:
    """
    Create models with different depths but same width.

    Example:
        # Create MLPs with 64 units and depths 1, 2, 4, 8
        models = create_depth_sweep(
            ModelArchitecture.MLP,
            width=64,
            depths=[1, 2, 4, 8],
            learning_rate=0.001,
        )
    """
    models = []
    for depth in depths:
        if architecture == ModelArchitecture.MLPSWIGLU:
            # For SwiGLU, each layer is a tuple
            hidden_dim = [(width, width, width)] * depth
        else:
            hidden_dim = [width] * depth

        model = (
            ModelConfigBuilder(architecture, hidden_dim)
            .with_training(**training_kwargs)
            .build()
        )
        models.append(model)
    return models


def create_width_sweep(
    architecture: ModelArchitecture, widths: List[int], depth: int, **training_kwargs
) -> List[ModelConfig]:
    """
    Create models with different widths but same depth.

    Example:
        # Create 3-layer MLPs with different widths
        models = create_width_sweep(
            ModelArchitecture.MLP,
            widths=[32, 64, 128],
            depth=3,
            learning_rate=0.001,
        )
    """
    models = []
    for width in widths:
        if architecture == ModelArchitecture.MLPSWIGLU:
            hidden_dim = [(width, width, width)] * depth
        else:
            hidden_dim = [width] * depth

        model = (
            ModelConfigBuilder(architecture, hidden_dim)
            .with_training(**training_kwargs)
            .build()
        )
        models.append(model)
    return models


# =============================================================================
# Example Usage
# =============================================================================


def example_simple_config():
    """Example: Simple config with a few models."""
    config = ExperimentConfig(
        experiment_name="simple_test",
        seed=42,
        dataset=DatasetConfig(
            name=DatasetEnum.DIGITS,
            path="experiments/data/datasets/digits",
        ),
        models=[
            ModelConfigBuilder(ModelArchitecture.MLP, [64])
            .with_training(learning_rate=0.001)
            .build(),
            ModelConfigBuilder(ModelArchitecture.MLP, [64, 64])
            .with_training(learning_rate=0.001, weight_decay=0.0001)
            .build(),
            ModelConfigBuilder(ModelArchitecture.MLPSWIGLU, [[10, 10, 10]])
            .with_training(learning_rate=0.002)
            .build(),
        ],
        run_hessian_analysis=True,
    )
    return config


def example_hyperparameter_sweep():
    """Example: Sweep over learning rates and weight decays."""
    mlp_64 = create_hyperparameter_sweep(
        ModelArchitecture.MLP,
        [64],
        learning_rates=[5e-4, 1e-3, 2e-3],
        weight_decays=[0.0, 1e-4, 1e-3],
        epochs=500,
    )

    mlp_128 = create_hyperparameter_sweep(
        ModelArchitecture.MLP,
        [128],
        learning_rates=[5e-4, 1e-3],
        weight_decays=[0.0, 1e-4],
        epochs=500,
    )

    config = ExperimentConfig(
        experiment_name="hyperparam_sweep",
        seed=42,
        dataset=DatasetConfig(
            name=DatasetEnum.DIGITS,
            path="experiments/data/datasets/digits",
        ),
        models=mlp_64 + mlp_128,
        run_hessian_analysis=True,
    )
    return config


def example_architecture_sweep():
    """Example: Compare different architectures."""
    architectures = [
        (ModelArchitecture.MLP, [16]),
        (ModelArchitecture.MLP, [32]),
        (ModelArchitecture.MLP, [64]),
        (ModelArchitecture.MLP, [16, 16, 16, 16]),
        (ModelArchitecture.MLP, [32, 32, 32, 32]),
        (ModelArchitecture.MLPSWIGLU, [[5, 5, 5]]),
        (ModelArchitecture.MLPSWIGLU, [[10, 10, 10]]),
        (ModelArchitecture.MLPSWIGLU, [[5, 5, 5]] * 4),
    ]

    config = ExperimentConfig(
        experiment_name="architecture_sweep",
        seed=42,
        dataset=DatasetConfig(
            name=DatasetEnum.DIGITS,
            path="experiments/data/datasets/digits",
        ),
        models=create_architecture_sweep(
            architectures, training_config={"learning_rate": 0.001, "epochs": 500}
        ),
        run_hessian_analysis=True,
    )
    return config


def example_depth_sweep():
    """Example: Study effect of depth."""
    models = []

    # MLP depth sweep
    models.extend(
        create_depth_sweep(
            ModelArchitecture.MLP,
            width=64,
            depths=[1, 2, 4, 8],
            learning_rate=0.001,
        )
    )

    # SwiGLU depth sweep
    models.extend(
        create_depth_sweep(
            ModelArchitecture.MLPSWIGLU,
            width=10,
            depths=[1, 2, 4, 8],
            learning_rate=0.002,
        )
    )

    config = ExperimentConfig(
        experiment_name="depth_sweep",
        seed=42,
        dataset=DatasetConfig(
            name=DatasetEnum.DIGITS,
            path="experiments/data/datasets/digits",
        ),
        models=models,
        run_hessian_analysis=True,
    )
    return config


def base_sweep():
    """Recreate your original sweep in a cleaner way."""
    models = []

    # Learning rates and weight decays to sweep
    lrs = [5e-4, 1e-3, 2e-3, 1e-2]
    wds = [0.0, 1e-4, 1e-3, 1e-2]
    # lrs = [1e-3]
    # wds = [0.0]

    # MLP architectures
    mlp_configs = [
        [16],
        [16] * 4,
        [16] * 8,
        [32],
        [32] * 4,
        [32] * 8,
        [64],
        [64] * 4,
        [64] * 8,
    ]

    for hidden_dim in mlp_configs:
        models.extend(
            create_hyperparameter_sweep(
                ModelArchitecture.MLP,
                hidden_dim,
                learning_rates=lrs,
                weight_decays=wds,
                epochs=500,
                batch_size=128,
            )
        )

    # MLPSwiGLU architectures
    swiglu_configs = [
        [(5, 5, 5)],
        [(5, 5, 5)] * 4,
        [(5, 5, 5)] * 8,
        [(10, 10, 10)],
        [(10, 10, 10)] * 4,
        [(10, 10, 10)] * 8,
        [(21, 21, 21)],
        [(21, 21, 21)] * 4,
        [(21, 21, 21)] * 8,
    ]

    for hidden_dim in swiglu_configs:
        models.extend(
            create_hyperparameter_sweep(
                ModelArchitecture.MLPSWIGLU,
                hidden_dim,
                learning_rates=lrs,
                weight_decays=wds,
                epochs=500,
                batch_size=32,
            )
        )

    config = ExperimentConfig(
        experiment_name="",
        seed=42,
        base_output_dir="experiments/sweep_1/data/",
        dataset=DatasetConfig(
            name=DatasetEnum.DIGITS,
            path="experiments/sweep_1/data/datasets/digits",
        ),
        models=models,
        run_hessian_analysis=True,
        hessian_analysis=HessianAnalysisConfig(
            accuracy_threshold=0.40,
            base_collector_dir="experiments/sweep_1/data/collector",
            base_ekfac_dir="experiments/sweep_1/data/ekfac",
            computation_config=HessianComputationConfig(
                approximators=[
                    HessianApproximator.GNH,
                    HessianApproximator.BLOCK_HESSIAN,
                    HessianApproximator.FIM,
                    HessianApproximator.KFAC,
                    HessianApproximator.EKFAC,
                ],
                comparison_references=[
                    HessianApproximator.EXACT,
                    HessianApproximator.GNH,
                ],
                computation_types=[
                    ComputationType.MATRIX,
                    ComputationType.HVP,
                    ComputationType.IHVP,
                ],
                damping_strategy=DampingStrategy.AUTO_MEAN_EIGENVALUE,
                damping=0.1,
            ),
        ),
    )

    return config


if __name__ == "__main__":
    # Generate your original sweep config
    config = base_sweep()

    # Custom representer to handle enums as strings
    def enum_representer(dumper, data):
        return dumper.represent_str(data.value)

    # Register enum representers
    yaml.add_representer(ModelArchitecture, enum_representer)
    yaml.add_representer(OptimizerType, enum_representer)
    yaml.add_representer(LOSS_TYPE, enum_representer)
    yaml.add_representer(DatasetEnum, enum_representer)
    yaml.add_representer(HessianApproximator, enum_representer)
    yaml.add_representer(ComputationType, enum_representer)
    yaml.add_representer(DampingStrategy, enum_representer)
    yaml.add_representer(VectorSamplingMethod, enum_representer)
    yaml.add_representer(VectorMetric, enum_representer)
    yaml.add_representer(FullMatrixMetric, enum_representer)

    with open("experiments/sweep_1/configs/base_sweep.yaml", "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)

    print(f"Generated config with {len(config.models)} models")
    print("Saved to experiments/sweep_1/configs/base_sweep.yaml")
