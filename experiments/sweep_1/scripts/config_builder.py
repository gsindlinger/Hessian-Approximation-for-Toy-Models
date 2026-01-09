"""
Simplified config builder that generates training-only configurations.
Uses the TrainingExperimentConfig instead of full ExperimentConfig.
"""

import logging
from dataclasses import asdict
from typing import List

import yaml
from simple_parsing import ArgumentParser

from src.config import (
    ComputationType,
    DampingStrategy,
    DatasetConfig,
    DatasetEnum,
    ExperimentConfig,
    HessianAnalysisConfig,
    HessianApproximator,
    HessianComputationConfig,
    LossType,
    MatrixAnalysisConfig,
    ModelArchitecture,
    ModelConfig,
    OptimizerType,
    TrainingConfig,
    TrainingExperimentConfig,
    VectorAnalysisConfig,
    VectorSamplingMethod,
)
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric

logger = logging.getLogger(__name__)


class ModelConfigBuilder:
    """Builder pattern for creating ModelConfig with training settings."""

    def __init__(self, architecture: ModelArchitecture, hidden_dim):
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.loss = LossType.CROSS_ENTROPY
        self.init_seed = 42
        self.skip_existing = True

        # Training defaults
        self.learning_rate = 1e-3
        self.weight_decay = 0.0
        self.optimizer = OptimizerType.ADAMW
        self.epochs = 500
        self.batch_size = 32

    def with_loss(self, loss: LossType):
        self.loss = loss
        return self

    def with_seed(self, seed: int):
        self.init_seed = seed
        return self

    def with_training(
        self,
        learning_rate=None,
        weight_decay=None,
        optimizer=None,
        epochs=None,
        batch_size=None,
        input_dim=None,
        output_dim=None,
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
        if input_dim is not None:
            self.input_dim = input_dim
        if output_dim is not None:
            self.output_dim = output_dim
        return self

    def build(self) -> ModelConfig:
        return ModelConfig(
            architecture=self.architecture,
            hidden_dim=self.hidden_dim,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            loss=self.loss,
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
    optimizer: OptimizerType = OptimizerType.ADAMW,
    input_dim: int = 0,
    output_dim: int = 0,
    **kwargs,
) -> List[ModelConfig]:
    """Create a grid of models with different hyperparameters."""
    models = []
    for lr in learning_rates:
        for wd in weight_decays:
            model = (
                ModelConfigBuilder(architecture, hidden_dim)
                .with_training(
                    learning_rate=lr,
                    weight_decay=wd,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    optimizer=optimizer,
                    **kwargs,
                )
                .build()
            )
            models.append(model)
    return models


def digits_sweep_simple():
    """Generate a simple training sweep configuration."""
    models = []

    # Hyperparameter grid
    lrs = [1e-3]
    wds = [0.0]

    # MLP architectures
    mlp_configs = [
        [16],
    ]

    for hidden_dim in mlp_configs:
        models.extend(
            create_hyperparameter_sweep(
                ModelArchitecture.MLP,
                hidden_dim,
                learning_rates=lrs,
                weight_decays=wds,
                epochs=100,
                batch_size=128,
                input_dim=64,
                output_dim=10,
            )
        )

    # MLPSwiGLU architectures
    swiglu_configs = [
        [(5, 5, 5)],
    ]
    for hidden_dim in swiglu_configs:
        models.extend(
            create_hyperparameter_sweep(
                ModelArchitecture.MLPSWIGLU,
                hidden_dim,
                learning_rates=lrs,
                weight_decays=wds,
                epochs=100,
                batch_size=32,
                input_dim=64,
                output_dim=10,
            )
        )

    # Create training-only config
    config = TrainingExperimentConfig(
        experiment_name="debug_sweep",
        seed=42,
        base_output_dir="experiments/sweep_1/data/",
        dataset=DatasetConfig(
            name=DatasetEnum.DIGITS,
            path="experiments/datasets/digits",
        ),
        models=models,
        selection_metric="val_accuracy",
        selection_minimize=False,
    )

    return config


def digits_sweep():
    """Generate training sweep configuration."""
    models = []

    # Hyperparameter grid
    lrs = [5e-4, 1e-3, 2e-3, 1e-2]
    wds = [0.0, 1e-4, 1e-3, 1e-2]

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
                input_dim=64,
                output_dim=10,
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
                input_dim=64,
                output_dim=10,
            )
        )

    # Create training-only config
    config = TrainingExperimentConfig(
        experiment_name="",
        seed=42,
        base_output_dir="experiments/sweep_1/data/",
        dataset=DatasetConfig(
            name=DatasetEnum.DIGITS,
            path="experiments/datasets/digits",
        ),
        models=models,
        selection_metric="val_accuracy",
        selection_minimize=False,
    )

    return config


def concrete_sweep():
    """Generate optimized training sweep configuration for Concrete dataset."""
    models = []

    # Optimized hyperparameter grid
    lrs = [
        1e-4,
        5e-4,
        1e-3,
        5e-3,
    ]  # More conservative for regression
    wds = [0.0, 1e-5, 1e-4, 5e-4, 1e-3]  # More regularization options

    # MLP architectures - focused on 1-4 layers (simpler is often better for tabular)
    mlp_configs = [
        [16],  # Single layer baselines
        2 * [16],
        4 * [16],
        [32],
        2 * [32],
        4 * [32],
        [64],
        2 * [64],
        4 * [64],
        [128],
        2 * [128],
    ]

    for hidden_dim in mlp_configs:
        models.extend(
            create_hyperparameter_sweep(
                ModelArchitecture.MLP,
                hidden_dim,
                learning_rates=lrs,
                weight_decays=wds,
                epochs=500,
                batch_size=32,
                input_dim=8,
                output_dim=1,
            )
        )

    # MLPSwiGLU architectures - also simplified
    swiglu_configs = [
        [(8, 8, 8)],
        [(8, 8, 8)] * 2,
        [(8, 8, 8)] * 4,
        [(16, 16, 16)],
        [(16, 16, 16)] * 2,
        [(16, 16, 16)] * 4,
        [(32, 32, 32)],
        [(32, 32, 32)] * 2,
        [(32, 32, 32)] * 4,
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
                input_dim=8,
                output_dim=1,
            )
        )

    # Apply MSE loss to all models
    for model in models:
        model.loss = LossType.MSE

    config = TrainingExperimentConfig(
        experiment_name="",
        seed=42,
        base_output_dir="experiments/sweep_1/data/",
        dataset=DatasetConfig(
            name=DatasetEnum.CONCRETE,
            path="experiments/datasets/concrete",
            test_size=0.1,
        ),
        models=models,
        selection_metric="val_loss",
        selection_minimize=True,
    )

    return config


def sweep_concrete_simple():
    config = concrete_sweep()
    config.experiment_name = "concrete_simple_sweep"

    models = []

    lrs = [1e-3]
    wds = [1e-4]

    mlp_configs = [
        2 * [64],
    ]
    mlp_swiglu_configs = [
        [(16, 16, 16)] * 2,
    ]

    for hidden_dim in mlp_configs:
        models.extend(
            create_hyperparameter_sweep(
                ModelArchitecture.MLP,
                hidden_dim,
                learning_rates=lrs,
                weight_decays=wds,
                epochs=500,
                batch_size=32,
                input_dim=8,
                output_dim=1,
            )
        )

    for hidden_dim in mlp_swiglu_configs:
        models.extend(
            create_hyperparameter_sweep(
                ModelArchitecture.MLPSWIGLU,
                hidden_dim,
                learning_rates=lrs,
                weight_decays=wds,
                epochs=500,
                batch_size=64,
                input_dim=8,
                output_dim=1,
            )
        )

    # Apply MSE loss to all models
    for model in models:
        model.loss = LossType.MSE

    config.models = models

    return config


def sweep_energy():
    config = concrete_sweep()
    config.experiment_name = "energy_sweep"
    config.dataset = DatasetConfig(
        name=DatasetEnum.ENERGY,
        path="experiments/datasets/energy",
        test_size=0.1,
    )
    for model in config.models:
        model.input_dim = 8
        model.output_dim = 2
        model.loss = LossType.MSE

    return config


def sweep_energy_simple():
    config = sweep_energy()
    config.experiment_name = "energy_simple_sweep"

    models = []

    lrs = [1e-3]
    wds = [1e-4]

    mlp_configs = [
        2 * [64],
    ]
    mlp_swiglu_configs = [
        [(16, 16, 16)] * 2,
    ]

    for hidden_dim in mlp_configs:
        models.extend(
            create_hyperparameter_sweep(
                ModelArchitecture.MLP,
                hidden_dim,
                learning_rates=lrs,
                weight_decays=wds,
                epochs=500,
                batch_size=32,
                input_dim=8,
                output_dim=2,
            )
        )

    for hidden_dim in mlp_swiglu_configs:
        models.extend(
            create_hyperparameter_sweep(
                ModelArchitecture.MLPSWIGLU,
                hidden_dim,
                learning_rates=lrs,
                weight_decays=wds,
                epochs=500,
                batch_size=64,
                input_dim=8,
                output_dim=2,
            )
        )

    # Apply MSE loss to all models
    for model in models:
        model.loss = LossType.MSE

    config.models = models

    return config


def normalize_for_yaml(obj):
    """Normalize objects for YAML serialization."""
    if isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, list):
        return [normalize_for_yaml(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: normalize_for_yaml(v) for k, v in obj.items()}
    return obj


def register_enum_representers():
    """Register custom YAML representers for enums."""

    def enum_representer(dumper, data):
        return dumper.represent_str(data.value)

    # Register enum representers
    yaml.add_representer(ModelArchitecture, enum_representer)
    yaml.add_representer(OptimizerType, enum_representer)
    yaml.add_representer(LossType, enum_representer)
    yaml.add_representer(DatasetEnum, enum_representer)
    yaml.add_representer(HessianApproximator, enum_representer)
    yaml.add_representer(ComputationType, enum_representer)
    yaml.add_representer(DampingStrategy, enum_representer)
    yaml.add_representer(VectorSamplingMethod, enum_representer)
    yaml.add_representer(VectorMetric, enum_representer)
    yaml.add_representer(FullMatrixMetric, enum_representer)


def hessian_analysis_sweep():
    config = ExperimentConfig(
        experiment_name="",
        seed=42,
        models=[],
        hessian_analysis=HessianAnalysisConfig(
            vector_config=VectorAnalysisConfig(
                num_samples=1000,
                sampling_method=VectorSamplingMethod.GRADIENTS,
                metrics=VectorMetric.all_metrics(),
            ),
            matrix_config=MatrixAnalysisConfig(
                metrics=FullMatrixMetric.all_metrics(
                    exclude=[FullMatrixMetric.CONDITION_NUMBER_LOG_RATIO]
                )
            ),
            computation_config=HessianComputationConfig(
                damping=0.1,
                damping_strategy=DampingStrategy.AUTO_MEAN_EIGENVALUE,
                approximators=HessianApproximator.get_approximator_list_except_exact(),
                computation_types=[
                    ComputationType.MATRIX,
                    ComputationType.HVP,
                    ComputationType.IHVP,
                ],
                comparison_references=[
                    HessianApproximator.EXACT,
                    HessianApproximator.GNH,
                ],
            ),
            results_output_dir="experiments/sweep_1/data/results/hessian_analysis",
        ),
    )
    return config


if __name__ == "__main__":
    # parse args and decide whether training or hessian config

    parser = ArgumentParser(description="Generate experiment configuration.")
    parser.add_argument(
        "--type",
        choices=[
            "hessian",
            "digits",
            "digits_simple",
            "concrete",
            "concrete_simple",
            "energy",
            "energy_simple",
        ],
        default="training",
        help="Type of configuration to generate",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path for the config file"
    )
    args = parser.parse_args()
    match args.type:
        case "digits":
            config = digits_sweep()
            output_path = "experiments/sweep_1/configs/digits_sweep.yaml"
        case "digits_simple":
            config = digits_sweep_simple()
            output_path = (
                "experiments/sweep_1/scripts/debug_scripts/configs/digits_simple.yaml"
            )

        case "hessian":
            config = hessian_analysis_sweep()
            output_path = "experiments/sweep_1/configs/hessian_analysis.yaml"
        case "concrete":
            config = concrete_sweep()
            output_path = "experiments/sweep_1/configs/concrete_sweep.yaml"
        case "concrete_simple":
            config = sweep_concrete_simple()
            output_path = (
                "experiments/sweep_1/scripts/debug_scripts/configs/concrete_simple.yaml"
            )
        case "energy":
            config = sweep_energy()
            output_path = "experiments/sweep_1/configs/energy_sweep.yaml"
        case "energy_simple":
            config = sweep_energy_simple()
            output_path = (
                "experiments/sweep_1/scripts/debug_scripts/configs/energy_simple.yaml"
            )
        case _:
            raise ValueError(f"Unknown config type: {args.type}")

    if args.output is not None:
        output_path = args.output

    register_enum_representers()

    with open(output_path, "w") as f:
        yaml.dump(
            normalize_for_yaml(asdict(config)),
            f,
            default_flow_style=False,
        )

    logger.info(f"Generated training config with {len(config.models)} models")
    logger.info(f"Saved to {output_path}")
    # Print output path for easy access in bash scripts
    print(output_path)
