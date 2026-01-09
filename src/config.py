from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric

# -----------------------------------------------------------------------------
# Enums for controlled choices
# -----------------------------------------------------------------------------


class LossType(str, Enum):
    """Types of loss functions."""

    MSE = "mse"
    CROSS_ENTROPY = "cross_entropy"


class ModelArchitecture(str, Enum):
    """Available model architectures."""

    MLP = "mlp"
    MLPSWIGLU = "mlp_swiglu"
    LINEAR = "linear"


class OptimizerType(str, Enum):
    """Available optimizers."""

    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    SGD_SCHEDULE_COSINE = "sgd_schedule_cosine"


class HessianApproximator(str, Enum):
    """Available Hessian approximation methods."""

    EXACT = "exact"
    KFAC = "kfac"
    EKFAC = "ekfac"
    GNH = "gnh"
    FIM = "fim"
    BLOCK_FIM = "block_fim"
    BLOCK_HESSIAN = "block_hessian"

    @staticmethod
    def get_approximator_list_except_exact() -> List["HessianApproximator"]:
        """Get a list of all available Hessian approximators (except EXACT)."""
        return [
            HessianApproximator.KFAC,
            HessianApproximator.EKFAC,
            HessianApproximator.GNH,
            HessianApproximator.FIM,
            HessianApproximator.BLOCK_FIM,
            HessianApproximator.BLOCK_HESSIAN,
        ]


class ComputationType(str, Enum):
    """Types of Hessian computations."""

    MATRIX = "matrix"
    HVP = "hvp"
    IHVP = "ihvp"


class DampingStrategy(str, Enum):
    """Strategy for computing the damping parameter."""

    FIXED = "fixed"
    AUTO_MEAN_EIGENVALUE = "auto_mean_eigenvalue"
    AUTO_MEAN_EIGENVALUE_CORRECTION = "auto_mean_eigenvalue_correction"


class DatasetEnum(str, Enum):
    """Available datasets."""

    DIGITS = "digits"
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    FASHION_MNIST = "fashion_mnist"
    SKLEARN_DIGITS = "sklearn_digits"
    ENERGY = "energy"
    CONCRETE = "concrete"
    CANCER = "cancer"


class VectorSamplingMethod(str, Enum):
    """Methods for sampling vectors for HVP computation."""

    GRADIENTS = "gradients"
    RANDOM = "random"


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and splitting."""

    name: DatasetEnum
    path: str
    test_size: float = 0.1
    store_on_disk: bool = True

    def __post_init__(self):
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: OptimizerType = OptimizerType.ADAMW
    epochs: int = 500
    batch_size: int = 128

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class ModelConfig:
    """Configuration for a single model including its training setup."""

    # Model architecture
    architecture: ModelArchitecture

    # Model dimensions
    input_dim: int = field(default=0)
    hidden_dim: List[int] | List[Tuple[int, int, int]] | None = field(default=None)
    output_dim: int = field(default=0)

    # Loss function
    loss: LossType = LossType.CROSS_ENTROPY

    # Training configuration (embedded)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Storage
    directory: Optional[str] = None

    skip_existing: bool = True

    def get_model_base_name(self) -> str:
        """Get the base name from the directory path."""
        if self.directory is None:
            raise ValueError("Model directory is None.")
        return os.path.basename(self.directory)

    def get_model_display_name(self) -> str:
        """Get a human-readable display name for the model."""
        return f"{self.architecture.value}_hidden{self.hidden_dim}"

    def serialize(self) -> dict:
        """Serialize the ModelConfig to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ModelConfig:
        data = dict(data)
        if isinstance(data.get("architecture"), str):
            data["architecture"] = ModelArchitecture(data["architecture"])

        if isinstance(data.get("loss"), str):
            data["loss"] = LossType(data["loss"])

        if isinstance(data.get("training"), dict):
            data["training"] = TrainingConfig(**data["training"])

        return cls(**data)


@dataclass
class VectorAnalysisConfig:
    """Configuration for HVP/IHVP analysis."""

    num_samples: int = 1000
    sampling_method: VectorSamplingMethod = VectorSamplingMethod.GRADIENTS
    metrics: List[VectorMetric] = field(
        default_factory=lambda: VectorMetric.all_metrics()
    )


@dataclass
class MatrixAnalysisConfig:
    """Configuration for Hessian matrix analysis."""

    metrics: List[FullMatrixMetric] = field(
        default_factory=lambda: FullMatrixMetric.all_metrics()
    )


@dataclass
class HessianComputationConfig:
    """Configuration specifying which Hessian computations to perform."""

    approximators: List[HessianApproximator] = field(
        default_factory=lambda: HessianApproximator.get_approximator_list_except_exact()
    )
    comparison_references: List[HessianApproximator] = field(
        default_factory=lambda: [HessianApproximator.EXACT, HessianApproximator.GNH]
    )
    computation_types: List[ComputationType] = field(
        default_factory=lambda: [
            ComputationType.MATRIX,
            ComputationType.HVP,
            ComputationType.IHVP,
        ]
    )
    damping: float = 0.1
    damping_strategy: DampingStrategy = DampingStrategy.AUTO_MEAN_EIGENVALUE


@dataclass
class HessianAnalysisConfig:
    """Configuration for Hessian approximation and analysis."""

    # Analysis settings
    vector_config: VectorAnalysisConfig = field(default_factory=VectorAnalysisConfig)
    matrix_config: MatrixAnalysisConfig = field(default_factory=MatrixAnalysisConfig)
    computation_config: HessianComputationConfig = field(
        default_factory=HessianComputationConfig
    )

    # Storage
    results_output_dir: str = "experiments/results"


@dataclass
class TrainingExperimentConfig:
    """Top-level training experiment configuration (no Hessian analysis)."""

    # Experiment identification
    experiment_name: str = "training_experiment"
    base_output_dir: str = "experiments"
    seed: int = 42

    # Dataset
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            name=DatasetEnum.DIGITS, path="experiments/data/datasets/digits"
        )
    )

    # Models to train (list of individual model configs)
    models: List[ModelConfig] = field(default_factory=list)

    # Model selection criteria
    selection_metric: str = "val_accuracy"  # or "val_loss"
    selection_minimize: bool = False  # False for accuracy, True for loss

    def get_results_dir(self) -> str:
        return os.path.join(
            self.base_output_dir,
            "results",
            self.experiment_name,
            self.dataset.name.value,
        )

    def get_models_base_dir(self) -> str:
        return os.path.join(
            self.base_output_dir,
            "models",
            self.experiment_name,
            self.dataset.name.value,
        )

    def get_dataset_dir(self) -> str:
        return os.path.join(self.base_output_dir, self.experiment_name, "datasets")


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    # Experiment identification
    experiment_name: str = "experiment"
    seed: int = 42

    # Dataset
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            name=DatasetEnum.DIGITS, path="experiments/data/datasets/digits"
        )
    )

    # List of model_directories with model checkpoints and model definition
    models: List[str] = field(default_factory=list)

    # Which different approaches to compare and analyze
    hessian_analysis: HessianAnalysisConfig = field(
        default_factory=HessianAnalysisConfig
    )
