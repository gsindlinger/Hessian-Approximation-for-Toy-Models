from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric

# -----------------------------------------------------------------------------
# Enums for controlled choices
# -----------------------------------------------------------------------------


class LOSS_TYPE(str, Enum):
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


class ComparisonReference(str, Enum):
    """Reference methods for Hessian comparisons."""

    EXACT = "exact"
    GNH = "gnh"


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


class Stage(str, Enum):
    """Execution stage."""

    TRAIN = "train"
    HESSIAN = "hessian"
    BOTH = "both"


class Datasets(str, Enum):
    """Available datasets."""

    DIGITS = "digits"
    MNIST = "mnist"
    CIFAR10 = "cifar10"


class VectorSamplingMethod(str, Enum):
    """Methods for sampling vectors for HVP computation."""

    GRADIENTS = "gradients"
    RANDOM = "random"


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and splitting."""

    name: Datasets
    """Name of the dataset to use."""

    path: str
    """Path to the dataset on disk."""

    test_size: float = 0.1
    """Fraction of the dataset reserved for validation."""

    split_seed: int = 42
    """Random seed used for train/validation splitting."""

    store_on_disk: bool = True
    """Whether to cache the dataset on disk."""

    def __post_init__(self):
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")


@dataclass
class ModelConfig:
    """Configuration for a single layer stack."""

    architecture: ModelArchitecture
    """Model architecture type."""

    hidden_dims: List[int] | List[Tuple[int, int, int]] | None
    """Hidden layer dimensions, e.g. [64, 64]."""

    init_seed: int = 42
    """Random seed for model initialization."""

    directory: str | None = None
    """Directory to save or load the model."""

    skip_existing: bool = True
    """Whether to skip training if checkpoints already exist."""

    loss: LOSS_TYPE = LOSS_TYPE.MSE
    """Loss function to use for training the model."""

    training_config: TrainingConfig | None = None
    """Training configuration for the model, if applicable."""

    def get_model_base_name(self) -> str:
        # final directory name as model_key, i.e. {model_name}_{hash}
        if self.directory is None:
            raise ValueError("Model directory is None.")
        model_key = os.path.basename(self.directory)
        return model_key


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    """Model architecture configuration."""
    learning_rate: float = 1e-3
    """Learning rate for the optimizer."""

    weight_decay: float = 0.0
    """Weight decay (L2 regularization) coefficient."""

    optimizer: OptimizerType = OptimizerType.ADAMW
    """Optimizer used for training."""

    epochs: int = 500
    """Number of training epochs."""

    batch_size: int = 32
    """Mini-batch size."""

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


# -----------------------------------------------------------------------------
# Hyperparameter Search Configuration
# -----------------------------------------------------------------------------


@dataclass
class HyperparameterConfig:
    """Grid search configuration for optimizer hyperparameters."""

    learning_rates: list[float] = field(default_factory=lambda: [1e-3])
    """Learning rates to use or sweep over."""

    weight_decays: list[float] = field(default_factory=lambda: [0.0])
    """Weight decay values to use or sweep over.
    If None, weight decay is disabled (equivalent to [0.0])."""

    def __post_init__(self):
        if any(lr <= 0 for lr in self.learning_rates):
            raise ValueError("All learning rates must be positive")
        if any(wd < 0 for wd in self.weight_decays):
            raise ValueError("Weight decay must be non-negative")

    def is_grid_search(self) -> bool:
        """Return True if more than one hyperparameter configuration exists."""
        return len(self.learning_rates) > 1 or len(self.weight_decays) > 1

    def num_configurations(self) -> int:
        """Return the total number of hyperparameter combinations."""
        return len(self.learning_rates) * len(self.weight_decays)


# -----------------------------------------------------------------------------
# Damping Configuration
# -----------------------------------------------------------------------------


@dataclass
class DampingConfig:
    """Configuration for damping in Hessian computations."""

    strategy: DampingStrategy = DampingStrategy.AUTO_MEAN_EIGENVALUE
    """Strategy used to compute the damping parameter."""

    factor: float = 0.1
    """Scaling factor applied to the selected damping strategy."""

    def __post_init__(self):
        if self.factor <= 0:
            raise ValueError("factor must be positive")

    def get_damping_value(
        self,
        mean_eigenvalue: Optional[float] = None,
        mean_correction: Optional[float] = None,
    ) -> float:
        """Compute the damping value according to the selected strategy."""
        if self.strategy == DampingStrategy.FIXED:
            return self.factor
        if self.strategy == DampingStrategy.AUTO_MEAN_EIGENVALUE:
            if mean_eigenvalue is None:
                raise ValueError("mean_eigenvalue required")
            return self.factor * mean_eigenvalue
        if self.strategy == DampingStrategy.AUTO_MEAN_EIGENVALUE_CORRECTION:
            if mean_correction is None:
                raise ValueError("mean_correction required")
            return self.factor * mean_correction
        raise ValueError(f"Unknown damping strategy: {self.strategy}")


# -----------------------------------------------------------------------------
# Hessian Analysis Configuration
# -----------------------------------------------------------------------------


@dataclass
class CollectorConfig:
    """Configuration for gradient and activation collection."""

    num_pseudo_target_runs: int = 1
    """Number of independent pseudo-target runs."""

    pseudo_target_seeds: List[int] = field(default_factory=lambda: [42])
    """Optional random seeds for pseudo-target generation."""

    collector_output_dirs: List[str] = field(
        default_factory=lambda: ["experiments/collector"]
    )
    """Directories for storing collected statistics."""

    try_load_cached: bool = True
    """Whether to reuse cached collector data if available."""

    def __post_init__(self):
        assert len(self.collector_output_dirs) == self.num_pseudo_target_runs, (
            "Length of collector_output_dirs must match num_pseudo_target_runs"
        )
        if self.pseudo_target_seeds is not None:
            if len(self.pseudo_target_seeds) != self.num_pseudo_target_runs:
                raise ValueError(
                    "Length of pseudo_target_seeds must match num_pseudo_target_runs"
                )


@dataclass
class EKFACApproximatorConfig:
    """Configuration specific to EKFAC approximator."""

    directory: str = "experiments/ekfac"
    """Directory to save or load EKFAC data."""

    collector_dir_1: str = "experiments/collector/run1"
    """Directory containing first set of collector statistics."""

    collector_dir_2: str = "experiments/collector/run2"
    """Directory containing second set of collector statistics."""


@dataclass
class ModelContextConfig:
    """Configuration for model context used in Hessian computations."""

    model_config: ModelConfig
    """Model configuration."""

    dataset_config: DatasetConfig
    """Dataset configuration."""


@dataclass
class HessianComputationConfig:
    """Configuration specifying which Hessian computations to perform."""

    approximator_configs: Dict[
        HessianApproximator,
        Union[EKFACApproximatorConfig, ModelContextConfig, CollectorConfig],
    ] = field(default_factory=lambda: {})
    """Types of Hessian computations to perform."""

    comparison_references: Dict[HessianApproximator, ModelContextConfig] = field(
        default_factory=lambda: {}
    )
    """Reference methods used for comparisons."""

    computation_types: Dict[
        ComputationType, VectorAnalysisConfig | MatrixAnalysisConfig
    ] = field(default_factory=lambda: {})

    damping: float = 0.1
    """Fixed damping value to use for Hessian computations."""

    damping_strategy: DampingStrategy = DampingStrategy.AUTO_MEAN_EIGENVALUE
    """Strategy for computing damping value."""

    def __post_init__(self):
        if self.damping_strategy in [
            DampingStrategy.AUTO_MEAN_EIGENVALUE,
            DampingStrategy.AUTO_MEAN_EIGENVALUE_CORRECTION,
        ] and not (
            HessianApproximator.EKFAC in self.approximator_configs
            or HessianApproximator.KFAC in self.approximator_configs
        ):
            raise ValueError(
                "Damping strategy requires EKFAC or KFAC approximator to compute mean eigenvalue."
            )


@dataclass
class VectorAnalysisConfig:
    """Configuration for HVP analysis."""

    num_samples: int = 1000
    """Number of HVP samples to compute."""

    sampling_method: VectorSamplingMethod = VectorSamplingMethod.GRADIENTS
    """Method for sampling vectors for HVP / IHVP computation."""

    seed: int = 42
    """Random seed for vector sampling."""

    metrics: List[VectorMetric] = field(
        default_factory=lambda: VectorMetric.all_metrics()
    )
    """Metrics to compute during Hessian vector analysis."""


@dataclass
class MatrixAnalysisConfig:
    """Configuration for Hessian matrix analysis."""

    metrics: List[FullMatrixMetric] = field(
        default_factory=lambda: FullMatrixMetric.all_metrics()
    )


@dataclass
class HessianAnalysisConfig:
    """Configuration for Hessian approximation and analysis."""

    input_model_dir: Optional[str] = None
    """Path to a directory containing a single trained model."""

    model_filter: Optional[str] = None
    """Regex pattern for filtering models from the manifest."""

    collector_config: CollectorConfig = field(default_factory=CollectorConfig)
    """Configuration for gradient and activation collection."""

    computation_config: HessianComputationConfig = field(
        default_factory=HessianComputationConfig
    )
    """Hessian computation configuration."""

    damping_config: DampingConfig = field(default_factory=DampingConfig)
    """Damping configuration."""

    hessian_output_dir: str = "experiments/hessian"
    """Directory where Hessian data is stored."""

    results_output_dir: str = "experiments/results"
    """Directory where analysis results are stored."""

    analysis_seed: int = 42
    """Random seed for Hessian analysis."""
