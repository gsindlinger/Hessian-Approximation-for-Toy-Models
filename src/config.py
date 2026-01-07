from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# -----------------------------------------------------------------------------
# Enums for controlled choices
# -----------------------------------------------------------------------------


class ModelArchitecture(str, Enum):
    """Available model architectures."""

    MLP = "mlp"
    MLP_SWIGLU = "mlp_swiglu"
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


# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and splitting."""

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


# -----------------------------------------------------------------------------
# Model Architecture Configuration
# -----------------------------------------------------------------------------


@dataclass
class LayerConfig:
    """Configuration for a single layer stack."""

    hidden_dims: list[int]
    """Hidden layer dimensions, e.g. [64, 64]."""

    def __post_init__(self):
        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        if any(d <= 0 for d in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive")


@dataclass
class ModelArchitectureConfig:
    """Configuration for model architectures to train."""

    architectures: list[ModelArchitecture] = field(
        default_factory=lambda: [ModelArchitecture.MLP]
    )
    """Which model architectures to train."""

    layer_configs: list[LayerConfig] = field(
        default_factory=lambda: [LayerConfig([16])]
    )
    """Layer configurations to sweep over."""

    init_seed: int = 42
    """Random seed for model initialization."""


# -----------------------------------------------------------------------------
# Hyperparameter Search Configuration
# -----------------------------------------------------------------------------


@dataclass
class HyperparameterConfig:
    """Grid search configuration for optimizer hyperparameters."""

    learning_rates: list[float] = field(default_factory=lambda: [1e-3])
    """Learning rates to use or sweep over."""

    weight_decays: Optional[list[float]] = None
    """Weight decay values to use or sweep over.
    If None, weight decay is disabled (equivalent to [0.0])."""

    def __post_init__(self):
        if self.learning_rates is None:
            raise ValueError("learning_rates must be specified")

        if not isinstance(self.learning_rates, list):
            self.learning_rates = [self.learning_rates]

        if self.weight_decays is None:
            self.weight_decays = [0.0]
        elif not isinstance(self.weight_decays, list):
            self.weight_decays = [self.weight_decays]

        if not self.learning_rates:
            raise ValueError("learning_rates cannot be empty")
        if not self.weight_decays:
            raise ValueError("weight_decays cannot be empty")
        if any(lr <= 0 for lr in self.learning_rates):
            raise ValueError("All learning rates must be positive")
        if any(wd < 0 for wd in self.weight_decays):
            raise ValueError("Weight decay must be non-negative")

    def is_grid_search(self) -> bool:
        """Return True if more than one hyperparameter configuration exists."""
        return (
            len(self.learning_rates) > 1
            or self.weight_decays is not None
            and len(self.weight_decays) > 1
        )

    def num_configurations(self) -> int:
        """Return the total number of hyperparameter combinations."""
        return len(self.learning_rates) * len(self.weight_decays)


# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_config: ModelArchitectureConfig = field(
        default_factory=ModelArchitectureConfig
    )
    """Model architecture configuration."""

    hyperparam_grid: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    """Hyperparameter grid for training."""

    optimizer: OptimizerType = OptimizerType.ADAMW
    """Optimizer used for training."""

    epochs: int = 500
    """Number of training epochs."""

    batch_size: int = 32
    """Mini-batch size."""

    min_accuracy_threshold: float = 0.4
    """Minimum validation accuracy required for Hessian analysis."""

    model_output_dir: str = "experiments/models"
    """Directory where trained models are saved."""

    manifest_output_path: str = "experiments/training_manifest.json"
    """Path where the training manifest JSON is written."""

    skip_existing: bool = True
    """Whether to skip training if checkpoints already exist."""

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.min_accuracy_threshold <= 1.0:
            raise ValueError("min_accuracy_threshold must be in [0, 1]")


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

    num_gradient_samples: int = 1000
    """Number of samples used for gradient estimation."""

    num_pseudo_target_runs: int = 2
    """Number of independent pseudo-target runs."""

    pseudo_target_seeds: Optional[list[int]] = None
    """Optional random seeds for pseudo-target generation."""

    collector_output_dir: str = "experiments/collector"
    """Directory for storing collected statistics."""

    try_load_cached: bool = True
    """Whether to reuse cached collector data if available."""


@dataclass
class HessianComputationConfig:
    """Configuration specifying which Hessian computations to perform."""

    approximators: list[HessianApproximator] = field(
        default_factory=lambda: [
            HessianApproximator.EKFAC,
            HessianApproximator.KFAC,
            HessianApproximator.GNH,
        ]
    )
    """Hessian approximation methods to compute."""

    comparison_references: list[ComparisonReference] = field(
        default_factory=lambda: [ComparisonReference.EXACT, ComparisonReference.GNH]
    )
    """Reference methods used for comparisons."""

    computation_types: list[ComputationType] = field(
        default_factory=lambda: [
            ComputationType.MATRIX,
            ComputationType.HVP,
            ComputationType.IHVP,
        ]
    )
    """Types of Hessian computations to perform."""


@dataclass
class HessianAnalysisConfig:
    """Configuration for Hessian approximation and analysis."""

    input_manifest_path: Optional[str] = None
    """Path to a training manifest JSON file."""

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

    def __post_init__(self):
        if self.input_manifest_path is None and self.input_model_dir is None:
            raise ValueError(
                "Either input_manifest_path or input_model_dir must be set"
            )
        if self.input_manifest_path and self.input_model_dir:
            raise ValueError(
                "Only one of input_manifest_path or input_model_dir allowed"
            )


# -----------------------------------------------------------------------------
# Global Experiment Configuration
# -----------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Top-level configuration for training and Hessian experiments."""

    stage: Stage = Stage.BOTH
    """Which stage of the pipeline to execute."""

    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(path=""))
    """Dataset configuration."""

    training: Optional[TrainingConfig] = None
    """Training configuration."""

    hessian: Optional[HessianAnalysisConfig] = None
    """Hessian analysis configuration."""

    global_seed: int = 42
    """Global random seed."""

    experiment_name: str = "experiment"
    """Human-readable experiment identifier."""

    root_output_dir: str = "experiments"
    """Root directory for all experiment outputs."""

    verbose: bool = True
    """Whether to enable verbose logging."""

    def __post_init__(self):
        if self.stage in [Stage.TRAIN, Stage.BOTH] and self.training is None:
            raise ValueError("training config required")
        if self.stage in [Stage.HESSIAN, Stage.BOTH] and self.hessian is None:
            raise ValueError("hessian config required")
        if not self.dataset.path:
            raise ValueError("dataset.path must be specified")
        if self.stage == Stage.BOTH and self.training and self.hessian:
            if self.hessian.input_manifest_path is None:
                self.hessian.input_manifest_path = self.training.manifest_output_path

    def get_training_config(self) -> TrainingConfig:
        """Return the training configuration."""
        if self.training is None:
            raise ValueError("Training config not available")
        return self.training

    def get_hessian_config(self) -> HessianAnalysisConfig:
        """Return the Hessian analysis configuration."""
        if self.hessian is None:
            raise ValueError("Hessian config not available")
        return self.hessian
