"""
Model registry for managing trained model manifests.

This module provides functionality to:
- Save and load training manifests
- Query models by filters
- Track model metadata and performance
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from src.config import ModelArchitecture, OptimizerType

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class ModelEntry:
    """Entry for a single trained model in the manifest."""

    # Identification
    model_name: str
    model_type: ModelArchitecture
    model_hash: str

    # Architecture
    hidden_layers: list[int] | list[tuple[int, int, int]]
    num_params: int

    # Training configuration
    learning_rate: float
    weight_decay: float
    optimizer: OptimizerType
    epochs: int
    seed: int

    # Performance metrics
    train_loss: float
    val_loss: float
    val_accuracy: float

    # Paths
    model_dir: str
    checkpoint_path: str

    # Optional metadata
    training_time_seconds: Optional[float] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["model_type"] = self.model_type.value
        d["optimizer"] = self.optimizer.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ModelEntry":
        data = dict(data)
        data["model_type"] = ModelArchitecture(data["model_type"])
        data["optimizer"] = OptimizerType(data["optimizer"])
        return cls(**data)

    def matches_filter(self, pattern: Optional[str] = None) -> bool:
        """Check if model matches a regex filter pattern."""
        if pattern is None:
            return True
        try:
            return re.search(pattern, self.model_name) is not None
        except re.error:
            logger.warning(f"Invalid regex pattern: {pattern}")
            return False

    def meets_threshold(self, min_accuracy: float) -> bool:
        """Check if model meets minimum accuracy threshold."""
        return self.val_accuracy >= min_accuracy


@dataclass
class TrainingManifest:
    """Manifest containing all trained models from an experiment."""

    # Metadata
    experiment_name: str
    timestamp: str
    seed: int
    dataset_path: str

    # Hyperparameter grid info
    hyperparameter_grid: dict

    # Model entries
    models: list[ModelEntry]

    # Best models per architecture
    best_models: dict[str, str]  # architecture -> model_name

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "dataset_path": self.dataset_path,
            "hyperparameter_grid": self.hyperparameter_grid,
            "models": [m.to_dict() for m in self.models],
            "best_models": self.best_models,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingManifest":
        """Create from dictionary."""
        models = [ModelEntry.from_dict(m) for m in data["models"]]
        return cls(
            experiment_name=data["experiment_name"],
            timestamp=data["timestamp"],
            seed=data["seed"],
            dataset_path=data["dataset_path"],
            hyperparameter_grid=data["hyperparameter_grid"],
            models=models,
            best_models=data["best_models"],
        )

    def filter_models(
        self,
        pattern: Optional[str] = None,
        min_accuracy: Optional[float] = None,
        model_types: Optional[list[str]] = None,
    ) -> list[ModelEntry]:
        """Filter models based on criteria."""
        filtered = self.models

        if pattern is not None:
            filtered = [m for m in filtered if m.matches_filter(pattern)]

        if min_accuracy is not None:
            filtered = [m for m in filtered if m.meets_threshold(min_accuracy)]

        if model_types is not None:
            filtered = [m for m in filtered if m.model_type in model_types]

        return filtered

    def get_best_models(self) -> list[ModelEntry]:
        """Get all best models (one per architecture)."""
        best_names = set(self.best_models.values())
        return [m for m in self.models if m.model_name in best_names]

    def get_model_by_name(self, model_name: str) -> Optional[ModelEntry]:
        """Get a specific model by name."""
        for model in self.models:
            if model.model_name == model_name:
                return model
        return None

    def add_model(self, model: ModelEntry):
        """Add a model entry to the manifest."""
        self.models.append(model)

        # Update best model if this is better
        if model.model_type not in self.best_models:
            self.best_models[model.model_type] = model.model_name
        else:
            current_best = self.get_model_by_name(self.best_models[model.model_type])
            if current_best and model.val_loss < current_best.val_loss:
                self.best_models[model.model_type] = model.model_name
                logger.info(
                    f"New best {model.model_type}: {model.model_name} "
                    f"(val_loss={model.val_loss:.6f})"
                )

    def summary_stats(self) -> dict:
        """Get summary statistics of all models."""
        if not self.models:
            return {}

        return {
            "total_models": len(self.models),
            "model_types": list(set(m.model_type for m in self.models)),
            "avg_val_accuracy": sum(m.val_accuracy for m in self.models)
            / len(self.models),
            "best_val_accuracy": max(m.val_accuracy for m in self.models),
            "worst_val_accuracy": min(m.val_accuracy for m in self.models),
            "avg_val_loss": sum(m.val_loss for m in self.models) / len(self.models),
            "best_val_loss": min(m.val_loss for m in self.models),
        }


# -----------------------------------------------------------------------------
# Manifest I/O
# -----------------------------------------------------------------------------


def save_manifest(manifest: TrainingManifest, path: str | Path):
    """Save training manifest to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    logger.info(f"Saved manifest with {len(manifest.models)} models to {path}")


def load_manifest(path: str | Path) -> TrainingManifest:
    """Load training manifest from JSON file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    manifest = TrainingManifest.from_dict(data)
    logger.info(f"Loaded manifest with {len(manifest.models)} models from {path}")

    return manifest


def create_empty_manifest(
    experiment_name: str,
    timestamp: str,
    seed: int,
    dataset_path: str,
    hyperparameter_grid: dict,
) -> TrainingManifest:
    """Create an empty manifest for a new experiment."""
    return TrainingManifest(
        experiment_name=experiment_name,
        timestamp=timestamp,
        seed=seed,
        dataset_path=dataset_path,
        hyperparameter_grid=hyperparameter_grid,
        models=[],
        best_models={},
    )


def list_models_table(models: list[ModelEntry], max_rows: int = 20):
    """Print models in a formatted table."""
    if not models:
        logger.info("No models to display.")
        return

    logger.info(f"\n{len(models)} models:")
    logger.info(f"{'Model Name':<50} {'Type':<12} {'Val Acc':<10} {'Val Loss':<10}")
    logger.info("-" * 82)

    for i, model in enumerate(models[:max_rows]):
        logger.info(
            f"{model.model_name:<50} "
            f"{model.model_type:<12} "
            f"{model.val_accuracy:<10.4f} "
            f"{model.val_loss:<10.6f}"
        )

    if len(models) > max_rows:
        logger.info(f"... and {len(models) - max_rows} more models")
    logger.info("")
