from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Main configuration for the project."""

    dataset_path: str
    """ File path of dataset. """

    model: ModelConfig
    """ Model configuration. """

    hessian_approximation: HessianApproximationConfig
    """ Hessian approximation configuration. """

    seed: int = 42
    """ Random seed for reproducibility. """


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str
    """ Name or type of the model. """

    directory: str | None = None
    """ Path to model parameters, if applicable. """

    metadata: dict | None = None
    """ Additional metadata for model configuration. """


@dataclass
class HessianApproximationConfig:
    """Configuration for Hessian approximation methods."""

    method: str
    """Hessian approximation method (e.g., 'KFAC', 'EKFAC', 'FIM')."""

    directory: str | None = None
    """Path to save the Hessian approximation results."""
