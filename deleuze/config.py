from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Main configuration for the project."""

    data: DatasetConfig
    """ Dataset configuration. """

    model: ModelConfig
    """ Model configuration. """

    seed: int = 42
    """ Random seed for reproducibility. """


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    dataset: str
    """ Dataset identifer or name. """

    file_path: str | None = None
    """ Path to dataset file, if applicable. """

    metadata: dict | None = None
    """ Addtional metadata for dataset configuration. """


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str
    """ Name or type of the model. """

    params_path: str | None = None
    """ Path to model parameters, if applicable. """

    metadata: dict | None = None
    """ Additional metadata for model configuration. """
