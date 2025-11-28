from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainingConfig:
    """Configuration for training."""

    epochs: int = 20  # Number of training epochs
    lr: float = 0.01  # Learning rate
    save_checkpoint: bool = True  # Whether to save model checkpoints
    batch_size: int = 32  # Batch size for training
    optimizer: Literal["sgd", "adam"] = "sgd"  # Optimizer type
    loss: Literal["mse", "cross_entropy"] = "mse"  # Loss function
