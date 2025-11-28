from __future__ import annotations

from typing import Literal

import optax


def get_optimizer(
    optimizer_str: Literal["sgd", "adam"], lr: float
) -> optax.GradientTransformation:
    """Return optimizer."""
    match optimizer_str:
        case "sgd":
            return optax.sgd(lr)
        case "adam":
            return optax.adam(lr)
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer_str}")
