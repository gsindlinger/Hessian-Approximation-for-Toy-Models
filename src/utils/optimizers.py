from __future__ import annotations

from typing import Literal

import optax


def optimizer(
    optimizer_str: Literal["sgd", "adam", "adamw", "sgd_schedule_cosine"], lr: float
) -> optax.GradientTransformation:
    """Return optimizer."""
    match optimizer_str:
        case "sgd":
            return optax.sgd(lr)
        case "adam":
            return optax.adam(lr)
        case "adamw":
            return optax.adamw(lr)
        case "sgd_schedule_cosine":
            schedule = optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=1000,
            )
            return optax.sgd(schedule)
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer_str}")
