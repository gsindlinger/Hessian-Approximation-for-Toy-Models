from typing import Literal

import optax


def optimizer(
    optimizer_str: Literal["sgd", "adam", "adamw", "sgd_schedule_cosine"],
    lr: float,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Return optimizer with optional L2 weight decay."""
    match optimizer_str:
        case "sgd":
            base = optax.sgd(lr)
        case "adam":
            base = optax.adam(lr)
        case "adamw":
            # AdamW has weight_decay built in
            return optax.adamw(learning_rate=lr, weight_decay=weight_decay)
        case "sgd_schedule_cosine":
            schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=1000)
            base = optax.sgd(schedule)
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer_str}")

    # if weight_decay > 0, apply L2 weight decay before the base optimizer
    if weight_decay > 0:
        return optax.chain(
            optax.add_decayed_weights(weight_decay),
            base,
        )
    else:
        return base
