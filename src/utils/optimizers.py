from typing import Optional

import optax

from src.config import LRSchedule, OptimizerType


def optimizer(
    optimizer_enum: OptimizerType,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    lr_schedule: LRSchedule = LRSchedule.NONE,
    total_steps: Optional[int] = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> optax.GradientTransformation:
    """Return optimizer with optional L2 weight decay and LR schedule.

    Args:
        lr_schedule: When ``LRSchedule.COSINE``, wraps the base learning rate
            with a cosine decay schedule that anneals from ``lr`` to 0 over
            ``total_steps`` gradient steps.
        total_steps: Total number of gradient updates across all training.
            Required when ``lr_schedule=LRSchedule.COSINE``; ignored otherwise.
    """
    if not 0 <= momentum < 1:
        raise ValueError("momentum must be in [0, 1)")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("Adam beta1 and beta2 must be in [0, 1)")
    if lr <= 0 or weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative")
    if lr_schedule == LRSchedule.COSINE:
        if total_steps is None or total_steps <= 0:
            raise ValueError(
                "total_steps must be provided when lr_schedule=LRSchedule.COSINE"
            )
        effective_lr: optax.Schedule | float = optax.cosine_decay_schedule(
            init_value=lr, decay_steps=total_steps
        )
    else:
        effective_lr = lr

    match optimizer_enum:
        case OptimizerType.SGD:
            base = optax.sgd(learning_rate=effective_lr, momentum=momentum)
        case OptimizerType.ADAM:
            base = optax.adam(learning_rate=effective_lr, b1=beta1, b2=beta2)
        case OptimizerType.ADAMW:
            # AdamW has weight_decay built in; cosine schedule is applied via effective_lr
            return optax.adamw(learning_rate=effective_lr, weight_decay=weight_decay,
                               b1=beta1, b2=beta2)
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer_enum}")

    if weight_decay > 0:
        return optax.chain(
            optax.add_decayed_weights(weight_decay),
            base,
        )
    else:
        return base
