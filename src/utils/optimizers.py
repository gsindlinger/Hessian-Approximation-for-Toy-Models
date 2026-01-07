import optax

from src.config import OptimizerType


def optimizer(
    optimizer_enum: OptimizerType,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> optax.GradientTransformation:
    """Return optimizer with optional L2 weight decay."""
    match optimizer_enum:
        case OptimizerType.SGD:
            base = optax.sgd(learning_rate=lr, momentum=momentum)
        case OptimizerType.ADAM:
            base = optax.adam(learning_rate=lr)
        case OptimizerType.ADAMW:
            # AdamW has weight_decay built in
            return optax.adamw(learning_rate=lr, weight_decay=weight_decay)
        case OptimizerType.SGD_SCHEDULE_COSINE:
            schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=1000)
            base = optax.sgd(learning_rate=schedule, momentum=momentum)
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer_enum}")

    # if weight_decay > 0, apply L2 weight decay before the base optimizer
    if weight_decay > 0:
        return optax.chain(
            optax.add_decayed_weights(weight_decay),
            base,
        )
    else:
        return base
