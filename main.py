import jax.numpy as jnp

from config.config import Config
from hessian_approximations.factory import (
    create_hessian_by_name,
    hessian_approximation,
)
from models.train import train_or_load
from models.utils.loss import get_loss_fn


def main():
    config = Config.parse_args()
    model, dataset, params = train_or_load(config)

    train_data, train_targets = dataset.get_train_data()
    hessian_method = create_hessian_by_name("kfac")
    loss_fn = get_loss_fn(config.model.loss)

    hessian_matrix = hessian_approximation(
        method=hessian_method,
        model=model,
        parameters=params,
        test_data=jnp.asarray(train_data),
        test_targets=jnp.asarray(train_targets),
        loss=loss_fn,
    )

    print("Hessian matrix shape:", hessian_matrix.shape)


if __name__ == "__main__":
    main()
