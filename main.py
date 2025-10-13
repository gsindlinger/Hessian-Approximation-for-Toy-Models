import jax
import torch
from config.config import Config
from data.data import create_dataset
from hessian_approximations.hessian_approximations import (
    create_hessian,
    hessian_approximation,
    hessian_vector_product,
)
from models.models import create_model, get_loss_fn, mse_loss, train_model
from utils.utils import PlotUtils
import jax.numpy as jnp
from jax import flatten_util


def create_dataset_and_model(config: Config):
    dataset = create_dataset(config.dataset)
    model = create_model(
        config, input_dim=dataset.input_dim(), output_dim=dataset.output_dim()
    )
    return dataset, model


def train_and_evaluate(config: Config):
    dataset, model = create_dataset_and_model(config)
    model, params = train_model(model, dataset.get_dataloaders(), config.training)

    return model, dataset, params


def main():
    torch.manual_seed(0)

    config = Config.parse_args()
    model, dataset, params = train_and_evaluate(config)

    hessian_method = create_hessian(config)
    train_data, train_targets = dataset.get_train_data()

    # Example of computing full Hessian
    hessian = hessian_approximation(
        hessian_method,
        model,
        params,
        jnp.asarray(train_data),
        jnp.asarray(train_targets),
        loss=get_loss_fn(config.model.loss),
    )

    # Example of computing Hessian-vector product
    params_flat, unravel_fn = flatten_util.ravel_pytree(params)
    hessian_vp = hessian_vector_product(
        hessian_method,
        model,
        params,
        jnp.asarray(train_data),
        jnp.asarray(train_targets),
        loss=get_loss_fn(config.model.loss),
        vector=jnp.ones_like(params_flat),
    )

    # convert ndarray to format which is visualizable by data wrangler
    print(hessian)

    print("Making predictions on the test set...")


if __name__ == "__main__":
    main()
