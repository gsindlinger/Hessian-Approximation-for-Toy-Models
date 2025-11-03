import jax.numpy as jnp

from config.config import Config
from data.data import create_dataset
from hessian_approximations.factory import (
    create_hessian_by_name,
    hessian_approximation,
)
from models.loss import get_loss_fn
from models.train import create_model, train_model


def create_dataset_and_model(config: Config):
    dataset = create_dataset(config.dataset)
    model = create_model(
        config, input_dim=dataset.input_dim(), output_dim=dataset.output_dim()
    )
    return dataset, model


def train(config: Config, reload_model: bool = True):
    dataset, model = create_dataset_and_model(config)
    if not reload_model and model.check_saved_model():
        model, params = model.load_model()
    else:
        model, params = train_model(model, dataset.get_dataloaders(), config.training)
    return model, dataset, params


def main():
    config = Config.parse_args()
    model, dataset, params = train(config)

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
