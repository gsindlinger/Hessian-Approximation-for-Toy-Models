import torch
from config.config import Config
from data.data import create_dataset
from hessian_approximations.hessian_approximations import (
    create_hessian,
    hessian_approximation,
)
from models.models import create_model, get_loss, predict, train_model
from utils.utils import PlotUtils


def create_dataset_and_model(config: Config):
    dataset = create_dataset(config.dataset)
    model = create_model(
        config, input_dim=dataset.input_dim(), output_dim=dataset.output_dim()
    )
    return dataset, model


def train_and_evaluate(config: Config):
    dataset, model = create_dataset_and_model(config)
    model = train_model(model, dataset.test_train_to_dataloader(), config.training)
    return model, dataset


def main():
    torch.manual_seed(0)

    config = Config.parse_args()
    model, dataset = train_and_evaluate(config)

    hessian_method = create_hessian(config)
    train_data, train_targets = dataset.train_dataset[:]  # type: ignore
    hessian = hessian_approximation(
        hessian_method,
        model,
        train_data,
        train_targets,
        loss=get_loss(config.model.loss),
    )

    print("Making predictions on the test set...")


if __name__ == "__main__":
    main()
