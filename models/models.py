from __future__ import annotations
from abc import ABC, abstractmethod
import os
from typing import Literal, Tuple
from typing_extensions import override
import torch
from tqdm import tqdm
from config.config import Config, TrainingConfig
from torch.utils.data import DataLoader


def create_model(config: Config, input_dim: int, output_dim: int) -> ApproximationModel:
    """Create model from config."""
    model_map = {
        "linear": LinearModel,
        # Add more models here as you implement them
    }

    model_cls = model_map.get(config.model.name)
    if model_cls is None:
        raise ValueError(f"Unknown model: {config.model.name}")

    model_kwargs = vars(config.model).copy()
    model_kwargs.pop("name", None)
    model_kwargs.pop("loss", None)
    model_kwargs.update({"input_dim": input_dim, "output_dim": output_dim})
    return model_cls(**model_kwargs)


def get_loss(loss_str: Literal["mse", "cross_entropy"] = "mse"):
    match loss_str:
        case "mse":
            return torch.nn.MSELoss()
        case "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        case _:
            raise ValueError(f"Unknown loss function: {loss_str}")


def get_optimizer(
    optimizer_str: Literal["sgd", "adam"], model: ApproximationModel, lr: float
):
    match optimizer_str:
        case "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr)
        case "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer_str}")


def validate_model(model, dataloader: DataLoader, training_config: TrainingConfig):
    loss_fn = get_loss(training_config.loss)
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_data, batch_targets in dataloader:
            outputs = model(batch_data)
            loss_value: torch.Tensor = loss_fn(outputs, batch_targets)
            running_loss += loss_value.item() * batch_data.size(0)
    val_loss = running_loss / len(dataloader.dataset)  # type: ignore
    return val_loss


def train_model(
    model: ApproximationModel,
    dataloader: Tuple[DataLoader, DataLoader | None],
    training_config: TrainingConfig,
):
    train_loader, val_loader = dataloader

    loss_fn = get_loss(training_config.loss)
    optimizer = get_optimizer(training_config.optimizer, model, training_config.lr)

    for epoch in tqdm(range(training_config.epochs)):
        model.train()
        running_loss = 0.0

        for batch_data, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss_value: torch.Tensor = loss_fn(outputs, batch_targets)
            loss_value.backward()
            optimizer.step()
            running_loss += loss_value.item() * batch_data.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)  # type: ignore

        if val_loader is not None:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_loss = validate_model(model, val_loader, training_config)
                tqdm.write(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f}")
                tqdm.write(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

    # save checkpoint of the model
    if training_config.save_checkpoint:
        save_model(model)

    return model


def save_model(model):
    model_name = model.__class__.__name__
    os.makedirs("data/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"data/checkpoints/{model_name}.pth")


def load_model(model):
    model_name = model.__class__.__name__
    checkpoint_path = f"data/checkpoints/{model_name}.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        return True
    return False


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)


def evaluate(model, data, targets, loss_str: Literal["mse", "cross_entropy"] = "mse"):
    loss = get_loss(loss_str)
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        loss_temp = loss(outputs, targets)
    return loss_temp.item()


class ApproximationModel(torch.nn.Module, ABC):
    def get_activation(self, act_str: str):
        match act_str:
            case "relu":
                return torch.nn.ReLU()
            case "tanh":
                return torch.nn.Tanh()
            case _:
                raise ValueError(f"Unknown activation function: {act_str}")

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")


class LinearModel(ApproximationModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: list[int] = [],
    ):
        super().__init__()
        self.network = torch.nn.Sequential()
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dim):
            self.network.add_module(
                f"linear_{i}", torch.nn.Linear(prev_dim, h_dim, bias=True)
            )
            prev_dim = h_dim
        self.network.add_module(
            "output", torch.nn.Linear(prev_dim, output_dim, bias=True)
        )

    @override
    def forward(self, x):
        return self.network(x)
