from dataclasses import asdict

from src.config import ActivationFunction, ModelArchitecture, ModelConfig
from src.utils.models.approximation_model import ApproximationModel
from src.utils.models.linear_model import LinearModel
from src.utils.models.mlp import MLP
from src.utils.models.mlp_swiglu import MLPSwiGLU
from src.utils.models.resnet import ResNetMLP
from src.utils.models.resnet_swiglu import ResNetMLPSwiGLU


class ModelRegistry:
    """Registry for model architectures."""

    REGISTRY: dict[ModelArchitecture, type[ApproximationModel]] = {
        ModelArchitecture.MLP: MLP,
        ModelArchitecture.MLPSWIGLU: MLPSwiGLU,
        ModelArchitecture.LINEAR: LinearModel,
        ModelArchitecture.RESNETMLP: ResNetMLP,
        ModelArchitecture.RESNETMLPSWIGLU: ResNetMLPSwiGLU,
    }

    @staticmethod
    def get_model(model_config: ModelConfig, seed: int = 42) -> ApproximationModel:
        model_cls = ModelRegistry.REGISTRY[model_config.architecture]
        hidden_dim = asdict(model_config).get("hidden_dim", [])
        activation = asdict(model_config).get("activation", None)
        if activation is not None:
            activation = ActivationFunction(activation)
            return model_cls(
                input_dim=model_config.input_dim,
                output_dim=model_config.output_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                seed=seed,
            )
        else:
            return model_cls(
                input_dim=model_config.input_dim,
                output_dim=model_config.output_dim,
                hidden_dim=hidden_dim,
                seed=seed,
            )
