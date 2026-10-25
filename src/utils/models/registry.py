from dataclasses import asdict

from src.config import ActivationFunction, ModelArchitecture, ModelConfig
from src.utils.models.approximation_model import ApproximationModel
from src.utils.models.linear_model import LinearModel
from src.utils.models.mlp import MLP
from src.utils.models.mlp_swiglu import MLPSwiGLU
from src.utils.models.resnet import ResNetMLP
from src.utils.models.resnet_swiglu import ResNetMLPSwiGLU
from src.utils.models.transformer import TinyTransformer


class ModelRegistry:
    """Registry for model architectures."""

    REGISTRY: dict[ModelArchitecture, type[ApproximationModel]] = {
        ModelArchitecture.MLP: MLP,
        ModelArchitecture.MLPSWIGLU: MLPSwiGLU,
        ModelArchitecture.LINEAR: LinearModel,
        ModelArchitecture.RESNETMLP: ResNetMLP,
        ModelArchitecture.RESNETMLPSWIGLU: ResNetMLPSwiGLU,
        ModelArchitecture.TRANSFORMER: TinyTransformer,
    }

    @staticmethod
    def get_model(model_config: ModelConfig, seed: int = 42) -> ApproximationModel:
        model_cls = ModelRegistry.REGISTRY[model_config.architecture]
        hidden_dim = asdict(model_config).get("hidden_dim", [])
        activation = asdict(model_config).get("activation", None)
        if model_cls is TinyTransformer:
            transformer_fields = {
                key: value for key, value in asdict(model_config).items()
                if key.startswith("transformer_")
            }
            return model_cls(
                input_dim=model_config.input_dim,
                output_dim=model_config.output_dim,
                hidden_dim=hidden_dim,
                activation=ActivationFunction(activation) if activation is not None else ActivationFunction.RELU,
                seed=seed,
                **transformer_fields,
            )
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
