from dataclasses import asdict

from src.config import ModelArchitecture, ModelConfig
from src.utils.models.approximation_model import ApproximationModel
from src.utils.models.linear_model import LinearModel
from src.utils.models.mlp import MLP
from src.utils.models.mlp_swiglu import MLPSwiGLU


class ModelRegistry:
    """Registry for model architectures."""

    REGISTRY: dict[ModelArchitecture, type[ApproximationModel]] = {
        ModelArchitecture.MLP: MLP,
        ModelArchitecture.MLPSWIGLU: MLPSwiGLU,
        ModelArchitecture.LINEAR: LinearModel,
    }

    @staticmethod
    def get_model(model_config: ModelConfig) -> ApproximationModel:
        model_cls = ModelRegistry.REGISTRY[model_config.architecture]
        hidden_dim = asdict(model_config).get("hidden_dim", [])
        return model_cls(
            input_dim=model_config.input_dim,
            output_dim=model_config.output_dim,
            hidden_dim=hidden_dim,
        )
