from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, Dict, Tuple

from flax import linen as nn


class ApproximationModel(nn.Module):
    MODEL_CHECKPOINT_DIR = "data/model_checkpoints"

    input_dim: int
    output_dim: int
    use_bias: bool = False

    @staticmethod
    def get_activation(act_str: str):
        activations = {
            "relu": nn.relu,
            "tanh": nn.tanh,
        }
        if act_str not in activations:
            raise ValueError(f"Unknown activation: {act_str}")
        return activations[act_str]

    def save_model(self, params: Dict):
        """Save model parameters using orbax for better compatibility."""
        try:
            import pickle

            from flax import serialization

            model_name = self.__class__.__name__
            os.makedirs(self.MODEL_CHECKPOINT_DIR, exist_ok=True)

            # Use Flax serialization for better handling of PyTrees
            bytes_output = serialization.to_bytes(params)
            with open(f"{self.MODEL_CHECKPOINT_DIR}/{model_name}.msgpack", "wb") as f:
                f.write(bytes_output)
        except ImportError:
            # Fallback to pickle if serialization not available
            import pickle

            with open(f"{self.MODEL_CHECKPOINT_DIR}/{model_name}.pkl", "wb") as f:
                pickle.dump(params, f)

    def check_saved_model(self) -> bool:
        """Check if saved model parameters exist."""
        model_name = self.__class__.__name__
        checkpoint_path_msgpack = f"{self.MODEL_CHECKPOINT_DIR}/{model_name}.msgpack"
        checkpoint_path_pickle = f"{self.MODEL_CHECKPOINT_DIR}/{model_name}.pkl"
        return os.path.exists(checkpoint_path_msgpack) or os.path.exists(
            checkpoint_path_pickle
        )

    def load_model(self) -> Tuple[ApproximationModel, Dict]:
        """Load model parameters."""
        model_name = self.__class__.__name__

        # Try msgpack first (Flax serialization)
        checkpoint_path = f"{self.MODEL_CHECKPOINT_DIR}/{model_name}.msgpack"
        if os.path.exists(checkpoint_path):
            from flax import serialization

            with open(checkpoint_path, "rb") as f:
                bytes_input = f.read()
            params = serialization.from_bytes(None, bytes_input)
            return self, params

        # Fallback to pickle
        checkpoint_path = f"{self.MODEL_CHECKPOINT_DIR}/{model_name}.pkl"
        if os.path.exists(checkpoint_path):
            import pickle

            with open(checkpoint_path, "rb") as f:
                params = pickle.load(f)
            return self, params

        raise FileNotFoundError("No saved model parameters found.")

    @abstractmethod
    def kfac_apply(self, x, collector) -> Any:
        """Special apply method for K-FAC that wraps layers by custom VJP."""
        pass
