import os
from typing import Any

from flax import linen as nn


class ApproximationModel(nn.Module):
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

    def save_model(self, params: Any):
        """Save model parameters using orbax for better compatibility."""
        try:
            import pickle

            from flax import serialization

            model_name = self.__class__.__name__
            os.makedirs(f"data/{model_name}/checkpoints", exist_ok=True)

            # Use Flax serialization for better handling of PyTrees
            bytes_output = serialization.to_bytes(params)
            with open(f"data/{model_name}/checkpoints/{model_name}.msgpack", "wb") as f:
                f.write(bytes_output)
        except ImportError:
            # Fallback to pickle if serialization not available
            import pickle

            model_name = self.__class__.__name__
            os.makedirs(f"data/{model_name}/checkpoints", exist_ok=True)
            with open(f"data/{model_name}/checkpoints/{model_name}.pkl", "wb") as f:
                pickle.dump(params, f)

    def load_model(self):
        """Load model parameters."""
        model_name = self.__class__.__name__

        # Try msgpack first (Flax serialization)
        checkpoint_path = f"data/{model_name}/checkpoints/{model_name}.msgpack"
        if os.path.exists(checkpoint_path):
            from flax import serialization

            with open(checkpoint_path, "rb") as f:
                bytes_input = f.read()
            params = serialization.from_bytes(None, bytes_input)
            return params

        # Fallback to pickle
        checkpoint_path = f"data/{model_name}/checkpoints/{model_name}.pkl"
        if os.path.exists(checkpoint_path):
            import pickle

            with open(checkpoint_path, "rb") as f:
                params = pickle.load(f)
            return params

        return None
