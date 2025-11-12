from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict
from typing import Dict

from flax import serialization

from config.config import Config, TrainingConfig
from config.dataset_config import DatasetConfig
from config.model_config import ModelConfig

CHECKPOINT_STR = "checkpoint"
CONFIGS_STR = "configs"


def save_model_checkpoint(
    params: Dict,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    base_path: str = "data/artifacts/model_checkpoints",
) -> None:
    """Save model parameters using orbax for better compatibility."""

    model_checkpoint_dir = generate_model_checkpoint_path(
        dataset_config, model_config, training_config, base_path
    )

    # write model parameters and joined configs (one single json)
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    with open(f"{model_checkpoint_dir}/{CONFIGS_STR}.json", "w") as f:
        json.dump(
            {
                "dataset_config": asdict(dataset_config),
                "model_config": asdict(model_config),
                "training_config": asdict(training_config),
            },
            f,
            indent=4,
        )

    try:
        # Use Flax serialization for better handling of PyTrees
        bytes_output = serialization.to_bytes(params)
        with open(f"{model_checkpoint_dir}/{CHECKPOINT_STR}.msgpack", "wb") as f:
            f.write(bytes_output)
    except ImportError:
        # Fallback to pickle if serialization not available
        try:
            with open(f"{model_checkpoint_dir}/{CHECKPOINT_STR}.pkl", "wb") as f:
                pickle.dump(params, f)
        except Exception as e:
            print(f"Failed to save model parameters: {e}")


def generate_model_checkpoint_path(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    base_path: str = "data/artifacts/model_checkpoints",
) -> str:
    dataset_name = dataset_config.name
    model_name = model_config.name
    hashed_model_dataset_training = Config.model_training_dataset_hash(
        dataset_config, model_config, training_config
    )

    model_checkpoint_dir = os.path.join(
        base_path, dataset_name, model_name + "_" + hashed_model_dataset_training
    )

    return model_checkpoint_dir


def check_saved_model(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> bool:
    """Check if a saved model exists and matches the given configs."""
    model_checkpoint_dir = generate_model_checkpoint_path(
        dataset_config, model_config, training_config
    )

    # Check for both msgpack and pickle files
    msgpack_path = f"{model_checkpoint_dir}/{CHECKPOINT_STR}.msgpack"
    pickle_path = f"{model_checkpoint_dir}/{CHECKPOINT_STR}.pkl"

    if os.path.exists(msgpack_path) or os.path.exists(pickle_path):
        # Check whether configs match; if so return True
        with open(f"{model_checkpoint_dir}/{CONFIGS_STR}.json", "r") as f:
            saved_config = json.load(f)
        if (
            saved_config["dataset_config"] == asdict(dataset_config)
            and saved_config["model_config"] == asdict(model_config)
            and saved_config["training_config"] == asdict(training_config)
        ):
            return True
    return False


def load_model_checkpoint(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> Dict:
    """Load model parameters from checkpoint. Assumes the checkpoint exists."""
    model_checkpoint_dir = generate_model_checkpoint_path(
        dataset_config, model_config, training_config
    )

    msgpack_path = f"{model_checkpoint_dir}/{CHECKPOINT_STR}.msgpack"
    pickle_path = f"{model_checkpoint_dir}/{CHECKPOINT_STR}.pkl"

    if os.path.exists(msgpack_path):
        try:
            with open(msgpack_path, "rb") as f:
                bytes_input = f.read()
            params = serialization.from_bytes(None, bytes_input)
            return params
        except ImportError:
            print("Flax serialization not available. Falling back to pickle.")

    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, "rb") as f:
                params = pickle.load(f)
            return params
        except Exception as e:
            print(f"Failed to load model parameters: {e}")

    raise FileNotFoundError("No valid checkpoint file found.")
