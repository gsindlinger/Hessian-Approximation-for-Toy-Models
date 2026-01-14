import gc
import logging
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import List, Tuple, get_args, get_type_hints

import jax
import yaml
from jax.tree_util import tree_map
from omegaconf import DictConfig, OmegaConf
from typing_extensions import get_origin

from src.config import DatasetConfig
from src.utils.utils import get_peak_bytes_in_use

logger = logging.getLogger(__name__)


def to_dataclass(cls, config):
    """Recursively convert OmegaConf DictConfig to dataclass instances."""
    if isinstance(config, cls):
        return config

    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    if config is None:
        return None

    if not is_dataclass(cls):
        return config

    if not isinstance(config, dict):
        return config

    kwargs = {}

    try:
        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {f.name: f.type for f in fields(cls)}

    for field_info in fields(cls):
        field_name = field_info.name

        if field_name not in config:
            continue

        value = config[field_name]

        if value is None:
            kwargs[field_name] = None
            continue

        field_type = type_hints.get(field_name, field_info.type)

        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Union types
        if origin is type(None) or (origin and origin.__name__ == "Union"):
            non_none_types = [arg for arg in args if arg is not type(None)]
            if non_none_types:
                field_type = non_none_types[0]
                origin = get_origin(field_type)
                args = get_args(field_type)

        if origin is list:
            item_type = args[0] if args else None
            new_list = []

            for item in value:
                if item_type is None:
                    new_list.append(item)
                elif is_dataclass(item_type):
                    new_list.append(to_dataclass(item_type, item))
                else:
                    new_list.append(_convert_value(item_type, item))

            kwargs[field_name] = new_list

        elif is_dataclass(field_type):
            kwargs[field_name] = to_dataclass(field_type, value)

        else:
            kwargs[field_name] = _convert_value(field_type, value)

    return cls(**kwargs)  # type: ignore


def _convert_value(target_type, value):
    if isinstance(value, target_type):
        return value

    try:
        if isinstance(target_type, type) and issubclass(target_type, Enum):
            return target_type(value)
    except TypeError:
        pass

    origin = get_origin(target_type)
    args = get_args(target_type)

    if origin is tuple and isinstance(value, list):
        if args:
            return tuple(_convert_value(t, v) for t, v in zip(args, value))
        return tuple(value)

    return value


def load_experiment_override_from_yaml(
    path: str,
) -> Tuple[List[str], DatasetConfig, int | None]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if "models" in data:
        models = data.get("models", [])

    if "dataset" in data:
        dataset = to_dataclass(DatasetConfig, data["dataset"])

    seed = data.get("seed", None)

    return models, dataset, seed  # type: ignore


def cleanup_memory(stage: str | None = None):
    """Force garbage collection and clear JAX caches."""
    gc.collect()
    jax.clear_caches()
    msg = f"[MEMORY] peak_bytes={get_peak_bytes_in_use():.2f} GB"
    if stage:
        msg = f"[MEMORY] after {stage}: {msg}"
    logger.info(msg)


def block_tree(x, name: str):
    """Block until all arrays in a tree are ready."""
    logger.info(f"[SYNC] Blocking on {name}")
    try:
        x = tree_map(lambda y: y.block_until_ready(), x)
    except Exception:
        logger.exception(f"[SYNC] Failure while blocking on {name}")
        raise
    logger.info(f"[SYNC] Completed {name}")
    return x
