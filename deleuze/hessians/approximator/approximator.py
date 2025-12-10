from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Type, get_args, get_origin, get_type_hints

import jax.numpy as jnp
import numpy as np

from deleuze.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ApproximatorBase(ABC):
    config: Config
    data: Optional[Any] = field(
        default=None
    )  # subclasses define concrete dataclass here

    def build(self, model, params, dataset, loss_fn):
        logger.info(
            f"Start collecting & computing components for: {self.__class__.__name__}"
        )
        self._build(model, params, dataset, loss_fn)

        logger.info(
            f"Finished collecting & computing components for: {self.__class__.__name__}"
        )

    @abstractmethod
    def _build(self, model, params, data, loss_fn):
        """This is where the magic happens.
        Each subclass must implement its own logic for collecting information about the Hessian approximation."""
        pass

    def save(self, directory: str) -> None:
        if self.data is None:
            raise ValueError(
                "self.data is None — subclasses must populate it in _build()"
            )

        if not is_dataclass(self.data):
            raise TypeError("self.data must be a dataclass instance")

        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(save_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=4)

        arrays = {}

        for field_t in fields(self.data):
            value = getattr(self.data, field_t.name)
            field_type = field_t.type

            # Dict[str, array]
            if get_origin(field_type) is dict:
                key_type, _ = get_args(field_type)
                if key_type is str:
                    for k, arr in value.items():
                        arrays[f"{field_t.name}/{k}"] = np.asarray(arr)
                    continue

            arrays[field_t.name] = np.asarray(value)

        np.savez_compressed(save_dir / "data.npz", **arrays)

    @classmethod
    def load(cls: Type[ApproximatorBase], directory: str) -> ApproximatorBase:
        load_dir = Path(directory)

        # Load config
        with open(load_dir / "config.json") as f:
            config_dict = json.load(f)
        config = Config(**config_dict)

        # Load arrays
        loaded = np.load(load_dir / "data.npz")

        # Determine the data class from the subclass annotation
        hints = get_type_hints(cls)
        data_type = hints["data"]
        if get_origin(data_type) is Optional:
            data_type = get_args(data_type)[0]

        if not is_dataclass(data_type):
            raise TypeError("The subclass must annotate `data` with a dataclass type")

        data_kwargs = {}

        for field_t in fields(data_type):
            field_type = field_t.type

            # Dict[str, array]
            if get_origin(field_type) is dict:
                key_type, _ = get_args(field_type)
                if key_type is str:
                    prefix = f"{field_t.name}/"
                    d = {}
                    for k in loaded.files:
                        if k.startswith(prefix):
                            d[k[len(prefix) :]] = jnp.array(loaded[k])
                    data_kwargs[field_t.name] = d
                    continue

            # Scalar / array
            if field_t.name in loaded:
                arr = loaded[field_t.name]
                if arr.shape == ():
                    data_kwargs[field_t.name] = arr.item()
                else:
                    data_kwargs[field_t.name] = jnp.array(arr)

        data_obj = data_type(**data_kwargs)  # type: ignore

        return cls(config=config, data=data_obj)
