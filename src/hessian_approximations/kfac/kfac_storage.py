from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ...config.config import Config
from ...config.dataset_config import DatasetConfig
from ...config.hessian_approximation_config import KFACBuildConfig
from ...config.model_config import ModelConfig
from ...config.training_config import TrainingConfig
from .data.data import KFACData, MeanEigenvaluesAndCorrections
from .data.layer_components import LayerComponents


class KFACStorage:
    """Handles all disk I/O operations for (E)KFAC components using NumPy's npz format."""

    model_config: ModelConfig
    dataset_config: DatasetConfig
    training_config: TrainingConfig
    kfac_build_config: KFACBuildConfig

    def __init__(
        self,
        configs: Tuple[ModelConfig, DatasetConfig, TrainingConfig, KFACBuildConfig],
    ):
        """Initialize KFAC storage with given configs and set up storage directory."""
        (
            self.model_config,
            self.dataset_config,
            self.training_config,
            self.kfac_build_config,
        ) = configs
        if self.kfac_build_config.storage_dir is not None:
            base_path = self.kfac_build_config.storage_dir
        else:
            base_path = Path("./data/kfac_storage")
        self.base_directory = self.generate_kfac_directory(base_path)

        # When the storage directory is created only to load existing data,
        # we don't need to store the config again. But when the opposite is true,
        # we need to save the config for future reference. The components then will
        # be saved after computation elsewhere.
        if not self.check_component_storage():
            self.save_config()

    def generate_kfac_directory(
        self,
        base_path: str | Path,
    ) -> Path:
        """Generate the KFAC directory path based on the config."""
        dataset_config = self.dataset_config
        model_config = self.model_config
        training_config = self.training_config
        kfac_build_config = self.kfac_build_config

        # Create a unique hash for model, dataset, training, and kfac configs
        dataset_name = dataset_config.name
        model_name = model_config.name
        use_pseudo_targets = kfac_build_config.use_pseudo_targets
        hashed_model_dataset_training = Config.model_training_dataset_hash(
            dataset_config, model_config, training_config, kfac_build_config
        )

        model_checkpoint_dir = Path(
            base_path,
            dataset_name,
            model_name
            + "_"
            + f"psT_{'Y' if use_pseudo_targets else 'N'}_"
            + hashed_model_dataset_training,
        )
        model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return model_checkpoint_dir

    def check_component_storage(self) -> bool:
        """Check if EKFAC components are already stored on disk."""
        for filename in [
            "covariances.npz",
            "eigenvectors.npz",
            "eigenvalues.npz",
            "eigenvalue_corrections.npz",
            "mean_eigenvalues_and_corrections.json",
        ]:
            path = self._get_path(filename)
            if not path.exists():
                return False
        return True

    def check_storage(self) -> bool:
        """Check if the stored config matches the current config."""
        config_path = self._get_path("kfac_storage_config.json")
        if not config_path.exists():
            return False
        with open(config_path, "r") as f:
            saved_config = json.load(f)

        return (
            saved_config["dataset_config"] == asdict(self.dataset_config)
            and saved_config["model_config"] == asdict(self.model_config)
            and saved_config["training_config"] == asdict(self.training_config)
            and saved_config["kfac_config"] == asdict(self.kfac_build_config)
            and self.check_component_storage()
        )

    def save_config(self) -> None:
        """Save the current config to disk."""
        config_path = self._get_path("kfac_storage_config.json")

        with open(config_path, "w") as f:
            json.dump(
                {
                    "dataset_config": asdict(self.dataset_config),
                    "model_config": asdict(self.model_config),
                    "training_config": asdict(self.training_config),
                    "kfac_config": asdict(self.kfac_build_config),
                },
                f,
                indent=4,
            )

    def delete_storage(self) -> None:
        """Delete all stored EKFAC component files from disk including parent folder."""
        for filename in [
            "covariances.npz",
            "eigenvectors.npz",
            "eigenvalues.npz",
            "eigenvalue_corrections.npz",
            "mean_eigenvalues_and_corrections.json",
            "kfac_storage_config.json",
        ]:
            path = self._get_path(filename)
            if path.exists():
                path.unlink()
        try:
            self.base_directory.rmdir()
        except OSError:
            pass  # Directory not empty

    def _get_path(self, filename: str) -> Path:
        return self.base_directory / filename

    def _save_layer_components(
        self, filename: str, components: LayerComponents
    ) -> None:
        """Save layer components to disk."""
        path = self._get_path(filename)
        save_dict = {}

        for prefix, group in (
            ("activations", components.activations),
            ("gradients", components.gradients),
        ):
            for name, arr in group.items():
                save_dict[f"{prefix}_{name}"] = jnp.asarray(arr)

        np.savez_compressed(path, **save_dict)  # type: ignore

    def _load_layer_components(self, filename: str) -> LayerComponents:
        """Load layer components from disk."""
        path = self._get_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"No file found at {path}")

        data = np.load(path, allow_pickle=False)
        activations, gradients = {}, {}

        for key in data.files:
            prefix, name = key.split("_", 1)
            arr = jnp.asarray(data[key])

            if prefix == "activations":
                activations[name] = arr
            else:
                gradients[name] = arr

        return LayerComponents(activations, gradients)

    def save_covariances(self, covariances: LayerComponents) -> None:
        self._save_layer_components("covariances.npz", covariances)

    def load_covariances(self) -> LayerComponents:
        return self._load_layer_components("covariances.npz")

    def save_eigenvectors(self, eigenvectors: LayerComponents) -> None:
        self._save_layer_components("eigenvectors.npz", eigenvectors)

    def load_eigenvectors(self) -> LayerComponents:
        return self._load_layer_components("eigenvectors.npz")

    def save_eigenvalues(self, eigenvalues: LayerComponents) -> None:
        self._save_layer_components("eigenvalues.npz", eigenvalues)

    def load_eigenvalues(self) -> LayerComponents:
        return self._load_layer_components("eigenvalues.npz")

    def save_eigenvalue_corrections(
        self, corrections: Dict[str, Float[Array, "I O"]]
    ) -> None:
        path = self._get_path("eigenvalue_corrections.npz")
        save_dict = {name: jnp.asarray(arr) for name, arr in corrections.items()}
        np.savez_compressed(path, **save_dict)  # type: ignore

    def load_eigenvalue_corrections(self) -> Dict[str, Float[Array, "I O"]]:
        path = self._get_path("eigenvalue_corrections.npz")
        if not path.exists():
            raise FileNotFoundError(f"No eigenvalue correction file found at {path}")

        data = np.load(path, allow_pickle=False)
        return {name: jnp.asarray(data[name]) for name in data.files}

    def save_mean_eigenvalues_and_corrections(
        self,
        kfac_means: MeanEigenvaluesAndCorrections,
    ) -> None:
        """Save mean eigenvalues and corrections as JSON."""
        path = self._get_path("mean_eigenvalues_and_corrections.json")

        def to_float(val):
            if isinstance(val, (jnp.ndarray, np.ndarray)):
                return float(val)
            return float(val)

        with open(path, "w") as f:
            json.dump(
                {
                    "mean_eigenvalues": {
                        k: to_float(v) for k, v in kfac_means.eigenvalues.items()
                    },
                    "mean_corrections": {
                        k: to_float(v) for k, v in kfac_means.corrections.items()
                    },
                    "overall_mean_eigenvalue": to_float(
                        kfac_means.overall_mean_eigenvalues
                    ),
                    "overall_mean_correction": to_float(
                        kfac_means.overall_mean_corrections
                    ),
                },
                f,
                indent=4,
            )

    def load_mean_eigenvalues_and_corrections(
        self,
    ) -> Tuple[
        Dict[str, Float[Array, ""]],
        Dict[str, Float[Array, ""]],
        Float[Array, ""],
        Float[Array, ""],
    ]:
        """Load mean eigenvalues and corrections from disk.

        Returns:
            Tuple containing mean_eigenvalues, mean_corrections,
            overall_mean_eigenvalue, overall_mean_correction

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = self._get_path("mean_eigenvalues_and_corrections.json")
        if not path.exists():
            raise FileNotFoundError(
                f"No mean eigenvalues and corrections file found at {path}"
            )

        with open(path, "r") as f:
            data = json.load(f)

        return (
            {k: jnp.asarray(v) for k, v in data["mean_eigenvalues"].items()},
            {k: jnp.asarray(v) for k, v in data["mean_corrections"].items()},
            jnp.asarray(data["overall_mean_eigenvalue"]),
            jnp.asarray(data["overall_mean_correction"]),
        )

    def load_kfac_data(self) -> Tuple[KFACData, MeanEigenvaluesAndCorrections]:
        """Load all KFAC data from storage."""
        kfac_data = KFACData(
            covariances=self.load_covariances(),
            eigenvectors=self.load_eigenvectors(),
            eigenvalues=self.load_eigenvalues(),
            eigenvalue_corrections=self.load_eigenvalue_corrections(),
        )

        kfac_means = MeanEigenvaluesAndCorrections()
        (
            kfac_means.eigenvalues,
            kfac_means.corrections,
            kfac_means.overall_mean_eigenvalues,
            kfac_means.overall_mean_corrections,
        ) = self.load_mean_eigenvalues_and_corrections()

        return kfac_data, kfac_means
