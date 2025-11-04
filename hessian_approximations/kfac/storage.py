from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from hessian_approximations.kfac.layer_components import LayerComponents


class KFACStorage:
    """Handles all disk I/O operations for (E)KFAC components using NumPy's npz format."""

    def __init__(
        self,
        model_name: str = "model",
        dataset_name: str = "dataset",
        base_path: str | Path = "data",
    ):
        self.base_path = Path(base_path) / model_name / dataset_name
        self.base_path.mkdir(parents=True, exist_ok=True)

    def check_storage(self) -> bool:
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

    def delete_storage(self) -> None:
        """Delete all stored EKFAC component files from disk including parent folder."""
        for filename in [
            "covariances.npz",
            "eigenvectors.npz",
            "eigenvalues.npz",
            "eigenvalue_corrections.npz",
            "mean_eigenvalues_and_corrections.json",
        ]:
            path = self._get_path(filename)
            if path.exists():
                path.unlink()
        try:
            self.base_path.rmdir()
        except OSError:
            pass  # Directory not empty

    def _get_path(self, filename: str) -> Path:
        return self.base_path / filename

    def _save_layer_components(
        self, filename: str, components: LayerComponents
    ) -> None:
        path = self._get_path(filename)
        save_dict = {
            f"{prefix}_{name}": np.asarray(arr)
            for prefix, group in (
                ("activations", components.activations),
                ("gradients", components.gradients),
            )
            for name, arr in group.items()
        }
        np.savez_compressed(path, **save_dict)  # type: ignore

    def _load_layer_components(self, filename: str) -> LayerComponents:
        path = self._get_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"No file found at {path}")
        data = np.load(path, allow_pickle=False)
        activations, gradients = {}, {}
        for key in data.files:
            prefix, name = key.split("_", 1)
            (activations if prefix == "activations" else gradients)[name] = jnp.array(
                data[key]
            )
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
        np.savez_compressed(
            path,
            **{name: np.asarray(arr) for name, arr in corrections.items()},  # type: ignore
        )

    def load_eigenvalue_corrections(self) -> Dict[str, Float[Array, "I O"]]:
        path = self._get_path("eigenvalue_corrections.npz")
        if not path.exists():
            raise FileNotFoundError(f"No eigenvalue correction file found at {path}")
        data = np.load(path, allow_pickle=False)
        return {name: jnp.array(data[name]) for name in data.files}

    def save_mean_eigenvalues_and_corrections(
        self,
        mean_eigenvalues: Dict[str, Float[Array, ""]],
        mean_corrections: Dict[str, Float[Array, ""]],
        overall_mean_eigenvalue: Float[Array, ""],
        overall_mean_correction: Float[Array, ""],
    ) -> None:
        # store as a json file
        path = self._get_path("mean_eigenvalues_and_corrections.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "mean_eigenvalues": {
                        k: float(v) for k, v in mean_eigenvalues.items()
                    },
                    "mean_corrections": {
                        k: float(v) for k, v in mean_corrections.items()
                    },
                    "overall_mean_eigenvalue": float(overall_mean_eigenvalue),
                    "overall_mean_correction": float(overall_mean_correction),
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

        Raises:
            FileNotFoundError: If the file does not exist.

        Returns:
            Tuple[Dict[str, float], Dict[str, float], float, float]: mean_eigenvalues,
            mean_corrections, overall_mean_eigenvalue, overall_mean_correction
        """
        path = self._get_path("mean_eigenvalues_and_corrections.json")
        if not path.exists():
            raise FileNotFoundError(
                f"No mean eigenvalues and corrections file found at {path}"
            )
        with open(path, "r") as f:
            data = json.load(f)
        return (
            {k: jnp.array(v) for k, v in data["mean_eigenvalues"].items()},
            {k: jnp.array(v) for k, v in data["mean_corrections"].items()},
            jnp.array(data["overall_mean_eigenvalue"]),
            jnp.array(data["overall_mean_correction"]),
        )
