from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Optional,
    Tuple,
)

from jaxtyping import Array, Float

from src.hessians.utils.data import (
    ApproximationData,
    DataActivationsGradients,
    ModelContext,
)
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric

logger = logging.getLogger(__name__)


@dataclass
class HessianEstimator(ABC):
    compute_context: (
        ModelContext | Tuple[DataActivationsGradients, DataActivationsGradients]
    )

    is_built: bool = False
    precomputed_data: Optional[ApproximationData] = None
    precomputed_data_directory: Optional[str] = None

    def build(self, base_directory: Optional[str] = None) -> HessianEstimator:
        """Build the Hessian approximation by computing the relevant components. Optionally saves the components and config to the specified directory."""

        # If already built, return self
        if self.is_built:
            return self

        # Use base directory plus the class name to create path for precomputed data
        directory_path = None
        if base_directory is not None:
            # create directory path based on self type and replace computer with data
            if self.precomputed_data_directory is None:
                directory_name = (
                    type(self).__name__.lower().replace("computer", "_data")
                )
            else:
                directory_name = self.precomputed_data_directory
            directory_path = f"{base_directory}/{directory_name}"

        # Check if data exists on disk, if so load it
        if directory_path is not None and ApproximationData.exists(directory_path):
            assert self.precomputed_data is not None, (
                "precomputed_data must be set to load from disk."
            )
            self.precomputed_data = self.precomputed_data.load(directory_path)
            self.is_built = True

            logger.info(f"Loaded {directory_name} from directory: {directory_path}")
        # Otherwise, build the data and save it if a base directory is provided
        else:
            self.precomputed_data = self._build(compute_context=self.compute_context)
            if directory_path is not None and self.precomputed_data is not None:
                self.precomputed_data.save(directory=directory_path)
                output_type = "precomputed data"
                logger.info(f"Saved {output_type} to directory: {directory_path}")
            self.is_built = True
        return self

    @staticmethod
    def _build(
        compute_context: ModelContext
        | Tuple[DataActivationsGradients, DataActivationsGradients],
    ) -> ApproximationData | None:
        """
        Method which takes the provided compute context and performs any additional,
        approach specific pre-computations. For (E)K-FAC, this includes computing the
        eigenvectors, corrections, etc.

        If no specific additional computations are required,
        the method simply returns the object itself.

        Should be overridden by subclasses which require additional computations.
        """
        return None

    def estimate_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "n_params n_params"]:
        """Compute Hessian approximation."""
        if not self.is_built:
            raise RuntimeError(
                "HessianEstimator not built. Please call the 'build' method before estimating the Hessian."
            )
        return self._estimate_hessian(damping)

    @abstractmethod
    def _estimate_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "n_params n_params"]:
        """Compute Hessian approximation."""
        pass

    def compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """Compare Hessian approximation with another Hessian matrix."""
        if not self.is_built:
            raise RuntimeError(
                "HessianEstimator not built. Please call the 'build' method before comparing Hessian estimates."
            )
        return self._compare_full_hessian_estimates(comparison_matrix, damping, metric)

    @abstractmethod
    def _compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """Compare Hessian approximation with another Hessian matrix."""
        pass

    def estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Hessian-vector product."""
        if not self.is_built:
            raise RuntimeError(
                "HessianEstimator not built. Please call the 'build' method before estimating the Hessian-vector product."
            )
        return self._estimate_hvp(vectors, damping)

    @abstractmethod
    def _estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Hessian-vector product."""
        pass

    def estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Inverse Hessian-vector product."""
        if not self.is_built:
            raise RuntimeError(
                "HessianEstimator not built. Please call the 'build' method before estimating the Inverse Hessian-vector product."
            )
        return self._estimate_ihvp(vectors, damping)

    @abstractmethod
    def _estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Inverse Hessian-vector product."""
        pass

    def data_exists(self, directory: str) -> bool:
        """Check if the data and config files exist in the specified directory."""
        data_path = Path(directory) / "data.npz"
        config_path = Path(directory) / "config.json"
        return data_path.exists() and config_path.exists()


@dataclass
class ModelBasedHessianEstimator(HessianEstimator):
    compute_context: ModelContext

    @staticmethod
    def _build(compute_context: ModelContext) -> ApproximationData | None:
        """
        Method which takes the provided compute context and performs any additional,
        approach specific pre-computations. For (E)K-FAC, this includes computing the
        eigenvectors, corrections, etc.

        If no specific additional computations are required,
        the method simply returns the object itself.

        Should be overridden by subclasses which require additional computations.
        """
        return None


@dataclass
class CollectorBasedHessianEstimator(HessianEstimator):
    compute_context: Tuple[DataActivationsGradients, DataActivationsGradients]

    @staticmethod
    def _build(
        compute_context: Tuple[DataActivationsGradients, DataActivationsGradients],
    ) -> ApproximationData | None:
        """
        Method which takes the provided compute context and performs any additional,
        approach specific pre-computations. For (E)K-FAC, this includes computing the
        eigenvectors, corrections, etc.

        If no specific additional computations are required,
        the method simply returns the object itself.

        Should be overridden by subclasses which require additional computations.
        """
        return None
