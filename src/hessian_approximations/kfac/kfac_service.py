from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from jaxtyping import Array, Float
from typing_extensions import override

from ...config.config import Config
from ...config.hessian_approximation_config import (
    KFACBuildConfig,
    KFACConfig,
    KFACRunConfig,
)
from ...metrics.full_matrix_metrics import FullMatrixMetric
from ..hessian_approximations import HessianApproximation
from .kfac_computer import KFACComputer
from .kfac_data_provider import KFACProvider

logger = logging.getLogger(__name__)


@dataclass
class KFAC(HessianApproximation):
    """
    Kronecker-Factored Approximate Curvature (KFAC) and Eigenvalue-Corrected KFAC (EKFAC) Hessian approximation.
    """

    full_config: Config
    provider: KFACProvider = field(init=False)
    computer: KFACComputer = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        if not self.full_config.hessian_approximation:
            self.full_config.hessian_approximation = KFACConfig()

        if not isinstance(self.full_config.hessian_approximation, KFACConfig):
            raise ValueError(
                "KFAC Hessian approximation requires KFACConfig in the config."
            )

        self.provider = KFACProvider(
            model_context=self.model_context,
            configs=(
                self.full_config.model,
                self.full_config.dataset,
                self.full_config.training,
                self.full_config.hessian_approximation.build_config,
            ),
        )
        self.computer = KFACComputer(
            config=self.full_config.hessian_approximation.run_config
        )

    @classmethod
    def setup_with_run_and_build_config(
        cls,
        full_config: Config,
        run_config: KFACRunConfig | None = None,
        build_config: KFACBuildConfig | None = None,
    ) -> KFAC:
        """Setup KFAC instance with given run or build configuration.
        If either is None, the one from the initial config is used."""
        # create copy of config to ensure that a new instance is created
        full_config = copy.deepcopy(full_config)
        if not full_config.hessian_approximation:
            full_config.hessian_approximation = KFACConfig()
        elif not isinstance(full_config.hessian_approximation, KFACConfig):
            raise ValueError(
                "KFAC Hessian approximation requires KFACConfig in the config."
            )
        kfac_config = full_config.hessian_approximation

        if build_config is not None:
            kfac_config.build_config = build_config
        if run_config is not None:
            kfac_config.run_config = run_config
        full_config.hessian_approximation = kfac_config
        return cls(full_config=full_config)

    @override
    def compute_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute full Hessian approximation.
        """
        return self.computer.compute_hessian_or_inverse_hessian(
            provider=self.provider,
            damping=0.0 if damping is None else damping,
            method="normal",
        )

    @override
    def compute_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Hessian-vector product."""
        return self.computer.compute_ihvp_or_hvp(
            provider=self.provider,
            vectors=vectors,
            method="hvp",
            damping=0.0 if damping is None else damping,
        )

    @override
    def compute_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute inverse Hessian-vector product.
        """
        return self.computer.compute_ihvp_or_hvp(
            provider=self.provider,
            vectors=vectors,
            method="ihvp",
            damping=0.0 if damping is None else damping,
        )

    def compute_inverse_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute full inverse Hessian.
        """
        return self.computer.compute_hessian_or_inverse_hessian(
            provider=self.provider,
            method="inverse",
            damping=0.0 if damping is None else damping,
        )

    def compare_hessians(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> float:
        """
        Compare the (E)KFAC Hessian approximation to a given comparison matrix
        """
        return self.computer.compare_hessians(
            provider=self.provider,
            damping=0.0 if damping is None else damping,
            comparison_matrix=comparison_matrix,
            metric=metric.compute_fn(),
            method="normal",
        )

    def damping(
        self,
        base_damping: Optional[Float] = None,
        method: Literal["eigenvalues", "corrections"] = "eigenvalues",
    ) -> Float[Array, ""]:
        return self.provider.get_damping(
            base_damping=self.computer.config.damping_lambda
            if base_damping is None
            else base_damping,
            method=method,
        )
