from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Literal

from typing_extensions import override


class HessianName(str, Enum):
    EXACT_HESSIAN_REGRESSION = "exact-hessian-regression"
    HESSIAN = "hessian"
    FIM = "fim"
    GAUSS_NEWTON = "gauss-newton"
    KFAC = "kfac"
    LISSA = "lissa"


@dataclass
class HessianApproximationConfig:
    """Configuration for Hessian approximation."""

    name: HessianName

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LiSSAConfig(HessianApproximationConfig):
    name: HessianName = field(default=HessianName.LISSA, init=False)
    num_samples: int = 3
    recursion_depth: int = 500
    alpha: float = 0.05
    damping: float = 0.001
    batch_size: int = 128
    convergence_tol: float = 1e-6
    check_convergence_every: int = 50


@dataclass
class FisherInformationConfig(HessianApproximationConfig):
    name: HessianName = field(default=HessianName.FIM, init=False)
    fisher_type: Literal["empirical", "true"] = "true"


@dataclass
class GaussNewtonHessianConfig(HessianApproximationConfig):
    name: HessianName = field(default=HessianName.GAUSS_NEWTON, init=False)


@dataclass
class KFACBuildConfig:
    use_pseudo_targets: bool = True
    pseudo_target_noise_std: float = 0.1
    collector_batch_size: int | None = None
    recalc_ekfac_components: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        ignored = {"recalc_ekfac_components", "collector_batch_size"}
        return {k: v for k, v in data.items() if k not in ignored}


@dataclass
class KFACRunConfig:
    damping_lambda: float = 0.1
    damping_mode: Literal["mean_eigenvalue", "mean_corrections"] = "mean_eigenvalue"
    use_eigenvalue_correction: bool = True
    recalc_kfac_components: bool = False


@dataclass
class KFACConfig(HessianApproximationConfig):
    name: HessianName = field(default=HessianName.KFAC, init=False)
    build_config: KFACBuildConfig = field(default_factory=KFACBuildConfig)
    run_config: KFACRunConfig = field(default_factory=KFACRunConfig)

    @property
    def recalc_ekfac_components(self) -> bool:
        return (
            self.build_config.recalc_ekfac_components
            or self.run_config.recalc_kfac_components
        )

    @override
    def to_dict(self) -> dict:
        return self.build_config.to_dict()
