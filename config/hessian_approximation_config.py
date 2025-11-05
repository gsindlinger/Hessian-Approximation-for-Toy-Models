from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


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


@dataclass
class LiSSAConfig(HessianApproximationConfig):
    name: HessianName = field(default=HessianName.LISSA, init=False)
    num_samples: int = 3
    recursion_depth: int = 500
    alpha: float = 0.05
    damping: float = 0.001
    batch_size: int = 128
    seed: int = 42
    convergence_tol: float = 1e-6
    check_convergence_every: int = 50


@dataclass
class KFACBuildConfig:
    use_pseudo_targets: bool = False
    pseudo_target_noise_std: float = 0.1
    collector_batch_size: int | None = None
    recalc_ekfac_components: bool = False


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
