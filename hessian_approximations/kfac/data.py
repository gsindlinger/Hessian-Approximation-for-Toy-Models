from dataclasses import dataclass, field
from typing import Dict, List

import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float

from hessian_approximations.kfac.layer_components import LayerComponents


@dataclass
class KFACData:
    covariances: LayerComponents = field(default_factory=LayerComponents)
    eigenvectors: LayerComponents = field(default_factory=LayerComponents)
    eigenvalues: LayerComponents = field(default_factory=LayerComponents)
    eigenvalue_corrections: Dict[str, Float[Array, "d_in d_out"]] = field(
        default_factory=dict
    )

    def __bool__(self) -> bool:
        return bool(
            bool(self.covariances)
            and bool(self.eigenvectors)
            and bool(self.eigenvalues)
            and self.eigenvalue_corrections != {}
        )


@dataclass
class KFACMeanEigenvaluesAndCorrections:
    eigenvalues: Dict[str, Float[Array, ""]] = field(default_factory=dict)
    corrections: Dict[str, Float[Array, ""]] = field(default_factory=dict)
    overall_mean_eigenvalues: Float[Array, ""] = field(
        default_factory=lambda: jnp.array(0.0)
    )
    overall_mean_corrections: Float[Array, ""] = field(
        default_factory=lambda: jnp.array(0.0)
    )

    def __bool__(self) -> bool:
        return bool(
            bool(self.eigenvalues)
            and bool(self.corrections)
            and self.eigenvalues != {}
            and self.corrections != {}
            and self.overall_mean_eigenvalues != 0.0
            and self.overall_mean_corrections != 0.0
        )


@struct.dataclass
class KFACJITData:
    eigenvectors_A: List[Float[Array, "d_in d_out"]]
    eigenvectors_G: List[Float[Array, "d_in d_out"]]
    Lambdas: List[Float[Array, "d_in d_out"]]
