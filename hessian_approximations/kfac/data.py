from dataclasses import dataclass, field
from typing import Dict

import jax.numpy as jnp
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
