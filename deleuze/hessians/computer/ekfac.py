from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import override

from deleuze.hessians.computer.computer import HessianComputer
from deleuze.hessians.utils.data import EKFACData
from deleuze.metrics.full_matrix_metrics import FullMatrixMetric

from .kfac import KFACComputer


@dataclass
class EKFACComputer(HessianComputer):
    """
    Kronecker-Factored Approximate Curvature (KFAC) and Eigenvalue-Corrected KFAC (EKFAC) Hessian approximation.
    """

    compute_context: EKFACData

    @override
    def compute_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute full Hessian approximation.
        """
        return self.compute_hessian_or_inverse_hessian(
            method="normal",
            damping=0.0 if damping is None else damping,
        )

    @override
    def compute_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Hessian-vector product."""
        return self.compute_ihvp_or_hvp(
            vectors,
            method="hvp",
            damping=0.0 if damping is None else damping,
        )

    @override
    def compare_hessians(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> float:
        """
        Compare the EKFAC Hessian approximation to a given comparison matrix.
        Reuses the KFAC comparison implementation.
        """
        return KFACComputer._compare_hessians(
            activations_eigenvectors=list(
                self.compute_context.activation_eigenvectors.values()
            ),
            gradients_eigenvectors=list(
                self.compute_context.gradient_eigenvectors.values()
            ),
            Lambdas=list(self.compute_context.eigenvalue_corrections.values()),
            damping=0.0 if damping is None else damping,
            comparison_matrix=comparison_matrix,
            metric=metric.compute_fn(),
            method="normal",
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
        return self.compute_ihvp_or_hvp(
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
        return self.compute_hessian_or_inverse_hessian(
            method="inverse",
            damping=0.0 if damping is None else damping,
        )

    def compute_hessian_or_inverse_hessian(
        self,
        method: Literal["normal", "inverse"],
        damping: Float,
    ) -> Float[Array, "n_params n_params"]:
        """
        Unified helper method to compute either the full Hessian or its inverse.
        Reuses the KFAC implementation.

        Depending on configuration, this computes either:
        - (E)KFAC Hessians with eigenvalue corrections, or
        - Standard KFAC Hessians using Kronecker-factored covariances.

        Note:
        PyTorch assumes weights shaped [d_out, d_in] with vec(∇W) = a ⊗ ∇s
        (column-major). In contrast, JAX uses [d_in, d_out], which yields
        vec(∇W') = ∇s ⊗ a due to the forward pass formulation y = xW'
        instead of y = Wx (as in PyTorch).

        Because JAX flattens arrays in row-major (C-style) order, the effective
        vectorization swaps again, giving vec_row(∇W') = a ⊗ ∇s. This matches
        the ordering used when comparing with the true Hessian or constructing
        Kronecker-factored curvature blocks.
        """
        return KFACComputer._compute_hessian_or_inverse_hessian(
            eigenvectors_activations=list(
                self.compute_context.activation_eigenvectors.values()
            ),
            eigenvectors_gradients=list(
                self.compute_context.gradient_eigenvectors.values()
            ),
            Lambdas=list(self.compute_context.eigenvalue_corrections.values()),
            damping=damping,
            method=method,
        )

    def compute_ihvp_or_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        method: Literal["ihvp", "hvp"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute inverse Hessian-vector product or Hessian-vector product.
        Reuses the KFAC implementation.

        Note, that the vector to be multiplied is reshaped in row-major order to
        match the JAX weight layout which is reflected by
        the eigenvalue corrections shape [input_dim, output_dim].
        """

        # Collect all layer data upfront
        layer_names = self.compute_context.layer_names()

        Q_activations_list = []
        Q_gradients_list = []
        Lambda_list = []
        v_layers = []

        offset = 0
        for layer_name in layer_names:
            Lambda: Float[Array, "I O"] = self.compute_context.eigenvalue_corrections[
                layer_name
            ]
            input_dim, output_dim = Lambda.shape
            size = input_dim * output_dim

            # Extract and reshape vector for this layer
            v_flat: Float[Array, "*batch_size I*O"] = vectors[
                ..., offset : offset + size
            ]
            v_layer: Float[Array, "*batch_size I O"] = v_flat.reshape(
                v_flat.shape[:-1] + (input_dim, output_dim)
            )

            # Collect all components
            Q_activations_list.append(
                self.compute_context.activation_eigenvectors[layer_name]
            )
            Q_gradients_list.append(
                self.compute_context.gradient_eigenvectors[layer_name]
            )
            Lambda_list.append(Lambda)
            v_layers.append(v_layer)
            offset += size

        # Process all layers at once
        vp_pieces = KFACComputer.compute_ihvp_or_hvp_all_layers(
            v_layers=v_layers,
            Q_activations=Q_activations_list,
            Q_gradients=Q_gradients_list,
            Lambdas=Lambda_list,
            damping=damping,
            method=method,
        )

        # Concatenate all layer results
        return jnp.concatenate(vp_pieces, axis=-1)
