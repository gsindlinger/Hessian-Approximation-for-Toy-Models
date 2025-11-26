from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import override

from config.hessian_approximation_config import FisherInformationConfig
from hessian_approximations.hessian_approximations import HessianApproximation
from hessian_approximations.kfac.kfac import KFAC
from metrics.full_matrix_metrics import FullMatrixMetric
from models.dataclasses.hessian_compute_context import HessianComputeContext


@dataclass
class FisherInformation(HessianApproximation):
    """
    Fisher Information Matrix approximation.

    The Fisher Information Matrix is defined as:
    FIM = E[∇log p(y|x) ∇log p(y|x)^T]

    Two variants are supported:
    - Empirical FIM: Uses actual training data and labels
    - True FIM: Samples labels from the model's predictive distribution
    """

    fisher_config: FisherInformationConfig = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        if not self.full_config.hessian_approximation:
            self.full_config.hessian_approximation = FisherInformationConfig()

        if not isinstance(
            self.full_config.hessian_approximation, FisherInformationConfig
        ):
            raise ValueError(
                "FisherInformation requires a FisherInformationConfig in full_config.hessian_approximation"
            )
        self.fisher_config = self.full_config.hessian_approximation

    @override
    def compute_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> jnp.ndarray:
        """
        Compute the Fisher Information Matrix.
        """
        compute_data = self.get_compute_data_with_pseudo_targets()
        damping = 0.0 if damping is None else damping

        return self._compute_fim(compute_data, damping)

    def compare_hessians(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """
        Compare the Fisher Information Matrix with another Hessian matrix using the specified metric.
        """
        compute_data = self.get_compute_data_with_pseudo_targets()
        damping = 0.0 if damping is None else damping

        return metric.compute_fn()(
            comparison_matrix,
            self._compute_fim(compute_data, damping),
        )

    @staticmethod
    @jax.jit
    def _compute_fim(
        compute_data: HessianComputeContext,
        damping: Float = 0.0,
    ) -> Float[Array, "n_params n_params"]:
        def loss_single(p, x, y):
            params_unflat = compute_data.unravel_fn(p)
            preds = compute_data.model_apply_fn(params_unflat, x[None, ...])
            return compute_data.loss_fn(preds.squeeze(0), y)

        # per-sample gradient of loss
        grad_loss = jax.grad(loss_single)

        @jax.jit
        def compute_sample_fim(p_flat, x, y):
            g = grad_loss(p_flat, x, y)  # shape = (n_params,)
            return jnp.outer(g, g)  # (n_params, n_params)

        def scan_body(carry, xy):
            p_flat, F = carry
            x_i, y_i = xy
            F_i = compute_sample_fim(p_flat, x_i, y_i)
            return (p_flat, F + F_i), None

        p_flat = compute_data.params_flat
        X = compute_data.training_data
        Y = compute_data.training_targets

        F0 = jnp.zeros((p_flat.size, p_flat.size))

        (_, F_full), _ = jax.lax.scan(scan_body, init=(p_flat, F0), xs=(X, Y))

        F_full = F_full / X.shape[0]  # average

        # add damping like in your Hessian
        return F_full + damping * jnp.eye(F_full.shape[0])

    @override
    def compute_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the Fisher-vector product (FVP).
        """
        compute_data = self.get_compute_data_with_pseudo_targets()
        damping = 0.0 if damping is None else damping

        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )
        result = self._compute_fvp(compute_data, vectors_2D, damping)
        return result.squeeze(0) if is_single else result

    @override
    def compute_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the inverse Fisher-vector product (IFVP) using Conjugate Gradient.
        """
        compute_data = self.get_compute_data_with_pseudo_targets()
        damping = 0.0 if damping is None else damping

        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        result_2D = self._compute_ifvp_batched(compute_data, vectors_2D, damping)
        return result_2D.squeeze(0) if is_single else result_2D

    @staticmethod
    @jax.jit
    def _compute_ifvp_batched(
        compute_data: HessianComputeContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        # Take the simplest approach: calculate full FIM and solve by linalg solve
        fim = FisherInformation._compute_fim(compute_data, damping)
        return jnp.linalg.solve(fim, vectors.T).T

    @staticmethod
    @jax.jit
    def _compute_fvp(
        compute_data: HessianComputeContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        x, y = compute_data.training_data, compute_data.training_targets
        p0 = compute_data.params_flat
        N = x.shape[0]

        # per-sample loss
        def loss_single(p, xi, yi):
            params = compute_data.unravel_fn(p)
            preds = compute_data.model_apply_fn(params, xi[None, ...])
            return compute_data.loss_fn(preds.squeeze(0), yi)

        # per-sample grads of the loss
        grads = jax.vmap(lambda xi, yi: jax.grad(loss_single)(p0, xi, yi))(
            x, y
        )  # shape (N, D)

        def fvp_single(v):
            proj = grads @ v  # (N,)
            fvp = grads.T @ proj / N  # (D,)
            return fvp + damping * v

        return jax.vmap(fvp_single)(vectors)

    def get_compute_data_with_pseudo_targets(
        self,
    ) -> HessianComputeContext:
        compute_data = HessianComputeContext.get_data_and_params_for_hessian(
            self.model_context
        )
        # Generate pseudo-targets if using true FIM
        if self.fisher_config.fisher_type == "true":
            compute_data = compute_data.replace(  # type: ignore
                training_targets=KFAC.generate_pseudo_targets(
                    model=self.model_context.model,
                    training_data=compute_data.training_data,
                    params=self.model_context.params,
                    loss_fn=self.model_context.loss,
                    rng_key=jax.random.PRNGKey(self.full_config.seed + 111),
                )
            )
        return compute_data
