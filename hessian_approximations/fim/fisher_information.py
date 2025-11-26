from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import override

from config.hessian_approximation_config import FisherInformationConfig
from hessian_approximations.hessian_approximations import HessianApproximation
from hessian_approximations.kfac.kfac import KFAC
from models.dataclasses.hessian_compute_context import HessianComputeContext
from models.utils.loss import get_loss_name


@dataclass
class FisherInformation(HessianApproximation):
    """
    Fisher Information Matrix approximation.

    The Fisher Information Matrix is defined as:
    FIM = E[∇log p(y|x) ∇log p(y|x)^T]

    Two variants are supported:
    - Empirical FIM: Uses actual training data and labels
    - True FIM: Samples labels from the model's predictive distribution

    The score ∇log p(y|x) is computed differently for each loss type.
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

        if get_loss_name(self.model_context.loss) == "cross_entropy":
            return self._compute_crossentropy_fim(compute_data, damping)
        else:
            return self._compute_mse_fim(compute_data, damping)

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

        if get_loss_name(self.model_context.loss) == "cross_entropy":
            result_2D = self._compute_ifvp_batched_cross_entropy(
                compute_data, vectors_2D, damping
            )
        else:
            result_2D = self._compute_ifvp_batched_mse(
                compute_data, vectors_2D, damping
            )
        return result_2D.squeeze(0) if is_single else result_2D

    @staticmethod
    @jax.jit
    def _compute_mse_fim(
        compute_data: HessianComputeContext,
        damping: Float,
    ) -> Float[Array, "n_params n_params"]:
        x = compute_data.training_data
        y = compute_data.training_targets
        p0 = compute_data.params_flat
        N = x.shape[0]

        unravel_fn = compute_data.unravel_fn
        model_apply_fn = compute_data.model_apply_fn
        loss = compute_data.loss_fn

        # Per-example gradient of log-likelihood
        def per_example_grad(p_flat, x_i, y_i):
            return jax.grad(FisherInformation.log_likelihood, argnums=0)(
                p_flat,
                x_i,
                y_i,
                unravel_fn,
                model_apply_fn,
                loss,
            )

        grads = jax.vmap(per_example_grad, in_axes=(None, 0, 0))(p0, x, y)
        # grads: (N, D)

        fim = (grads.T @ grads) / N

        return fim + damping * jnp.eye(fim.shape[0])

    @staticmethod
    @jax.jit
    def _compute_crossentropy_fim(compute_data: HessianComputeContext, damping: Float):
        x, y = compute_data.training_data, compute_data.training_targets
        p0 = compute_data.params_flat
        n = x.shape[0]

        def log_likelihood(p_flat, xi, yi):
            params = compute_data.unravel_fn(p_flat)
            logits = compute_data.model_apply_fn(params, xi[None, ...])[0]
            log_probs = jax.nn.log_softmax(logits)
            return log_probs[yi] if yi.ndim == 0 else jnp.sum(yi * log_probs)

        # per-example grad: shape (N, D)
        grads = jax.vmap(jax.grad(log_likelihood), in_axes=(None, 0, 0))(p0, x, y)

        # compute FIM
        fim = (grads.T @ grads) / n
        return fim + damping * jnp.eye(fim.shape[0])

    @staticmethod
    @jax.jit
    def _compute_ifvp_batched_cross_entropy(
        compute_data: HessianComputeContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        # Take the simplest approach: calculate full FIM and solve by linalg solve
        fim = FisherInformation._compute_crossentropy_fim(compute_data, damping)
        return jnp.linalg.solve(fim, vectors.T).T

    @staticmethod
    @jax.jit
    def _compute_ifvp_batched_mse(
        compute_data: HessianComputeContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        # Take the simplest approach: calculate full FIM and solve by linalg solve
        fim = FisherInformation._compute_mse_fim(compute_data, damping)
        return jnp.linalg.solve(fim, vectors.T).T

    @staticmethod
    @jax.jit
    def _compute_fvp(
        compute_data: HessianComputeContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Unified Fisher-vector product (FVP) for ANY loss function.
        Uses class-level `log_likelihood` to compute per-example gradients.
        """
        x, y = compute_data.training_data, compute_data.training_targets
        p0 = compute_data.params_flat
        N = x.shape[0]

        # Per-example score vectors
        grads = jax.vmap(
            jax.grad(FisherInformation.log_likelihood, argnums=0),
            in_axes=(None, 0, 0, None, None, None),
        )(
            p0,
            x,
            y,
            compute_data.unravel_fn,
            compute_data.model_apply_fn,
            compute_data.loss_fn,
        )  # shape: (N, D)

        # Compute Fv for each vector in the batch
        def fvp_single(v):
            proj = grads @ v  # (N,)
            fvp = grads.T @ proj / N  # (D,)
            return fvp + damping * v  # damping regularization

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
                )
            )
        return compute_data

    @staticmethod
    @partial(jax.jit, static_argnames=("loss", "unravel_fn", "model_apply_fn"))
    def log_likelihood(
        p_flat: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        unravel_fn: Callable,
        model_apply_fn: Callable,
        loss: Callable,
    ):
        params = unravel_fn(p_flat)
        preds = model_apply_fn(params, x)
        if get_loss_name(loss) == "mse":
            return -0.5 * jnp.sum((y - preds) ** 2)
        elif get_loss_name(loss) == "cross_entropy":
            log_probs = jax.nn.log_softmax(preds)
            if y.ndim == 0:
                return log_probs[y]
            else:
                return jnp.sum(y * log_probs)
        else:
            raise ValueError(f"Unsupported loss for log-likelihood: {loss}")
