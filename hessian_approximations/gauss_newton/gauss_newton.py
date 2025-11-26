from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import override

from config.hessian_approximation_config import GaussNewtonHessianConfig
from hessian_approximations.hessian_approximations import HessianApproximation
from metrics.full_matrix_metrics import FullMatrixMetric
from models.dataclasses.hessian_compute_context import HessianComputeContext
from models.utils.loss import get_loss_name


@dataclass
class GaussNewton(HessianApproximation):
    """
    Gauss-Newton Hessian approximation.

    The Gauss-Newton approximation is defined as:
    GNH = J^T H_L J

    where:
    - J is the Jacobian of the model output w.r.t. parameters
    - H_L is the Hessian of the loss w.r.t. the model output

    For exponential family losses (e.g., CrossEntropy), GNH equals FIM.
    GNH is always positive semi-definite, unlike the full Hessian.
    """

    def __post_init__(self):
        super().__post_init__()
        if not self.full_config.hessian_approximation:
            self.full_config.hessian_approximation = GaussNewtonHessianConfig()

    @override
    def compute_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> jnp.ndarray:
        """
        Compute the Generalized Gauss-Newton approximation of the Hessian.
        """

        compute_data = HessianComputeContext.get_data_and_params_for_hessian(
            self.model_context
        )
        damping = damping if damping is not None else 0.0
        return self._compute_gnh(compute_data, damping)

    def compare_hessians(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """
        Compare the Gauss-Newton Hessian with another Hessian matrix using the specified metric.
        """
        compute_data = HessianComputeContext.get_data_and_params_for_hessian(
            self.model_context
        )
        damping = 0.0 if damping is None else damping

        return metric.compute_fn()(
            comparison_matrix,
            self._compute_gnh(compute_data, damping),
        )

    @override
    def compute_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the Gauss-Newton-vector product (GNVP).
        """
        compute_data = HessianComputeContext.get_data_and_params_for_hessian(
            self.model_context
        )

        damping = damping if damping is not None else 0.0
        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        result_2D = self._compute_gnhvp(compute_data, vectors_2D, damping)
        return result_2D.squeeze(0) if is_single else result_2D

    @override
    def compute_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> jnp.ndarray:
        """
        Compute the inverse Gauss-Newton-vector product (GNVP).
        """
        result = self._compute_ihvp_batched(
            data=HessianComputeContext.get_data_and_params_for_hessian(
                self.model_context
            ),
            vectors=vectors,
            damping=0.0 if damping is None else damping,
            loss_name=get_loss_name(self.model_context.loss),
        )
        return result

    @staticmethod
    @partial(jax.jit, static_argnames=("loss_name",))
    def _compute_ihvp_batched(
        data: HessianComputeContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
        loss_name: str,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Compute inverse Gauss-Newton-vector product (IHVP) using JAX's automatic differentiation.

        Args:
            model: The Flax model.
            params: Model parameters (PyTree structure).
            training_data: Input data.
            training_targets: Target values.
            loss_fn: Loss function (e.g., mse_loss or cross_entropy_loss).
            vector: Vector to multiply with the inverse Gauss-Newton matrix.

        Returns:
            IHVP result as a 1D array.
        """
        gnh = GaussNewton._compute_gnh(data, damping)

        return jnp.linalg.solve(gnh, vectors.T).T

    @staticmethod
    @jax.jit
    def _compute_gnh(
        compute_data: HessianComputeContext, damping: Float
    ) -> Float[Array, "n_params n_params"]:
        """
        Computes full Gauss–Newton Hessian for ANY convex loss w.r.t. outputs.
        Works for:
        - Cross-entropy (softmax logits)
        - Mean squared error
        - Any loss where L(z, y) is convex in z.
        """

        def model_out(p_flat, x):
            params_unflat = compute_data.unravel_fn(p_flat)
            return compute_data.model_apply_fn(params_unflat, x[None, ...]).squeeze(0)

        def loss_wrt_output(z, y):
            return compute_data.loss_fn(z, y)

        @jax.jit
        def per_sample_gn(p_flat, x_i, y_i):
            z = model_out(p_flat, x_i)  # model output for sample i
            H_z = jax.hessian(lambda z_: loss_wrt_output(z_, y_i))(z)

            def logits_fn(p):
                return model_out(p, x_i)  # model output function

            J = jax.jacrev(logits_fn)(p_flat)  # Jacobian of model output w.r.t. params

            return J.T @ H_z @ J

        # Loop through data
        def scan_body(carry, xy):
            p_flat, G = carry
            x_i, y_i = xy
            G_i = per_sample_gn(p_flat, x_i, y_i)
            return (p_flat, G + G_i), None

        p_flat = compute_data.params_flat
        X = compute_data.training_data
        Y = compute_data.training_targets
        n_params = p_flat.size

        G0 = jnp.zeros((n_params, n_params))

        (_, G_full), _ = jax.lax.scan(scan_body, init=(p_flat, G0), xs=(X, Y))

        # Average over dataset + damping
        G_full = G_full / X.shape[0]
        return G_full + damping * jnp.eye(n_params)

    @staticmethod
    @jax.jit
    def _compute_gnhvp(
        compute_data: HessianComputeContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: float,
    ) -> jnp.ndarray:
        """
        Computes Gauss-Newton Hessian-vector products.
        """

        p_flat = compute_data.params_flat  # shape: [n_params]
        X = compute_data.training_data  # shape: [n_samples, ...]
        Y = compute_data.training_targets  # shape: [n_samples, ...]

        def model_out(p, x):
            params = compute_data.unravel_fn(p)
            return compute_data.model_apply_fn(params, x[None, ...]).squeeze(0)

        def loss_wrt_output(z, y):
            return compute_data.loss_fn(z, y)

        def gnhvp_single_data_point(p_flat, x_i, y_i, v_batch):
            z = model_out(p_flat, x_i)

            H_z = jax.hessian(lambda z_: loss_wrt_output(z_, y_i))(
                z
            )  # Hessian of loss wrt outputs

            def logits_fn(p):
                return model_out(p, x_i)

            J = jax.jacrev(logits_fn)(
                p_flat
            )  # Jacobian of model outputs wrt parameters
            Jv = v_batch @ J.T
            HJv = Jv @ H_z.T
            Gv = HJv @ J

            return Gv

        # vmap over dataset and sum contributions
        Gv_sum = jax.vmap(
            lambda x_i, y_i: gnhvp_single_data_point(p_flat, x_i, y_i, vectors),
            in_axes=(0, 0),
        )(X, Y).sum(axis=0)

        # Average over batch + damping
        n = X.shape[0]
        return Gv_sum / n + damping * vectors
