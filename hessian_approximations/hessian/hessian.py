from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import flatten_util
from typing_extensions import override

from hessian_approximations.hessian_approximations import HessianApproximation
from models.train import ModelData
from models.utils.loss import loss_wrapper_flattened, loss_wrapper_with_apply_fn


@dataclass
class Hessian(HessianApproximation):
    """Hessian Calculation via automatic differentiation (JAX native)."""

    @staticmethod
    def get_param_index_mapping(params: Dict):
        """
        Build a mapping from parameter names to index ranges in the flattened vector.
        Helps to debug by identifying which parameter in the flattened parameter array corresponds to which entry.
        Returns:
            dict[str, tuple[int, int]] mapping each param path to (start, end)
        """
        leaves, _ = jax.tree_util.tree_flatten(params)
        flat_params, _ = flatten_util.ravel_pytree(params)

        index_mapping = {}
        idx = 0

        # Traverse parameter tree with paths
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
            path_str = "/".join(str(k) for k in path)
            size = leaf.size
            index_mapping[path_str] = (idx, idx + size)
            idx += size

        return index_mapping, flat_params.size

    @override
    def compute_hessian(
        self,
    ) -> jnp.ndarray:
        """
        Compute the exact Hessian matrix of the loss function for the whole model.

        Args:
            model: The Flax model.
            params: Model parameters (PyTree structure).
            training_data: Input data.
            training_targets: Target values.
            loss_fn: Loss function (e.g., mse_loss or cross_entropy_loss).

        Returns:
            Hessian matrix as a 2D array.
        """
        (
            training_data,
            training_targets,
            params_flat,
            unravel_fn,
            model_apply_fn,
            loss_fn,
        ) = self.get_data_and_params_for_hessian(self.model_data)

        return self.compute_hessian_jitted(
            params_flat,
            unravel_fn,
            training_data,
            training_targets,
            model_apply_fn,
            loss_fn,
        )

    @staticmethod
    def get_data_and_params_for_hessian(
        model_data: ModelData,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Callable, Callable, Callable]:
        # Important: Flattening structure for linear modules with bias is the following: b, w
        # So for output dim 2, input dim 3, the order is: b1, b2, w1
        training_data, training_targets = model_data.dataset.get_train_data()
        params_flat, unravel_fn = flatten_util.ravel_pytree(model_data.params)
        return (
            training_data,
            training_targets,
            params_flat,
            unravel_fn,
            model_data.model.apply,
            model_data.loss,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["model_apply_fn", "loss_fn", "unravel_fn"])
    def compute_hessian_jitted(
        params_flat: jnp.ndarray,
        unravel_fn: Callable,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        model_apply_fn: Callable,
        loss_fn: Callable,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        JIT-compiled Hessian computation.

        Returns the Hessian matrix as a 2D array.
        """
        # Important: Flattening structure for linear modules with bias is the following: b, w
        # So for output dim 2, input dim 3, the order is: b1, b2, w1

        def loss_wrapper(p):
            return loss_wrapper_with_apply_fn(
                model_apply_fn,
                p,
                unravel_fn,
                loss_fn,
                training_data,
                training_targets,
            )

        # JIT and return Hessian function
        hessian_fn = jax.hessian(loss_wrapper)
        return hessian_fn(params_flat)

    @override
    def compute_hvp(
        self,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the Hessian-vector product (HVP) using JAX's automatic differentiation.

        Args:
            model: The Flax model.
            params: Model parameters (PyTree structure).
            training_data: Input data.
            training_targets: Target values.
            loss_fn: Loss function (e.g., mse_loss or cross_entropy_loss).
            vector: Vector to multiply with the Hessian.

        Returns:
            HVP result as a 1D array.
        """
        training_data, training_targets = self.model_data.dataset.get_train_data()

        # Flatten parameters once
        params_flat, unravel_fn = flatten_util.ravel_pytree(self.model_data.params)

        def loss_wrapper(p):
            return loss_wrapper_flattened(
                self.model_data.model,
                p,
                unravel_fn,
                self.model_data.loss,
                training_data,
                training_targets,
            )

        # Use jax.jvp for efficient Hessian-vector product
        _, hvp_result = jax.jvp(
            lambda p: jax.grad(loss_wrapper)(p),
            (params_flat,),
            (vector,),
        )

        return jnp.asarray(hvp_result)

    @override
    def compute_ihvp(
        self,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the inverse Hessian-vector product (IHVP) using JAX's automatic differentiation.

        Args:
            model: The Flax model.
            params: Model parameters (PyTree structure).
            training_data: Input data.
            training_targets: Target values.
            loss_fn: Loss function (e.g., mse_loss or cross_entropy_loss).
            vector: Vector to multiply with the inverse Hessian.

        Returns:
            IHVP result as a 1D array.
        """
        training_data, training_targets = self.model_data.dataset.get_train_data()

        # Flatten parameters once
        params_flat, unravel_fn = flatten_util.ravel_pytree(self.model_data.params)

        def loss_wrapper(p):
            return loss_wrapper_flattened(
                self.model_data.model,
                p,
                unravel_fn,
                self.model_data.loss,
                training_data,
                training_targets,
            )

        # Compute Hessian
        hessian_fn = jax.jit(jax.hessian(loss_wrapper))
        hessian = hessian_fn(params_flat)

        # Solve H x = v for x
        ihvp_result = jnp.linalg.solve(hessian, vector)

        return jnp.asarray(ihvp_result)
