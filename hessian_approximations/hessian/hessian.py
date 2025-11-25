from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import Array, Float
from typing_extensions import override

from hessian_approximations.hessian_approximations import HessianApproximation
from models.dataclasses.hessian_jit_data import HessianJITData
from models.utils.loss import loss_wrapper_with_apply_fn


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
        damping: Optional[Float] = None,
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
        return self.compute_hessian_jitted(
            HessianJITData.get_data_and_params_for_hessian(self.model_data),
            damping=0.0 if damping is None else damping,
        )

    @staticmethod
    @jax.jit
    def compute_hessian_jitted(
        data: HessianJITData,
        damping: Float,
    ) -> Float[Array, "n_params n_params"]:
        """
        JIT-compiled Hessian computation.

        Returns the Hessian matrix as a 2D array.
        """
        # Important: Flattening structure for linear modules with bias is the following: b, w
        # So for output dim 2, input dim 3, the order is: b1, b2, w1

        def loss_wrapper(p):
            return loss_wrapper_with_apply_fn(
                data.model_apply_fn,
                p,
                data.unravel_fn,
                data.loss_fn,
                data.training_data,
                data.training_targets,
            )

        # JIT and return Hessian function
        return jax.hessian(loss_wrapper)(data.params_flat) + damping * jnp.eye(
            data.num_params
        )

    @override
    def compute_hvp(
        self,
        vectors: Float[Array, "n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params"]:
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

        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        # Compute batched IHVP
        result: Float[Array, "batch_size n_params"] = self.compute_hvp_jitted_batched(
            HessianJITData(
                training_data=training_data,
                training_targets=training_targets,
                params_flat=params_flat,
                unravel_fn=unravel_fn,
                model_apply_fn=self.model_data.model.apply,
                loss_fn=self.model_data.loss,
            ),
            vectors_2D,
            damping=0.0 if damping is None else damping,
        )
        return result.squeeze(0) if is_single else result

    @staticmethod
    @jax.jit
    def compute_hvp_jitted_batched(
        data: HessianJITData,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "n_params"]:
        """
        JIT-compiled Hessian-vector product (HVP) computation.
        """
        # Vectorize over the batch dimension
        return jax.vmap(lambda v: Hessian.compute_hvp_jitted_single(data, v, damping))(
            vectors
        )

    @staticmethod
    @jax.jit
    def compute_hvp_jitted_single(
        data: HessianJITData,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """
        JIT-compiled Hessian-vector product (HVP) computation.
        """

        def loss_wrapper(p):
            return loss_wrapper_with_apply_fn(
                data.model_apply_fn,
                p,
                data.unravel_fn,
                data.loss_fn,
                data.training_data,
                data.training_targets,
            )

        # Use jax.jvp for efficient Hessian-vector product
        _, hvp_result = jax.jvp(
            lambda p: jax.grad(loss_wrapper)(p),
            (data.params_flat,),
            (vectors,),
        )
        return hvp_result + damping * vectors

    @override
    def compute_ihvp(
        self,
        vector: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params"]:
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

        return self.compute_ihvp_jitted_batched(
            HessianJITData(
                training_data=training_data,
                training_targets=training_targets,
                params_flat=params_flat,
                unravel_fn=unravel_fn,
                model_apply_fn=self.model_data.model.apply,
                loss_fn=self.model_data.loss,
            ),
            vector,
            damping=0.0 if damping is None else damping,
        )

    @staticmethod
    @jax.jit
    def compute_ihvp_jitted_batched(
        data: HessianJITData,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """
        JIT-compiled Inverse Hessian-vector product (IHVP) computation.
        """
        # Vectorize over the batch dimension
        return jax.vmap(lambda v: Hessian.compute_ihv_jitted(data, v, damping))(vectors)

    @staticmethod
    @jax.jit
    def compute_ihv_jitted(
        data: HessianJITData,
        vector: Float[Array, "n_params"],
        damping: Float,
    ) -> Float[Array, "n_params"]:
        """
        JIT-compiled Inverse Hessian-vector product (IHVP) computation.
        """

        def loss_wrapper(p):
            return loss_wrapper_with_apply_fn(
                data.model_apply_fn,
                p,
                data.unravel_fn,
                data.loss_fn,
                data.training_data,
                data.training_targets,
            )

        # Compute Hessian
        hessian_fn = jax.hessian(loss_wrapper)
        hessian = hessian_fn(data.params_flat) + damping * jnp.eye(
            data.params_flat.shape[0]
        )

        # Solve H x = v for x
        ihvp_result = jnp.linalg.solve(hessian, vector)

        return ihvp_result
