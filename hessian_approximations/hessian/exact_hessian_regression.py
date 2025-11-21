from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from typing_extensions import override

from hessian_approximations.hessian_approximations import HessianApproximation


@dataclass
class HessianExactRegression(HessianApproximation):
    """Exact Hessian for linear regression with MSE loss.
    Supports multi-dimensional outputs.
    Parameter ordering matches PyTorch: all weights first, then all biases.

    Note, that the Fisher Information Matrix (FIM) is equivalent to the Hessian, expect for a different sign.
    """

    @override
    def compute_hessian(
        self,
    ) -> jnp.ndarray:
        """
        Compute the Hessian of a linear regression model w.r.t. its parameters.
        """

        training_data, training_targets = self.model_data.dataset.get_train_data()
        n_samples, n_features = training_data.shape
        d_out = training_targets.shape[1]

        if self.model_data.model.use_bias:
            ones = jnp.ones((n_samples, 1))
            X_augmented = jnp.concatenate([ones, training_data], axis=1)
            block_size = n_features + 1
        else:
            X_augmented = training_data
            block_size = n_features

        # Compute Hessian base block for one output dimension by 2/n * X^T X
        H_block = (2.0 / (n_samples * d_out)) * (X_augmented.T @ X_augmented)

        # Create block-diagonal Hessian
        total_size = d_out * block_size
        H_blocked = jnp.zeros((total_size, total_size))

        for i in range(d_out):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            H_blocked = H_blocked.at[start_idx:end_idx, start_idx:end_idx].set(H_block)

        # Create permutation to convert from blocked to interleaved ordering
        # Blocked: [b_0, w_0_0, w_1_0, ..., b_1, w_0_1, w_1_1, ...] (if use_bias)
        #       or [w_0_0, w_1_0, ..., w_0_1, w_1_1, ...] (if not use_bias)
        # Interleaved: [b_0, b_1, ..., w_0_0, w_0_1, ..., w_1_0, w_1_1, ...]
        #           or [w_0_0, w_0_1, ..., w_1_0, w_1_1, ...]
        perm_indices = []

        if self.model_data.model.use_bias:
            # First, add all bias indices (first element of each block)
            for out_dim in range(d_out):
                perm_indices.append(out_dim * block_size)

        # Then, add all weight indices for each feature
        for feat_idx in range(n_features):
            for out_dim in range(d_out):
                offset = 1 if self.model_data.model.use_bias else 0
                perm_indices.append(out_dim * block_size + offset + feat_idx)

        perm_indices = jnp.array(perm_indices)

        # Permute rows and columns
        H_interleaved = H_blocked[perm_indices, :][:, perm_indices]

        return H_interleaved

    @override
    def compute_hvp(
        self,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        hessian = self.compute_hessian()
        hvp = hessian @ vector
        return hvp

    @override
    def compute_ihvp(
        self,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        hessian = self.compute_hessian()
        ihvp = jnp.linalg.solve(hessian, vector)
        return ihvp
