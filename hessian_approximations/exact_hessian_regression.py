from __future__ import annotations
from typing_extensions import override
from torch.nn import MSELoss, CrossEntropyLoss

import torch
from hessian_approximations.hessian_approximations import HessianApproximation
from models.models import ApproximationModel


class HessianExactRegression(HessianApproximation):
    """Exact Hessian for linear regression with MSE loss.
    Supports multi-dimensional outputs.
    Parameter ordering matches PyTorch: all weights first, then all biases.

    Note, that the Fisher Information Matrix (FIM) is equivalent to the Hessian, expect for a different sign.
    """

    def __init__(self):
        super().__init__()

    @override
    def compute(
        self,
        model: ApproximationModel,
        training_data: torch.Tensor,
        training_targets: torch.Tensor,
        loss: MSELoss | CrossEntropyLoss,
    ):
        """
        Compute the Hessian of a linear regression model w.r.t. its parameters.

        Parameter ordering (matching PyTorch):
        [w_1, w_2, ..., w_d_out, b_1, b_2, ..., b_d_out]
        where w_k is the weight vector for output dimension k.

        The Hessian has the structure:
        H = (2/n) * [[X^T X,    X^T 1],  (repeated d_out times in block diagonal)
                     [1^T X,    1^T 1]]

        Rearranged to match parameter ordering.
        """
        if not isinstance(loss, MSELoss):
            raise ValueError("HessianExactRegression only supports MSE loss.")

        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) != 2:
            raise ValueError(
                "HessianExactRegression only supports linear models with weights and bias."
            )

        n_samples = training_data.shape[0]

        weight = params[0]
        d_out, d_in = weight.shape
        X = training_data

        # Augment X with bias column
        X_augmented = torch.cat([X, torch.ones(n_samples, 1, device=X.device)], dim=1)

        # Compute base Hessian block for one output dimension, in the order [weights, bias] for a single output
        H_block = (2.0 / n_samples) * (X_augmented.T @ X_augmented)

        # Create block-diagonal Hessian with blocks in [w, b] order per output
        blocks = [H_block for _ in range(d_out)]
        H_interleaved = torch.block_diag(*blocks)

        # Rearrange from interleaved order to PyTorch order
        # Current order: [w_1, b_1, w_2, b_2, ..., w_d_out, b_d_out]
        # Target order:  [w_1, w_2, ..., w_d_out, b_1, b_2, ..., b_d_out]
        block_size = d_in + 1  # size of each block (weights + 1 bias)

        # Create permutation indices
        # For weights: indices 0, 1, ..., d_in-1 from each block
        # For biases: index d_in from each block
        perm_indices = []

        # All weight indices
        for k in range(d_out):
            block_start = k * block_size
            weight_indices = list(range(block_start, block_start + d_in))
            perm_indices.extend(weight_indices)

        # Then, collect all bias indices
        for k in range(d_out):
            block_start = k * block_size
            bias_index = block_start + d_in
            perm_indices.append(bias_index)

        # Apply permutation to rows and columns
        perm_tensor = torch.tensor(perm_indices, device=H_interleaved.device)
        full_hessian = H_interleaved[perm_tensor][:, perm_tensor]

        return full_hessian.to(dtype=torch.float64)
