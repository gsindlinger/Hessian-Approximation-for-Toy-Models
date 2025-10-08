from __future__ import annotations
from typing_extensions import override
from torch.nn import MSELoss, CrossEntropyLoss

import torch

from hessian_approximations.hessian_approximations import HessianApproximation
from models.models import ApproximationModel


class Hessian(HessianApproximation):
    """Hessian Calculation via double backpropagation"""

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
        Compute the exact Hessian matrix of the loss function for the whole model.
        Parameters without gradients getting zeroed out.
        """
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)

        hessian = torch.zeros(
            n_params, n_params, device=training_data.device, dtype=torch.float64
        )

        outputs = model(training_data)
        loss_value = loss(outputs, training_targets)

        first_grads = torch.autograd.grad(
            loss_value, params, create_graph=True, retain_graph=True
        )
        first_grads_flat = torch.cat([g.view(-1) for g in first_grads])

        for i in range(n_params):
            second_grads = torch.autograd.grad(
                first_grads_flat[i],
                params,
                retain_graph=True,
                allow_unused=True,
            )
            second_grads = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(second_grads, params)
            ]
            second_grads_flat = torch.cat([g.view(-1) for g in second_grads])
            hessian[i] = second_grads_flat

        return hessian.to(dtype=torch.float64)
