from __future__ import annotations
from typing_extensions import override
from torch.nn import MSELoss, CrossEntropyLoss

import torch
from hessian_approximations.hessian_approximations import HessianApproximation
from models.models import ApproximationModel
from torch.nn.utils import parameters_to_vector


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
        Compute the Generalized Gauss-Newton approximation of the Hessian.
        """
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        num_params = sum(p.numel() for p in params)
        device = training_data.device

        if isinstance(loss, MSELoss):
            return self._compute_mse_gnh(
                model, training_data, training_targets, params, num_params, device
            )
        elif isinstance(loss, CrossEntropyLoss):
            return self._compute_crossentropy_gnh(
                model, training_data, training_targets, params, num_params, device
            )
        else:
            raise ValueError(f"Unsupported loss type for GGN: {type(loss)}")

    def _compute_mse_gnh(
        self,
        model: torch.nn.Module,
        training_data: torch.Tensor,
        training_targets: torch.Tensor,
        params: list[torch.nn.Parameter],
        num_params: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute GGN for MSE loss with reduction='mean' (PyTorch default).

        For MSE loss L = (1/n) Σ ||f(x_i) - y_i||²:
        The Hessian of L w.r.t. output f(x_i) is:
        ∂²L/∂f² = (2/n) I

        Therefore: H_GGN = (2/n) Σ J_i^T J_i
        where J_i is the Jacobian of f(x_i) w.r.t. parameters
        """
        gnh = torch.zeros(num_params, num_params, device=device, dtype=torch.float64)
        n_samples = training_data.shape[0]

        for i in range(n_samples):
            x = training_data[i].unsqueeze(0)
            output = model(x)

            output_flat = output.view(
                -1
            )  # flatten output for multi-dimensional outputs

            for j in range(
                output_flat.shape[0]
            ):  # compute Jacobian for each output dimension
                model.zero_grad()

                # Compute gradient of output[j] w.r.t. parameters
                grads = torch.autograd.grad(
                    outputs=output_flat[j],
                    inputs=params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )

                grad_vec = parameters_to_vector(
                    [
                        g if g is not None else torch.zeros_like(p)
                        for g, p in zip(grads, params)
                    ]
                )

                gnh += torch.outer(grad_vec, grad_vec)  # Add outer product: J^T J

        gnh *= 2.0 / n_samples  # Scale by 2/n for MSE (assuming reduction='mean')
        return gnh.to(dtype=torch.float64)

    def _compute_crossentropy_gnh(
        self,
        model: torch.nn.Module,
        training_data: torch.Tensor,
        training_targets: torch.Tensor,
        params: list[torch.nn.Parameter],
        num_params: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute GGN for Cross-Entropy loss.

        For cross-entropy with softmax output:
        H_L = diag(p) - p p^T  (Hessian of loss w.r.t. logits)
        where p is the softmax probability vector

        H_GGN = (1/n) Σ J_i^T H_L J_i
        """
        gnh = torch.zeros(num_params, num_params, device=device, dtype=torch.float64)
        n_samples = training_data.shape[0]

        for i in range(n_samples):
            x = training_data[i].unsqueeze(0)
            logits = model(x).squeeze(0)

            probs = torch.softmax(logits, dim=-1)  # Compute softmax probabilities
            num_classes = probs.shape[0]

            # Compute Jacobian matrix: J[k, θ] = ∂logit_k/∂θ
            jacobian_list = []
            for k in range(num_classes):
                model.zero_grad()
                grads = torch.autograd.grad(
                    outputs=logits[k],
                    inputs=params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )

                grad_vec = parameters_to_vector(
                    [
                        g if g is not None else torch.zeros_like(p)
                        for g, p in zip(grads, params)
                    ]
                )
                jacobian_list.append(grad_vec)

            J = torch.stack(
                jacobian_list
            )  # Stack Jacobian matrix: [num_classes, num_params]
            H_L = torch.diag(probs) - torch.outer(
                probs, probs
            )  # Compute H_L = diag(p) - p p^T

            gnh += J.T @ H_L @ J

        gnh /= n_samples
        return gnh.to(dtype=torch.float64)
