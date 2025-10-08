from __future__ import annotations
from typing_extensions import override
from torch.nn import MSELoss, CrossEntropyLoss
from torch.distributions import Categorical

import torch

from hessian_approximations.hessian_approximations import HessianApproximation
from models.models import ApproximationModel
from torch.nn.utils import parameters_to_vector


class FisherInformation(HessianApproximation):
    def __init__(self, samples_per_input: int = 1):
        super().__init__()
        self.samples_per_input = samples_per_input

    @override
    def compute(
        self,
        model: ApproximationModel,
        training_data: torch.Tensor,
        training_targets: torch.Tensor,
        loss: MSELoss | CrossEntropyLoss,
    ):
        """
        Compute exact Fisher Information Matrix (FIM).
        FIM = E[∇log p(y|x,θ) ∇log p(y|x,θ)^T]

        Importantly, for each input x, we need to sample y ~ p(y|x,θ).
        Therefore we apply a repeated sampling strategy.
        """
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]

        if isinstance(loss, CrossEntropyLoss):
            return self._compute_classification_fim(model, training_data, params)
        elif isinstance(loss, MSELoss):
            return self._compute_regression_fim(model, training_data, params)
        else:
            raise ValueError(f"Unsupported loss type for FIM: {type(loss)}")

    def _compute_classification_fim(
        self,
        model: torch.nn.Module,
        training_data: torch.Tensor,
        params: list[torch.nn.Parameter],
    ) -> torch.Tensor:
        """
        Compute Fisher Information Matrix based on pseudo-gradients for classification.
        For generating pseudo-gradients, we sample y ~ p(y|x,θ) from the model's predicted distribution.
        """
        model.eval()
        device = next(model.parameters()).device
        num_params = sum(p.numel() for p in params)
        fim = torch.zeros(num_params, num_params, device=device)

        for x in training_data:
            x = x.unsqueeze(0).to(device)

            # Forward pass
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs=probs)

            sample_fim = torch.zeros(num_params, num_params, device=device)

            # Monte Carlo sampling for repeated sampling of y ~ p(y|x,θ)
            for _ in range(self.samples_per_input):
                sampled_y = dist.sample()
                log_prob = dist.log_prob(sampled_y)

                # Need create_graph=False but must enable gradients
                grads = torch.autograd.grad(
                    outputs=log_prob,
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
                sample_fim += torch.outer(grad_vec, grad_vec)

            fim += sample_fim / self.samples_per_input

        fim /= len(training_data)
        return fim.to(dtype=torch.float64)

    def _compute_regression_fim(
        self,
        model: torch.nn.Module,
        training_data: torch.Tensor,
        params: list[torch.nn.Parameter],
    ) -> torch.Tensor:
        """
        Compute Fisher Information Matrix for regression.
        Assumes Gaussian likelihood: y ~ N(f(x;θ), σ^2)

        For generating pseudo-gradients, we sample y ~ p(y|x,θ) from the model's predicted distribution.
        """
        model.eval()
        device = next(model.parameters()).device
        num_params = sum(p.numel() for p in params)
        fim = torch.zeros(num_params, num_params, device=device)

        # Assume fixed noise variance
        sigma2 = 1.0

        for x in training_data:
            x = x.unsqueeze(0).to(device)
            mu = model(x).squeeze(
                0
            )  # Trained prediction which serves as the mean for sampling

            sample_fim = torch.zeros(num_params, num_params, device=device)

            for _ in range(self.samples_per_input):
                # Sample y ~ N(mu, sigma^2)
                sampled_y = (mu + torch.randn_like(mu) * sigma2**0.5).detach()

                # log p(y|x,θ) for Gaussian: -0.5 * log(2πσ^2) - 0.5 * (y - mu)^2 / σ^2
                log_prob = (
                    -0.5 * torch.log(torch.tensor(2 * torch.pi * sigma2, device=device))
                    - 0.5 * ((sampled_y - mu) ** 2) / sigma2
                )
                log_prob = log_prob.sum()  # In case of multi-dimensional output

                grads = torch.autograd.grad(
                    outputs=log_prob,
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
                sample_fim += torch.outer(grad_vec, grad_vec)

            fim += sample_fim / self.samples_per_input

        fim /= len(training_data)
        return fim.to(dtype=torch.float64)
