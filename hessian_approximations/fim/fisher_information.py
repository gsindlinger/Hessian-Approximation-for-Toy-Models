from dataclasses import dataclass, field
from typing import Dict

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array
from typing_extensions import override

from config.config import Config
from hessian_approximations.hessian_approximations import HessianApproximation
from models.dataclasses.hessian_compute_context import HessianComputeContext
from models.train import ApproximationModel
from models.utils.loss import get_loss_name


@dataclass
class FisherInformation(HessianApproximation):
    fisher_type: str = field(default="empirical")
    num_samples: int = field(default=1)
    sigma: float = field(default=1.0)
    key: Array = field(default_factory=lambda: random.PRNGKey(0))

    @override
    def compute_hessian(self) -> jnp.ndarray:
        if get_loss_name(self.model_context.loss) == "cross_entropy":
            return self._compute_crossentropy_fim(
                self.model_context.model,
                self.model_context.params,
                self.model_context.dataset.get_train_data()[0],
                self.model_context.dataset.get_train_data()[1],
            )
        else:
            # Default to MSE/regression
            return self._compute_mse_fim(
                self.model_context.model,
                self.model_context.params,
                self.model_context.dataset.get_train_data()[0],
                self.model_context.dataset.get_train_data()[1],
            )

    @override
    def compute_hvp(
        self,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        raise NotImplementedError("Not implemented yet: Fisher Information HVP")

    @override
    def compute_ihvp(
        self,
        config: Config,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        raise NotImplementedError("Not implemented yet: Fisher Information IHVP")

    def _compute_mse_fim(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute FIM for regression (MSE loss).

        For Gaussian likelihood with constant variance σ²:
        FIM = (1/σ²) * (1/n) Σ J_i^T J_i
        where J_i is the Jacobian of the model output w.r.t. parameters

        We assume σ² = 1 for simplicity, so FIM = (1/n) Σ J_i^T J_i
        """
        compute_data = HessianComputeContext.get_data_and_params_for_hessian(
            self.model_context
        )
        n_samples = compute_data.training_data.shape[0]

        # Define the per-sample contribution function once
        @jax.jit
        def compute_sample_contribution(p_flat, x_sample):
            def model_output_fn(p):
                params_unflat = compute_data.unravel_fn(p)
                output = model.apply(params_unflat, jnp.expand_dims(x_sample, axis=0))
                if not isinstance(output, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                return output.flatten()

            jacobian = jax.jacfwd(model_output_fn)(p_flat)
            return jacobian.T @ jacobian

        if self.fisher_type == "empirical":
            # Use actual training data
            fim = jax.vmap(
                lambda x: compute_sample_contribution(compute_data.params_flat, x)
            )(training_data).sum(axis=0)

            fim /= n_samples

        elif self.fisher_type == "true":
            # Sample synthetic outputs from the model's predictive distribution
            keys = random.split(self.key, self.num_samples)
            fim_samples = []

            for s in range(self.num_samples):
                # Generate synthetic targets by sampling from model predictions
                preds = jax.vmap(
                    lambda x: compute_data.model_apply_fn(
                        params, jnp.expand_dims(x, axis=0)
                    )
                )(training_data)
                if not isinstance(preds, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                preds = preds.squeeze(axis=1)

                # Sample from Gaussian: y_synth ~ N(pred, σ²)
                fim_s = jax.vmap(
                    lambda x: compute_sample_contribution(compute_data.params_flat, x)
                )(training_data).sum(axis=0)

                fim_samples.append(fim_s)

            # Average over samples
            fim = jnp.mean(jnp.stack(fim_samples, axis=0), axis=0) / n_samples

        else:
            raise ValueError(f"Unknown fisher_type: {self.fisher_type}")

        return fim

    def _compute_crossentropy_fim(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute FIM for classification (cross-entropy loss).

        The Fisher Information Matrix is the expected outer product of the score:
        FIM = E[∇log p(y|x) ∇log p(y|x)^T]

        For categorical distribution, the score for class c is:
        ∇log p(y=c|x) = ∇f_c - Σ_k p_k ∇f_k
        where f are the logits and p are the softmax probabilities.
        """
        compute_data = HessianComputeContext.get_data_and_params_for_hessian(
            self.model_context
        )
        n_samples = compute_data.training_data.shape[0]

        # Define the per-sample contribution function once
        @jax.jit
        def compute_sample_contribution(p_flat, x_sample, y_sample):
            def logits_fn(p):
                params_unflat = compute_data.unravel_fn(p)
                logits = compute_data.model_apply_fn(
                    params_unflat, jnp.expand_dims(x_sample, axis=0)
                )
                if not isinstance(logits, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                return logits.squeeze(0)

            # Compute Jacobian of logits w.r.t. parameters
            jacobian = jax.jacfwd(logits_fn)(p_flat)  # shape: (n_classes, n_params)

            logits = logits_fn(p_flat)
            probs = jax.nn.softmax(logits, axis=-1)

            # Score: ∇log p(y|x) = J^T (e_y - p)
            # where e_y is one-hot vector for true class y
            # For empirical FIM, y_sample is the actual label
            # For true FIM, y_sample is sampled from the model's distribution
            if y_sample.ndim == 0:  # scalar label
                y_one_hot = jax.nn.one_hot(y_sample, num_classes=len(probs))
            else:  # already one-hot
                y_one_hot = y_sample

            score = jacobian.T @ (y_one_hot - probs)  # shape: (n_params,)

            # FIM is outer product of score
            return jnp.outer(score, score)

        if self.fisher_type == "empirical":
            # Use actual training data and labels
            fim = jax.vmap(
                lambda x, y: compute_sample_contribution(compute_data.params_flat, x, y)
            )(training_data, training_targets).sum(axis=0)

        elif self.fisher_type == "true":
            # Sample synthetic labels from the model's predictive distribution
            keys = random.split(self.key, self.num_samples)
            fim_samples = []

            for s in range(self.num_samples):
                # Get predictions and sample labels
                preds = jax.vmap(
                    lambda x: compute_data.model_apply_fn(
                        params, jnp.expand_dims(x, axis=0)
                    )
                )(training_data)

                if not isinstance(preds, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                preds = preds.squeeze(axis=1)

                probs = jax.nn.softmax(preds, axis=-1)
                # Sample categorical labels from the predictive distribution
                sample_keys = random.split(keys[s], n_samples)
                y_synth = jax.vmap(
                    lambda key, prob: random.categorical(key, jnp.log(prob))
                )(sample_keys, probs)

                fim_s = jax.vmap(
                    lambda x, y: compute_sample_contribution(
                        compute_data.params_flat, x, y
                    )
                )(training_data, y_synth).sum(axis=0)
                fim_samples.append(fim_s)
            # Average over samples
            fim = jnp.mean(jnp.stack(fim_samples, axis=0), axis=0)
        else:
            raise ValueError(f"Unknown fisher_type: {self.fisher_type}")

        fim /= n_samples
        return fim
