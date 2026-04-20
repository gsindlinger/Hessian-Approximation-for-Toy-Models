from typing import Callable, Dict, Tuple

import jax.numpy as jnp
import pytest
from jax import grad, vmap
from jax.random import PRNGKey

from src.config import PseudoTargetGenerationStrategy
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.eshampoo import EShampooComputer
from src.hessians.utils.data import DataActivationsGradients
from src.utils.data.data import Dataset
from src.utils.models.approximation_model import ApproximationModel
from tests.conftest import TrainingScenario
from tests._helpers import cached_train_model_for_dataset


@pytest.fixture(scope="session")
def shampoo_scenario(
    shared_multiclass_mlp_scenario: TrainingScenario,
) -> TrainingScenario:
    return shared_multiclass_mlp_scenario


@pytest.fixture(scope="session")
def shampoo_model_params_loss(
    trained_model_registry: Dict,
    shampoo_scenario: TrainingScenario,
) -> Tuple[ApproximationModel, Dict, Callable]:
    return cached_train_model_for_dataset(
        shampoo_scenario.model_config,
        shampoo_scenario.dataset,
        trained_model_registry,
        seed=shampoo_scenario.train_seed,
        shuffle=shampoo_scenario.shuffle,
    )


@pytest.fixture(scope="session")
def shampoo_collector_data_ef(
    shampoo_scenario: TrainingScenario,
    shampoo_model_params_loss: Tuple[ApproximationModel, Dict, Callable],
) -> DataActivationsGradients:
    model, params, loss_fn = shampoo_model_params_loss
    base = shampoo_scenario.model_config.directory
    assert base is not None
    collector = CollectorActivationsGradients(
        model=model,
        params=params,
        loss_fn=loss_fn,
        pseudo_target_strategy=PseudoTargetGenerationStrategy.EMPIRICAL_FISHER,
        pseudo_target_repetitions=1,
    )
    return collector.collect(
        dataset=shampoo_scenario.dataset,
        save_directory=f"{base}/shampoo_ef",
        try_load=True,
        rng_key=PRNGKey(42),
    )


def _autograd_shampoo_factors(
    model: ApproximationModel,
    params: Dict,
    loss_fn: Callable,
    dataset: Dataset,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Per-sample param gradients via autograd, reduced into Shampoo factors:

        gradient_cov[o,p] = (1/N) Σ_n (W_grad[n]^T W_grad[n])[o,p]   (O, O)
        activation_cov[i,j] = (1/N) Σ_n (W_grad[n] W_grad[n]^T)[i,j] (I, I)

    where `W_grad[n]` is shape `(I, O)` — the per-sample gradient of the
    sum-reduced loss w.r.t. the Dense kernel.
    """

    def sample_loss(p, x, y):
        pred = model.apply(p, x[None, :])
        return loss_fn(pred, y[None], reduction="sum")

    per_sample = vmap(grad(sample_loss), in_axes=(None, 0, 0))(
        params, jnp.asarray(dataset.inputs), jnp.asarray(dataset.targets)
    )
    N = dataset.inputs.shape[0]
    grad_covs: Dict[str, jnp.ndarray] = {}
    act_covs: Dict[str, jnp.ndarray] = {}
    for layer, leaf in per_sample["params"].items():
        W_grad = leaf["kernel"]  # (N, I, O)
        grad_covs[layer] = jnp.einsum("nio,nip->op", W_grad, W_grad) / N
        act_covs[layer] = jnp.einsum("nio,njo->ij", W_grad, W_grad) / N
    return grad_covs, act_covs


def test_shampoo_covariances_match_autograd(
    shampoo_scenario: TrainingScenario,
    shampoo_model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    shampoo_collector_data_ef: DataActivationsGradients,
):
    """Shampoo's L / R should equal the per-sample-autograd-gradient
    reduction `Σ W_grad^T W_grad / N`, `Σ W_grad W_grad^T / N`.

    EF only — for MCMC / ALL_CLASSES the "gradient" is computed against
    pseudo-targets generated inside the collector, out of autograd's view.
    """
    model, params, loss_fn = shampoo_model_params_loss
    grad_covs_ref, act_covs_ref = _autograd_shampoo_factors(
        model, params, loss_fn, shampoo_scenario.dataset
    )

    covs = EShampooComputer._compute_covariances(
        activations_dict=shampoo_collector_data_ef.activations,
        gradients_dict=shampoo_collector_data_ef.gradients,
        probs=shampoo_collector_data_ef.probs,
    )

    for layer in shampoo_collector_data_ef.layer_names:
        assert jnp.allclose(
            covs["gradient_cov"][layer],
            grad_covs_ref[layer],
            rtol=1e-4,
            atol=1e-4,
        ), f"gradient_cov mismatch on layer '{layer}'"
        assert jnp.allclose(
            covs["activation_cov"][layer],
            act_covs_ref[layer],
            rtol=1e-4,
            atol=1e-4,
        ), f"activation_cov mismatch on layer '{layer}'"
