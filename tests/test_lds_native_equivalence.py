"""Tiny CPU checks that vectorized LDS is the same seeded native recipe."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from src.config import (ActivationFunction, LossType, LRSchedule, ModelArchitecture,
                        ModelConfig, OptimizerType, TrainingConfig)
from src.utils.lds import _build_elso_cache_key, compute_elso_ground_truth


def _compare_paths(config, x, y):
    # Unequal complement sizes exercise separate cosine schedules and both
    # full and partial final minibatches; these are valid grouped subsets.
    masks = [np.array([True, True, True, False, False, False, False, False]),
             np.array([True, False, False, False, False, False, False, False])]
    kwargs = dict(model_config=config, full_train_inputs=x, full_train_targets=y,
                  query_inputs=x[:2], query_targets=y[:2], subsets=masks,
                  reps_per_subset=1, baseline_losses=np.zeros(2), base_seed=17,
                  epochs=config.training.epochs)
    vectorized = compute_elso_ground_truth(**kwargs, use_vmap=True)
    sequential = compute_elso_ground_truth(**kwargs, use_vmap=False)
    assert vectorized.shape == (2, 2) and np.isfinite(vectorized).all()
    np.testing.assert_allclose(vectorized, sequential, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("optimizer", [OptimizerType.SGD, OptimizerType.ADAMW])
def test_linear_vmap_matches_native_seed_optimizer_and_partial_batch(shuffle, optimizer):
    config = ModelConfig(
        architecture=ModelArchitecture.LINEAR, input_dim=2, output_dim=2,
        loss=LossType.CROSS_ENTROPY,
        training=TrainingConfig(epochs=3, batch_size=3, optimizer=optimizer,
                                learning_rate=0.02, weight_decay=0.01,
                                momentum=0.5, beta1=0.8, beta2=0.95,
                                lr_schedule=LRSchedule.COSINE, shuffle=shuffle),
    )
    x = jnp.arange(16, dtype=jnp.float32).reshape(8, 2) / 16
    y = jnp.arange(8, dtype=jnp.int32) % 2
    _compare_paths(config, x, y)


def test_tiny_transformer_vmap_matches_native_integer_tokens_one_epoch():
    config = ModelConfig(
        architecture=ModelArchitecture.TRANSFORMER, input_dim=4, output_dim=7,
        activation=ActivationFunction.RELU, loss=LossType.CROSS_ENTROPY,
        transformer_dim=4, transformer_num_heads=1, transformer_num_layers=1,
        transformer_vocab_size=7, transformer_max_length=5,
        training=TrainingConfig(epochs=1, batch_size=3, optimizer=OptimizerType.ADAMW,
                                learning_rate=0.001, weight_decay=0.1, shuffle=True,
                                beta1=0.8, beta2=0.95, lr_schedule=LRSchedule.COSINE),
    )
    a = jnp.arange(8, dtype=jnp.int32) % 5
    b = jnp.arange(8, dtype=jnp.int32) // 5
    x = jnp.stack([a, jnp.full_like(a, 6), b, jnp.full_like(a, 5)], axis=1)
    y = (a + b) % 5
    _compare_paths(config, x, y)


def test_lds_cache_versions_recipe_and_shuffle():
    config = ModelConfig(architecture=ModelArchitecture.LINEAR, input_dim=2, output_dim=2)
    kwargs = dict(num_subsets=2, subset_fraction=0.5, reps_per_subset=1, seed=3, n_queries=2, epochs=1)
    old = _build_elso_cache_key(config, **kwargs)
    changed = _build_elso_cache_key(replace(config, training=replace(config.training, shuffle=True)), **kwargs)
    assert old["retraining_implementation"] == "native_seed_shuffle_partialbatch_v2"
    assert old["normalization_policy"] == "fixed_baseline_training_statistics"
    assert old != changed


def test_vmap_refuses_to_cache_nonfinite_query_losses(tmp_path):
    config = ModelConfig(architecture=ModelArchitecture.LINEAR, input_dim=2, output_dim=2,
                         training=TrainingConfig(epochs=1, batch_size=2))
    x = jnp.ones((3, 2), dtype=jnp.float32)
    y = jnp.zeros(3, dtype=jnp.int32)
    with pytest.raises(FloatingPointError, match="Non-finite"):
        compute_elso_ground_truth(
            config, x, y, jnp.full((1, 2), jnp.nan), y[:1],
            [np.array([True, False, False])], 1, np.zeros(1), 17, 1,
            use_vmap=True, cache_directory=str(tmp_path),
        )
    assert not (tmp_path / "rep_mean.npz").exists()
