"""Small CPU checks for native attribution-campaign data/training support."""

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.config import (ActivationFunction, DatasetConfig, DatasetEnum, LossType,
                        ModelArchitecture, ModelConfig, OptimizerType, TrainingConfig)
from src.utils.data.data import (Dataset, DownloadableDataset, MNISTDataset,
                                 ModularAdditionDataset, ModularQuadraticDataset,
                                 RegressionDataset, normalize_for_loss_with_validation,
                                 resolve_split)
from src.utils.loss import get_loss
from src.utils.models.registry import ModelRegistry
from src.utils.optimizers import optimizer
from src.utils.train import (check_saved_model, evaluate_loss,
                             evaluate_loss_and_classification_accuracy,
                             initialize_model, load_model_checkpoint,
                             save_model_checkpoint, train_model)


@pytest.mark.parametrize("dataset_cls", [ModularAdditionDataset, ModularQuadraticDataset])
def test_modular_complete_task_and_integer_tokens(dataset_cls):
    raw = dataset_cls.download()  # Generated locally, despite the common loader's name.
    dataset = dataset_cls.create_dataset_from_raw(raw)
    inputs, targets = np.asarray(dataset.inputs), np.asarray(dataset.targets)
    assert inputs.shape == (97 * 97, 4)
    assert np.issubdtype(inputs.dtype, np.integer)
    assert np.issubdtype(targets.dtype, np.integer)
    assert dataset.input_dim() == 4 and dataset.output_dim() == 99
    assert np.all(inputs[:, 1] == 98) and np.all(inputs[:, 3] == 97)
    assert len(np.unique(inputs[:, [0, 2]], axis=0)) == 9409
    a, b = inputs[:, 0], inputs[:, 2]
    expected = (a + b) % 97 if dataset_cls is ModularAdditionDataset else (a * a + a * b + b * b) % 97
    np.testing.assert_array_equal(targets, expected)


def test_modular_split_roundtrip_and_seed_provenance(tmp_path):
    cfg = DatasetConfig(name=DatasetEnum.MODULAR_ADDITION, path=str(tmp_path / "addition"),
                        test_size=0.1, split_seed=11, validation_size=0.1,
                        validation_seed=12, subset_seed=13)
    first = resolve_split(cfg)
    second = resolve_split(cfg)
    assert (len(first.train), len(first.validation), len(first.test)) == (7622, 846, 941)
    for one, two in [(first.train, second.train), (first.validation, second.validation),
                     (first.test, second.test)]:
        np.testing.assert_array_equal(one.inputs, two.inputs)
        np.testing.assert_array_equal(one.row_indices, two.row_indices)
    groups = [set(np.asarray(part.row_indices).tolist())
              for part in [first.train, first.validation, first.test]]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert len(set.union(*groups)) == 9409
    manifest = json.loads(first.manifest_path().read_text())
    assert manifest["recipe"]["split_seed"] == 11
    assert manifest["recipe"]["validation_seed"] == 12
    changed = resolve_split(replace(cfg, validation_seed=99))
    assert changed.split_id != first.split_id
    np.testing.assert_array_equal(changed.test.row_indices, first.test.row_indices)


def test_mnist_exact_train_subsample_does_not_touch_canonical_test(tmp_path, monkeypatch):
    full = MNISTDataset(inputs=jnp.arange(70000, dtype=jnp.float32)[:, None],
                        targets=jnp.arange(70000, dtype=jnp.int32) % 10)
    monkeypatch.setattr(DownloadableDataset, "load", staticmethod(lambda *args, **kwargs: full))
    cfg = DatasetConfig(name=DatasetEnum.MNIST, path=str(tmp_path / "mnist"),
                        train_size=6000, validation_size=0.1, subset_seed=8, validation_seed=9)
    result = resolve_split(cfg)
    assert (len(result.train), len(result.validation), len(result.test)) == (6000, 6000, 10000)
    assert int(jnp.max(result.train.row_indices)) < 60000
    assert int(jnp.max(result.validation.row_indices)) < 60000
    np.testing.assert_array_equal(result.test.row_indices, np.arange(60000, 70000))
    assert not set(np.asarray(result.train.row_indices)) & set(np.asarray(result.validation.row_indices))
    another = resolve_split(replace(cfg, subset_seed=10))
    np.testing.assert_array_equal(result.validation.row_indices, another.validation.row_indices)
    np.testing.assert_array_equal(result.test.row_indices, another.test.row_indices)
    assert not np.array_equal(result.train.row_indices, another.train.row_indices)


def test_normalization_fits_final_training_only_and_keeps_row_ids():
    train = RegressionDataset(inputs=jnp.array([[0., 2.], [2., 4.], [4., 6.]]),
                              targets=jnp.array([[2., 4.], [4., 8.], [6., 12.]]),
                              row_indices=jnp.array([1, 2, 3]))
    test = RegressionDataset(inputs=jnp.array([[100., 200.]]), targets=jnp.array([[100., 200.]]))
    validation = RegressionDataset(inputs=jnp.array([[1e6, 2e6]]), targets=jnp.array([[1e6, 2e6]]))
    normalized, _, _ = normalize_for_loss_with_validation(train, test, LossType.MSE, validation)
    np.testing.assert_allclose(np.asarray(normalized.inputs).mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(normalized.targets).std(axis=0), 1, atol=1e-7)
    np.testing.assert_array_equal(normalized.row_indices, train.row_indices)
    assert normalize_for_loss_with_validation(train, test, LossType.CROSS_ENTROPY, validation)[0] is train


def _small_config(tmp_path):
    return ModelConfig(architecture=ModelArchitecture.MLP, input_dim=2, hidden_dim=[3],
                       output_dim=2, activation=ActivationFunction.TANH,
                       loss=LossType.CROSS_ENTROPY, directory=str(tmp_path / "model"),
                       training=TrainingConfig(epochs=2, batch_size=2, optimizer=OptimizerType.SGD))


def test_epoch_zero_loads_before_final_and_preserves_metadata(tmp_path):
    cfg = _small_config(tmp_path)
    model = ModelRegistry.get_model(cfg, seed=7)
    initial = initialize_model(model, 2)
    save_model_checkpoint(cfg, initial, metadata={"model_seed": 7, "dataset": {"name": "test"}}, epoch=0)
    assert not check_saved_model(cfg.directory)
    restored, _, _, metadata = load_model_checkpoint(cfg.directory, epoch=0)
    assert metadata["model_seed"] == 7
    for expected, actual in zip(jax.tree_util.tree_leaves(initial), jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(expected, actual)
    save_model_checkpoint(cfg, initial, epoch=2)
    assert check_saved_model(cfg.directory, epochs=[0, 2])
    assert load_model_checkpoint(cfg.directory)[3]["dataset"] == {"name": "test"}


def test_native_training_saves_local_curves_and_actual_initialization(tmp_path, monkeypatch):
    import src.utils.train as training_module
    monkeypatch.setattr(training_module, "wandb", None)
    cfg = _small_config(tmp_path)
    dataset = Dataset(inputs=jnp.array([[0., 1.], [1., 0.], [1., 1.], [0., 0.]]),
                      targets=jnp.array([0, 1, 1, 0]))
    model, params, history = train_model(
        cfg, dataset.get_dataloader(batch_size=2, shuffle=True, seed=7),
        get_loss(cfg.loss), optimizer(OptimizerType.SGD, lr=0.1, momentum=0.0),
        epochs=2, seed=7, save_epochs=[0, 1, 2], train_dataset=dataset, val_dataset=dataset,
        checkpoint_metadata={"dataset": {"name": "test"}})
    assert len(history) == 2
    initial, _, _, metadata = load_model_checkpoint(cfg.directory, epoch=0)
    expected = initialize_model(ModelRegistry.get_model(cfg, seed=7), 2)
    for one, two in zip(jax.tree_util.tree_leaves(initial), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(one, two)
    assert metadata["training_status"] == "complete"
    records = [json.loads(line) for line in (tmp_path / "model" / "training_metrics.jsonl").read_text().splitlines()]
    assert [record["epoch"] for record in records] == [1, 2]
    assert records[-1]["optimizer_step"] == 4
    assert {"train/loss", "val/loss", "train/accuracy", "val/accuracy"} <= records[-1].keys()
    full = evaluate_loss_and_classification_accuracy(model, params, dataset.inputs, dataset.targets, get_loss(cfg.loss))
    chunked = evaluate_loss_and_classification_accuracy(model, params, dataset.inputs, dataset.targets,
                                                       get_loss(cfg.loss), batch_size=3)
    np.testing.assert_allclose(chunked, full, rtol=1e-6)
    np.testing.assert_allclose(evaluate_loss(model, params, dataset.inputs, dataset.targets,
                                           get_loss(cfg.loss), batch_size=3), full[0], rtol=1e-6)


def test_nonfinite_training_persists_failure_diagnostic(tmp_path):
    cfg = _small_config(tmp_path)
    dataset = Dataset(inputs=jnp.array([[float("nan"), 0.]]), targets=jnp.array([0]))
    with pytest.raises(FloatingPointError, match="Non-finite"):
        train_model(cfg, dataset.get_dataloader(batch_size=1), get_loss(cfg.loss),
                    optimizer(OptimizerType.SGD, lr=0.1), epochs=2, save_epochs=[0])
    record = json.loads((tmp_path / "model" / "training_metrics.jsonl").read_text())
    assert record["status"] == "failed" and record["optimizer_step"] == 1
    assert record["train/running_loss"] is None


def test_shuffle_reproducible_across_epochs():
    dataset = Dataset(inputs=jnp.arange(20)[:, None], targets=jnp.arange(20))
    one = dataset.get_dataloader(batch_size=7, shuffle=True, seed=3)
    two = dataset.get_dataloader(batch_size=7, shuffle=True, seed=3)
    first = None
    for epoch in range(2):
        a = np.concatenate([np.asarray(targets) for _, targets in one])
        b = np.concatenate([np.asarray(targets) for _, targets in two])
        np.testing.assert_array_equal(a, b)
        if epoch == 0:
            first = a
        else:
            assert not np.array_equal(first, a)
