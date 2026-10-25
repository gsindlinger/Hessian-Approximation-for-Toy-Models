"""CPU-only cache/resume checks; expensive retraining is mocked except tiny prefixes."""

import threading
from dataclasses import replace
from math import ceil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.config import LossType, LRSchedule, ModelArchitecture, ModelConfig, OptimizerType, TrainingConfig
from src.utils import lds
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.loss import get_loss
from src.utils.optimizers import optimizer as create_optimizer
from src.utils.train import evaluate_per_example_losses, train_model


def _arguments():
    x = np.arange(16, dtype=np.float32).reshape(8, 2) / 16
    y = np.arange(8, dtype=np.int32) % 2
    config = ModelConfig(
        architecture=ModelArchitecture.LINEAR, input_dim=2, output_dim=2,
        loss=LossType.CROSS_ENTROPY,
        training=TrainingConfig(epochs=4, batch_size=2, optimizer=OptimizerType.SGD,
                                learning_rate=.05, momentum=0, shuffle=True,
                                lr_schedule=LRSchedule.COSINE),
    )
    masks = [np.arange(8) // 2 == index for index in range(3)]
    return dict(model_config=config, full_train_inputs=x, full_train_targets=y,
                query_inputs=x[:2], query_targets=y[:2], subsets=masks,
                reps_per_subset=2, baseline_losses=np.zeros(2), base_seed=17, epochs=2)


@pytest.fixture
def fake_retraining(monkeypatch):
    state = {"calls": 0, "fail_on_call": None}
    monkeypatch.setattr(lds, "_elso_runtime_identity", lambda: {"mock_runtime": 1})

    def build(**unused):
        def run(keys, x, y, qx, qy):
            state["calls"] += 1
            if state["calls"] == state["fail_on_call"]:
                raise RuntimeError("mock interruption")
            return np.asarray(keys)[:, 1, None] + np.asarray(x).sum() + np.arange(len(qx))[None, :]
        return run
    monkeypatch.setattr(lds, "_build_vmapped_elso_fn", build)
    return state


def test_resume_preserves_completed_subsets_and_only_trains_missing(tmp_path, fake_retraining):
    args = _arguments()
    fake_retraining["fail_on_call"] = 2
    with pytest.raises(RuntimeError, match="interruption"):
        lds.compute_elso_ground_truth(**args, cache_directory=str(tmp_path))
    files = list(tmp_path.glob("checkpointed_v1/*/subset_*.npz"))
    assert len(files) == 1
    saved = files[0].read_bytes()
    assert not list(tmp_path.glob("checkpointed_v1/*/rep_mean.npz"))
    fake_retraining.update(calls=0, fail_on_call=None)
    result = lds.compute_elso_ground_truth(**args, cache_directory=str(tmp_path))
    assert fake_retraining["calls"] == 2
    assert files[0].read_bytes() == saved
    assert len(list(tmp_path.glob("checkpointed_v1/*/subset_*.npz"))) == 3
    np.testing.assert_array_equal(result, lds.compute_elso_ground_truth(**args))
    calls = fake_retraining["calls"]
    np.testing.assert_array_equal(result, lds.compute_elso_ground_truth(**args, cache_directory=str(tmp_path)))
    assert fake_retraining["calls"] == calls


def test_shared_cache_reuses_across_model_directories_but_subtracts_own_baseline(tmp_path, fake_retraining):
    args = _arguments()
    root = str(tmp_path / "shared")
    first = lds.compute_elso_ground_truth(**args, cache_directory=str(tmp_path / "model42"), shared_cache_directory=root)
    assert fake_retraining["calls"] == 3
    args["model_config"] = replace(args["model_config"], directory="model43", skip_existing=False)
    args["baseline_losses"] = np.array([.7, 1.1])
    second = lds.compute_elso_ground_truth(**args, cache_directory=str(tmp_path / "model43"), shared_cache_directory=root)
    assert fake_retraining["calls"] == 3
    np.testing.assert_allclose(second, first - np.array([.7, 1.1])[:, None])
    assert len(list((tmp_path / "shared/checkpointed_v1").iterdir())) == 1


@pytest.mark.parametrize("change", ["train", "query", "mask_order", "query_order", "dtype",
                                    "target_epoch", "horizon", "learning_rate", "seed", "reps", "architecture"])
def test_content_identity_invalidates_numerical_changes(change, fake_retraining):
    args = _arguments()
    def identity(values):
        return lds._elso_content_identity(**{k: v for k, v in values.items() if k != "baseline_losses"}, use_vmap=True)
    initial = identity(args)
    if change == "train":
        args["full_train_inputs"] = args["full_train_inputs"] + .01
    elif change == "query":
        args["query_targets"] = 1 - args["query_targets"]
    elif change == "mask_order":
        args["subsets"] = args["subsets"][::-1]
    elif change == "query_order":
        args["query_inputs"] = args["query_inputs"][::-1]
    elif change == "dtype":
        args["full_train_inputs"] = args["full_train_inputs"].astype(np.float64)
    elif change == "target_epoch":
        args["epochs"] = 3
    elif change in ("horizon", "learning_rate"):
        training = replace(args["model_config"].training, **({"epochs": 5} if change == "horizon" else {"learning_rate": .03}))
        args["model_config"] = replace(args["model_config"], training=training)
    elif change == "seed":
        args["base_seed"] += 1
    elif change == "reps":
        args["reps_per_subset"] += 1
    else:
        args["model_config"] = replace(args["model_config"], architecture=ModelArchitecture.MLP, hidden_dim=[2])
    assert identity(args) != initial


def test_invalid_partial_identity_is_not_reused(tmp_path, fake_retraining):
    args = _arguments()
    fake_retraining["fail_on_call"] = 2
    with pytest.raises(RuntimeError):
        lds.compute_elso_ground_truth(**args, cache_directory=str(tmp_path))
    path = next(tmp_path.glob("checkpointed_v1/*/subset_*.npz"))
    np.savez(path, rep_mean=np.zeros(2), identity_sha256="wrong identity", subset_index=0)
    fake_retraining.update(calls=0, fail_on_call=None)
    lds.compute_elso_ground_truth(**args, cache_directory=str(tmp_path))
    assert fake_retraining["calls"] == 3


@pytest.mark.parametrize("target_epoch,legacy_hit", [(4, True), (2, False)])
def test_legacy_final_cache_remains_local_and_intermediate_cosine_is_rejected(tmp_path, fake_retraining, target_epoch, legacy_hit):
    args = _arguments()
    args["epochs"] = target_epoch
    legacy = np.full((2, 3), 123.)
    np.savez(tmp_path / "rep_mean.npz", rep_mean=legacy)
    saved = (tmp_path / "rep_mean.npz").read_bytes()
    result = lds.compute_elso_ground_truth(**args, cache_directory=str(tmp_path), shared_cache_directory=str(tmp_path / "shared"))
    assert (tmp_path / "rep_mean.npz").read_bytes() == saved
    if legacy_hit:
        assert fake_retraining["calls"] == 0
        np.testing.assert_array_equal(result, legacy)
        assert not (tmp_path / "shared").exists()
    else:
        assert fake_retraining["calls"] == 3
        assert not np.array_equal(result, legacy)


def test_flock_serializes_writers_and_releases_after_exception(tmp_path):
    entered = threading.Event()
    errors = []
    identity = {"test": True}
    def follower():
        try:
            with lds._CheckpointedELSO(tmp_path, None, identity, 2, 3):
                entered.set()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=follower, daemon=True)
    with pytest.raises(RuntimeError, match="writer crash"):
        with lds._CheckpointedELSO(tmp_path, None, identity, 2, 3):
            thread.start()
            assert not entered.wait(.05)
            raise RuntimeError("writer crash")
    thread.join(timeout=5)
    assert not thread.is_alive() and entered.is_set() and not errors
    assert list(tmp_path.glob("checkpointed_v1/*/writer.lock"))  # File persistence is harmless.


@pytest.mark.parametrize("use_vmap", [True, False])
def test_intermediate_cosine_is_prefix_of_original_horizon(use_vmap):
    args = _arguments()
    args["subsets"] = args["subsets"][:1]
    args["reps_per_subset"] = 1
    config = args["model_config"]
    mask = args["subsets"][0]
    x, y = args["full_train_inputs"][~mask], args["full_train_targets"][~mask]
    loss = get_loss(config.loss)
    def reference(horizon):
        loader = JAXDataLoader(jnp.asarray(x), jnp.asarray(y), batch_size=2,
                               shuffle=True, rng_key=jax.random.PRNGKey(args["base_seed"]))
        optimizer = create_optimizer(optimizer_enum=OptimizerType.SGD, lr=.05,
                                     momentum=0, lr_schedule=LRSchedule.COSINE,
                                     total_steps=horizon * ceil(len(x) / 2))
        model, params, _ = train_model(config, loader, loss, optimizer,
                                       epochs=args["epochs"], seed=args["base_seed"],
                                       save_checkpoints=False)
        return np.asarray(evaluate_per_example_losses(model, params, args["query_inputs"],
                                                       args["query_targets"], loss))
    expected = reference(config.training.epochs)
    compressed = reference(args["epochs"])
    actual = lds.compute_elso_ground_truth(**args, use_vmap=use_vmap)[:, 0]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert not np.allclose(actual, compressed, rtol=2e-5, atol=2e-6)
    key = lds._build_elso_cache_key(config, 1, .25, 1, 17, 2, args["epochs"])
    assert key["schedule_horizon_epochs"] == 4
    assert key["epochs"] == 2
