from dataclasses import asdict, replace

import pytest

from experiments.toy_pairs.prepare import PAIRS, training_recipe
from experiments.train_models import generate_model_directory
from experiments.utils import to_dataclass
from src.config import ActivationFunction, DatasetConfig, DatasetEnum, LRSchedule, TrainingExperimentConfig


@pytest.mark.parametrize("pair", PAIRS)
def test_native_toy_recipe_roundtrip(pair):
    recipe = to_dataclass(TrainingExperimentConfig, training_recipe(pair))
    assert recipe.dataset.validation_size > 0
    assert recipe.selection_metric == "val_loss"
    assert recipe.selection_minimize
    assert 0 in recipe.save_epochs
    assert all(model.training.shuffle for model in recipe.models)
    if pair == "mnist":
        assert recipe.dataset.train_size == 6000
    if pair == "modular_addition":
        assert all(model.output_dim == 99 for model in recipe.models)


def test_model_cache_pins_activation_schedule_and_data_recipe():
    recipe = to_dataclass(TrainingExperimentConfig, training_recipe("concrete"))
    model = recipe.models[0]
    dataset = asdict(recipe.dataset)

    def name(candidate=model, data=dataset, seed=42):
        return generate_model_directory(candidate, "/tmp/toy-model-test", seed, 42, data)

    baseline = name()
    assert name(replace(model, activation=ActivationFunction.TANH)) != baseline
    assert name(replace(model, training=replace(model.training, lr_schedule=LRSchedule.COSINE))) != baseline
    assert name(data={**dataset, "validation_seed": 123}) != baseline
    assert name(data={**dataset, "train_size": 100}) != baseline
    assert name(seed=43) != baseline
    assert name(replace(model, directory="/tmp/another-output", skip_existing=False)) == baseline


def test_model_cache_pins_transformer_dimensions():
    recipe = to_dataclass(TrainingExperimentConfig, training_recipe("modular_addition"))
    model = recipe.models[0]
    original = generate_model_directory(model, "/tmp/toy-model-test", 42, 42)
    changed = generate_model_directory(replace(model, transformer_dim=24), "/tmp/toy-model-test", 42, 42)
    assert original != changed


@pytest.mark.parametrize("fields", [{"validation_size": -0.1}, {"validation_size": 1.0}, {"train_size": 0}])
def test_invalid_split_recipe_rejected(fields):
    with pytest.raises(ValueError):
        DatasetConfig(name=DatasetEnum.MNIST, path="unused", **fields)
