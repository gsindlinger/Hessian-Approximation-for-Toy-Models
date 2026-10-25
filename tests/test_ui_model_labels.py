"""Display aliases must not change stored model/result identities."""

import pytest

from plots.hessian_data import MODEL_LABELS, model_label, result_label


@pytest.mark.parametrize("model_id,expected", [
    ("mlp_a3e9cd67136b", "MLP Concrete · seed 42"),
    ("mlp_2fe582adefa2", "MLP Concrete · seed 43"),
    ("mlp_4bab8fa15498", "MLP Concrete · seed 44"),
    ("mlp_195c1bd9c95f", "MLP Energy · seed 42"),
    ("mlp_7ea4e853bf54", "MLP Energy · seed 43"),
    ("mlp_089067f475df", "MLP Energy · seed 44"),
    ("mlp_e7f326581e66", "MLP MNIST · seed 42"),
    ("mlp_a436e58ba7ad", "MLP MNIST · seed 43"),
    ("mlp_77dae5213a9a", "MLP MNIST · seed 44"),
    ("transformer_a6c30af49009", "Transformer ModAdd · seed 42"),
    ("transformer_a578f82dbe72", "Transformer ModAdd · seed 43"),
    ("transformer_3480d0d6daae", "Transformer ModAdd · seed 44"),
])
def test_selected_models_have_concise_unique_labels(model_id, expected):
    assert model_label(model_id) == expected
    assert list(MODEL_LABELS.values()).count(expected) == 1


def test_existing_aliases_and_unknown_ids_are_preserved():
    assert model_label("mlp_08580ee2573a") == "Small MLP8"
    assert model_label("resnet_mlp_0a69ab6297da") == "Small ResNet MLP3"
    assert model_label("unknown_model_123") == "unknown_model_123"
    # The superseded Energy candidate must not masquerade as the selected one.
    assert model_label("mlp_81c00da84e1a") == "mlp_81c00da84e1a"


def test_result_description_uses_alias_without_mutating_identity():
    row = {"result_id": 7, "model_id": "mlp_195c1bd9c95f", "dataset_name": "energy",
           "epoch": 100, "damping_value": .01, "damping_strategy": "pseudo_inverse",
           "pseudo_inverse_factor": .01}
    original = row.copy()
    label = result_label(row)
    assert "MLP Energy · seed 42" in label
    assert "#7" in label and "ep100" in label
    assert row == original
