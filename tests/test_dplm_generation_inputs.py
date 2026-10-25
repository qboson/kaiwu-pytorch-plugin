"""Contract test for the DPLM ESM ``prepare_inputs_for_generation`` hooks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
ESM_PATCH = REPO / "example" / "qdiffusion" / "dplm" / "models" / "esm_patch.py"

CLASS_NAMES = ("_ModifiedEsmModel", "_EsmForDPLM")


def _load_esm_patch():
    spec = importlib.util.spec_from_file_location("dplm_esm_patch_inputs_uut", ESM_PATCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def esm_patch():
    return _load_esm_patch()


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_rejects_a_missing_input_ids_argument(esm_patch, class_name):
    model_class = getattr(esm_patch, class_name)

    with pytest.raises(ValueError, match="input_ids"):
        model_class.prepare_inputs_for_generation(object())


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_packs_input_ids_and_optional_past_state(esm_patch, class_name):
    model_class = getattr(esm_patch, class_name)
    input_ids = torch.tensor([[5, 6, 7]])
    attention_mask = torch.tensor([[1, 1, 0]])

    prepared = model_class.prepare_inputs_for_generation(
        object(),
        input_ids,
        None,
        attention_mask=attention_mask,
        extra_option="kept",
    )

    assert prepared["input_ids"] is input_ids
    assert prepared["attention_mask"] is attention_mask
    assert prepared["extra_option"] == "kept"
    # A ``None`` past state must not be forwarded to the model.
    assert "past_key_values" not in prepared

    past = (torch.zeros(2, 1, 3),)
    prepared_with_past = model_class.prepare_inputs_for_generation(
        object(), input_ids, past
    )
    assert prepared_with_past["input_ids"] is input_ids
    assert prepared_with_past["past_key_values"] is past
    assert "attention_mask" not in prepared_with_past
