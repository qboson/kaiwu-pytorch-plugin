"""Contract test for the DPLM ESM ``_reorder_cache`` beam-search hook."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
ESM_PATCH = REPO / "example" / "qdiffusion" / "dplm" / "models" / "esm_patch.py"

CLASS_NAMES = ("_ModifiedEsmModel", "_EsmForDPLM")


def _load_esm_patch():
    spec = importlib.util.spec_from_file_location("dplm_esm_patch_cache_uut", ESM_PATCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def esm_patch():
    return _load_esm_patch()


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_reorders_every_past_state_along_the_beam_axis(esm_patch, class_name):
    model_class = getattr(esm_patch, class_name)
    keys = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    values = torch.arange(3, dtype=torch.float32).reshape(3, 1)
    past_key_values = ((keys, values), (keys + 1.0, values + 1.0))
    beam_idx = torch.tensor([2, 0])

    reordered = model_class._reorder_cache(object(), past_key_values, beam_idx)

    assert isinstance(reordered, tuple)
    assert len(reordered) == 2
    assert isinstance(reordered[0], tuple)
    assert reordered[0][0].tolist() == [[30.0, 31.0], [10.0, 11.0]]
    assert reordered[0][1].flatten().tolist() == [2.0, 0.0]
    assert reordered[1][0].tolist() == [[31.0, 32.0], [11.0, 12.0]]
    # The cached tensors are reindexed, not modified in place.
    assert keys.tolist() == [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_preserves_the_layer_and_state_structure(esm_patch, class_name):
    model_class = getattr(esm_patch, class_name)
    past_key_values = tuple(
        (
            torch.full((3, 2), float(layer)),
            torch.full((3,), float(layer)),
        )
        for layer in range(4)
    )
    beam_idx = torch.tensor([1, 2, 0])

    reordered = model_class._reorder_cache(object(), past_key_values, beam_idx)

    assert len(reordered) == 4
    for layer, states in enumerate(reordered):
        assert len(states) == 2
        assert states[0].shape == (3, 2)
        assert states[1].shape == (3,)
        assert torch.equal(states[0], torch.full((3, 2), float(layer)))
