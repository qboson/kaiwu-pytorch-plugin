"""Contract tests for the feature-axis masking example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "example"
    / "feature_selection_axis"
    / "axis_masking.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "feature_selection_axis_example", EXAMPLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = _load_example()


def _feature_first_inputs() -> torch.Tensor:
    return torch.arange(
        example.FEATURE_DIM * example.BATCH, dtype=torch.float32
    ).reshape(example.FEATURE_DIM, example.BATCH)


def test_feature_rows_are_zeroed_on_axis_zero() -> None:
    selector = example.build_selector(example.FeatureFirstModel(), axis=0)
    example.install_mask(selector)
    inputs = _feature_first_inputs()
    masked = selector.apply_mask(inputs)
    assert torch.equal(masked[example.DROPPED], torch.zeros(3, example.BATCH))
    kept = [index for index in range(example.FEATURE_DIM) if index not in example.DROPPED]
    assert torch.equal(masked[kept], inputs[kept])


def test_forward_receives_the_masked_tensor() -> None:
    model = example.FeatureFirstModel()
    selector = example.build_selector(model, axis=0)
    example.install_mask(selector)
    inputs = _feature_first_inputs()
    output = selector(inputs)
    assert model.last_input is not None
    assert torch.equal(model.last_input, selector.apply_mask(inputs))
    assert tuple(output.shape) == (1, example.BATCH)


def test_default_axis_zeroes_feature_columns() -> None:
    selector = example.build_selector(example.FeatureFirstModel(), axis=-1)
    example.install_mask(selector)
    inputs = _feature_first_inputs().t()
    masked = selector.apply_mask(inputs)
    dropped_columns = list(example.DROPPED)
    assert torch.equal(masked[:, dropped_columns], torch.zeros(example.BATCH, 3))


def test_mismatched_feature_axis_size_raises_value_error() -> None:
    selector = example.build_selector(example.FeatureFirstModel(), axis=0)
    with pytest.raises(ValueError, match="input feature axis"):
        selector.apply_mask(torch.zeros(example.FEATURE_DIM - 1, example.BATCH))


def test_scalar_input_is_rejected() -> None:
    selector = example.build_selector(example.FeatureFirstModel(), axis=0)
    with pytest.raises(ValueError, match="at least one dimension"):
        selector.apply_mask(torch.tensor(1.0))


def test_non_contiguous_feature_first_input_is_masked() -> None:
    selector = example.build_selector(example.FeatureFirstModel(), axis=0)
    example.install_mask(selector)
    inputs = torch.arange(
        example.BATCH * example.FEATURE_DIM, dtype=torch.float32
    ).reshape(example.BATCH, example.FEATURE_DIM).t()
    assert not inputs.is_contiguous()
    masked = selector.apply_mask(inputs)
    assert torch.equal(masked[example.DROPPED], torch.zeros(3, example.BATCH))
