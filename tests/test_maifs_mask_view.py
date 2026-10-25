"""Mask-view contract for the MAIFS feature-selection wrapper."""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from kaiwu.torch_plugin.maifs.plugin import FeatureSelectionWrapper  # noqa: E402


def build_wrapper(axis=1, feature_dim=3):
    """Builds only the attributes ``_mask_view`` reads."""
    wrapper = FeatureSelectionWrapper.__new__(FeatureSelectionWrapper)
    wrapper.input_feature_axis = axis
    wrapper.feature_dim = feature_dim
    return wrapper


def test_mask_is_reshaped_for_broadcasting_with_matching_dtype():
    """A (N, F) input receives an (1, F) mask view with the input dtype."""
    wrapper = build_wrapper()
    inputs = torch.zeros(2, 3, dtype=torch.float64)
    mask = torch.tensor([1.0, 0.0, 1.0])
    view = wrapper._mask_view(inputs, mask)
    assert view.shape == (1, 3)
    assert view.dtype == inputs.dtype
    assert view.tolist() == [[1.0, 0.0, 1.0]]


def test_axis_is_taken_modulo_the_input_rank():
    """Negative or wrapped axes address the same feature dimension."""
    wrapper = build_wrapper(axis=-1)
    inputs = torch.zeros(2, 3)
    view = wrapper._mask_view(inputs, torch.tensor([1.0, 2.0, 3.0]))
    assert view.shape == (1, 3)


def test_mismatched_feature_axis_and_zero_rank_are_rejected():
    """Wrong widths and scalar inputs fail with the documented errors."""
    wrapper = build_wrapper()
    with pytest.raises(ValueError, match="expected 3"):
        wrapper._mask_view(torch.zeros(2, 4), torch.zeros(3))
    with pytest.raises(ValueError, match="at least one dimension"):
        wrapper._mask_view(torch.tensor(1.0), torch.zeros(3))
