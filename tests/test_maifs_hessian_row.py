"""Hessian-row contract for the MAIFS feature-selection wrapper."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from kaiwu.torch_plugin.maifs.plugin import FeatureSelectionWrapper  # noqa: E402


def build_gradient():
    """Builds a gradient whose row derivatives are known."""
    mask = torch.tensor([1.0, 2.0], requires_grad=True)
    first = 3.0 * mask[0] + 5.0 * mask[1]
    second = 7.0 * mask[0] + 11.0 * mask[1]
    return mask, torch.stack([first, second])


def test_full_row_keeps_every_partial_derivative():
    """The full mode returns d(gradient[i])/d(mask) over all entries."""
    mask, gradient = build_gradient()
    row = FeatureSelectionWrapper._hessian_row(gradient, mask, 0, "full")
    assert row.tolist() == [3.0, 5.0]
    second = FeatureSelectionWrapper._hessian_row(gradient, mask, 1, "full")
    assert second.tolist() == [7.0, 11.0]


def test_diagonal_mode_keeps_only_the_own_entry():
    """Diagonal mode zeroes every off-diagonal partial derivative."""
    mask, gradient = build_gradient()
    row = FeatureSelectionWrapper._hessian_row(gradient, mask, 0, "diagonal")
    assert row.tolist() == [3.0, 0.0]
    second = FeatureSelectionWrapper._hessian_row(gradient, mask, 1, "diagonal")
    assert second.tolist() == [0.0, 11.0]


def test_detached_gradient_yields_a_zero_row():
    """A gradient without graph entries contributes nothing."""
    mask = torch.tensor([1.0, 2.0], requires_grad=True)
    gradient = torch.tensor([1.0, 2.0])
    row = FeatureSelectionWrapper._hessian_row(gradient, mask, 0, "full")
    assert row.tolist() == [0.0, 0.0]
