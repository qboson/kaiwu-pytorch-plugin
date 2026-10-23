"""Feature selection batches follow parameters, not the mask buffer device.

``FeatureSelectionWrapper.mask`` is a parent buffer. ``wrapper.model.to(...)``
moves the wrapped parameters through ``_apply`` and leaves that buffer where
it was constructed. ``fit_weights`` and ``compute_mask_derivatives`` used to
place batches on ``self.mask.device``, so a CUDA or meta model received CPU
batches (or the copy out of meta failed before the forward).

Dedupe: this is the same stale-device class as #1268 and #1275, but those PRs
cover Boltzmann machines and GBRBM. #242 / #331 cover DBN. #488 covers
``QDiffusion`` metadata. No open PR retargets MAIFS batch placement.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
from torch import nn

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

import kaiwu

kaiwu_src_path = os.path.join(src_root, "kaiwu")
if kaiwu_src_path not in kaiwu.__path__:
    kaiwu.__path__ = [kaiwu_src_path] + list(kaiwu.__path__)

for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith(
        "kaiwu.torch_plugin."
    ):
        del sys.modules[module_name]
if hasattr(kaiwu, "torch_plugin"):
    delattr(kaiwu, "torch_plugin")

from kaiwu.torch_plugin import FeatureSelectionWrapper


class RecordingModel(nn.Module):
    """Linear map that records the device of the tensor it actually sees."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(feature_dim, 1))
        self.seen_device: torch.device | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.seen_device = features.device
        return features @ self.weight


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().mean()


def _stale_mask_wrapper(feature_dim: int = 3) -> FeatureSelectionWrapper:
    """Build a wrapper whose parameters were moved without the mask buffer."""
    wrapper = FeatureSelectionWrapper(RecordingModel(feature_dim), feature_dim)
    wrapper.model.to("meta")
    return wrapper


def test_constructor_places_mask_on_the_model_device() -> None:
    """A model that already lives off CPU must not get a CPU mask buffer."""
    model = RecordingModel(3).to("meta")

    wrapper = FeatureSelectionWrapper(model, feature_dim=3)

    assert wrapper.mask.device == next(wrapper.model.parameters()).device


def test_child_move_leaves_mask_buffer_behind() -> None:
    """Moving only the wrapped module must not be mistaken for moving the mask."""
    wrapper = FeatureSelectionWrapper(RecordingModel(3), feature_dim=3)

    wrapper.model.to("meta")

    assert next(wrapper.model.parameters()).device.type == "meta"
    assert wrapper.mask.device.type == "cpu"


def test_derivatives_follow_parameters_when_mask_buffer_is_stale() -> None:
    """Derivative batches stay on the parameter device after a child move.

    Meta tensors have no storage, so exporting the gradient to NumPy still
    fails. The failure must happen after the wrapped model sees its parameter
    device, not while copying the batch onto the stale CPU mask.
    """
    wrapper = _stale_mask_wrapper()
    features = torch.ones(4, 3, device="meta")
    targets = torch.zeros(4, 1, device="meta")

    with pytest.raises((NotImplementedError, RuntimeError), match="meta"):
        wrapper.compute_mask_derivatives([(features, targets)], _mse, max_samples=4)

    assert wrapper.model.seen_device == next(wrapper.model.parameters()).device


def test_fit_weights_follows_parameters_when_mask_buffer_is_stale() -> None:
    """Training batches and targets stay on the parameter device.

    ``float(loss)`` cannot read a meta scalar. The loss is still built from
    predictions and targets on the parameter device before that export.
    """
    wrapper = _stale_mask_wrapper()
    features = torch.ones(4, 3, device="meta")
    targets = torch.zeros(4, 1, device="meta")
    seen: dict[str, torch.device] = {}

    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        seen["prediction"] = prediction.device
        seen["target"] = target.device
        return _mse(prediction, target)

    optimizer = torch.optim.SGD(wrapper.model.parameters(), lr=0.1)
    with pytest.raises((NotImplementedError, RuntimeError), match="meta"):
        wrapper.fit_weights([(features, targets)], loss_fn, optimizer, train_epochs=1)

    parameter_device = next(wrapper.model.parameters()).device
    assert seen["prediction"] == parameter_device
    assert seen["target"] == parameter_device
    assert wrapper.model.seen_device == parameter_device


def test_cpu_derivatives_stay_finite_when_devices_match() -> None:
    """CPU models still return finite gradient and Hessian arrays."""
    torch.manual_seed(0)
    wrapper = FeatureSelectionWrapper(nn.Linear(3, 1), feature_dim=3)
    features = torch.randn(6, 3)
    targets = torch.randn(6, 1)

    gradient, hessian = wrapper.compute_mask_derivatives(
        [(features, targets)],
        _mse,
        max_samples=6,
    )

    assert gradient.shape == (3,)
    assert hessian.shape == (3, 3)
    assert np.isfinite(gradient).all()
    assert np.isfinite(hessian).all()
    assert wrapper.mask.device.type == "cpu"
