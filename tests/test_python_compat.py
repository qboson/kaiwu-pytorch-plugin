"""Python and PyTorch conversion compatibility regressions."""

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


@pytest.mark.parametrize("model_type", [BoltzmannMachine, RestrictedBoltzmannMachine])
def test_model_to_preserves_pytorch_arguments(model_type):
    """Models accept dtype-only and tensor-reference moves like PyTorch modules."""
    if model_type is RestrictedBoltzmannMachine:
        model = model_type(2, 2, device="cpu")
    else:
        model = model_type(4, device="cpu")
    assert model.to(dtype=torch.float64) is model
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
    assert model.device == torch.device("cpu")
    assert model.to(torch.zeros(1, dtype=torch.float32)) is model
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert model.device == torch.device("cpu")


@pytest.mark.parametrize("via_parent", [False, True])
def test_gbrbm_sampling_after_dtype_conversion(via_parent):
    """Sampling follows direct and parent-initiated dtype changes."""
    model = GaussianBernoulliRestrictedBoltzmannMachine(2, 2, device="cpu")
    target = model
    if via_parent:
        target = torch.nn.Module()
        target.model = model
    for dtype in (torch.float64, torch.float32):
        target.to("cpu", dtype=dtype)
        samples = model.gibbs_sample(n_step=1, n_sample=2)
        assert samples.shape == (2, 4)
        assert samples.dtype == dtype
        assert torch.isfinite(model(samples)).all()
