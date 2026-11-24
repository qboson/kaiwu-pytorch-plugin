"""Python and PyTorch conversion compatibility regressions."""

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine


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
