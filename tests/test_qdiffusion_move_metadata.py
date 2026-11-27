"""QDiffusion metadata follows all PyTorch module-conversion entry points."""

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import EnergyModel, QDiffusion, SequenceTokenSpec


def _model():
    return QDiffusion(
        nn.Embedding(8, 2), EnergyModel(),
        SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2),
        device="cpu", freeze_proposal=False,
    )


@pytest.mark.parametrize("method,dtype", [
    ("double", torch.float64), ("half", torch.float16), ("bfloat16", torch.bfloat16),
])
def test_dtype_convenience_methods_refresh_metadata(method, dtype):
    model = _model()
    assert getattr(model, method)() is model
    assert model.dtype == dtype
    assert model.device == torch.device("cpu")
    assert next(model.parameters()).dtype == model.dtype
    assert model(torch.tensor([[1, 2]])).dtype == model.dtype
    assert model.float() is model
    assert model.dtype == torch.float32


def test_parent_dtype_conversion_refreshes_child_metadata():
    model = _model()
    parent = nn.ModuleList([model])
    assert parent.to(dtype=torch.float64) is parent
    assert model.dtype == next(model.parameters()).dtype == torch.float64
    assert model(torch.tensor([[1, 2]])).dtype == model.dtype


@pytest.mark.parametrize("nested", [False, True])
def test_meta_device_move_refreshes_metadata(nested):
    model = _model()
    owner = nn.ModuleList([model]) if nested else model
    owner.to("meta")
    assert model.device == next(model.parameters()).device == torch.device("meta")
    # A caller using generator.device should allocate tokens beside the model.
    assert torch.empty((1, 2), dtype=torch.long, device=model.device).device.type == "meta"


def test_reference_tensor_to_retains_pytorch_contract():
    model = _model()
    assert model.to(torch.empty(0, dtype=torch.float64)) is model
    assert model.dtype == torch.float64
    assert model.device == torch.device("cpu")


def test_rejected_conversion_preserves_metadata():
    model = _model()
    with pytest.raises(TypeError):
        model.to(dtype=torch.long)
    assert model.dtype == next(model.parameters()).dtype == torch.float32
    assert model.device == next(model.parameters()).device == torch.device("cpu")
