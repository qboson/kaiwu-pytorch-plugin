"""Local DPLM checkpoints must initialize all model parameters."""

import pytest
import torch
from torch import nn

pytest.importorskip("omegaconf")
pytest.importorskip("transformers")

from example.qdiffusion.dplm.models import backbone


@pytest.fixture
def checkpoint_path(tmp_path, monkeypatch):
    class TinyNet(nn.Module):
        tokenizer = None
        pad_id, bos_id, eos_id, mask_id, x_id = range(5)

        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(2, 2)

    monkeypatch.setattr(backbone, "get_net", lambda config: TinyNet())
    hydra = tmp_path / ".hydra"
    hydra.mkdir()
    (hydra / "config.yaml").write_text("model:\n  _target_: local.DPLM\n", encoding="utf-8")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    return checkpoints / "local.ckpt"


def test_incomplete_local_checkpoint_reports_missing_weights(checkpoint_path):
    torch.save({"state_dict": {"model.net.projection.weight": torch.eye(2)}}, checkpoint_path)

    with pytest.raises(ValueError, match=r"missing 1 model parameter.*net.projection.bias"):
        backbone.DPLMBackbone.from_pretrained(checkpoint_path, from_huggingface=False)


def test_complete_local_checkpoint_allows_extra_training_state(checkpoint_path):
    bias = torch.tensor([2., 4.])
    torch.save({"state_dict": {
        "model.net.projection.weight": torch.eye(2),
        "model.net.projection.bias": bias,
        "trainer.step": torch.tensor(7),
    }}, checkpoint_path)

    model = backbone.DPLMBackbone.from_pretrained(checkpoint_path, from_huggingface=False)

    torch.testing.assert_close(model.net.projection.weight, torch.eye(2))
    torch.testing.assert_close(model.net.projection.bias, bias)
