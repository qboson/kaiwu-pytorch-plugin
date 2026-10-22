"""Unknown remasking strategies must fail loudly."""

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class Proposal(nn.Module):
    """Minimal proposal backbone."""

    def __init__(self, vocab_size=16, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        """Returns per-token logits."""
        del kwargs
        return self.lm_head(self.embedding(input_ids))


class ConstantEnergy(EnergyModel):
    """Energy stub."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Returns zeros."""
        del noisy_tokens, candidate_tokens, attention_mask
        return torch.zeros(1, 1)


def _model():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=ConstantEnergy(),
        token_spec=spec,
        config=QDiffusionConfig(num_candidates=2),
        freeze_proposal=False,
    )


def _call(model, strategy):
    tokens = torch.full((1, 4), 3, dtype=torch.long)
    return model._reparam_decoding(
        tokens,
        torch.zeros(1, 4),
        torch.full((1, 4), 7, dtype=torch.long),
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        strategy,
        torch.ones(1, 4, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
        t=2,
        max_step=5,
        noise=3,
    )


def test_unknown_schedule_is_rejected():
    with pytest.raises(NotImplementedError, match="Unknown schedule"):
        _call(_model(), "reparam-uncond-deterministic-cubic")


def test_unknown_topk_mode_is_rejected():
    with pytest.raises(NotImplementedError, match="Unknown topk mode"):
        _call(_model(), "reparam-uncond-random-linear")


def test_unknown_condition_mode_is_rejected():
    with pytest.raises(NotImplementedError, match="Unknown condition mode"):
        _call(_model(), "reparam-weird-deterministic-linear")
