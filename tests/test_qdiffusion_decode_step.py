"""_decode_step must honour disable_resample and append one history entry."""

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
    """Energy stub returning zeros."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Returns one zero energy per candidate."""
        del noisy_tokens, attention_mask
        return candidate_tokens.to(torch.float32).mean(dim=1, keepdim=True)


def _model(disable_resample):
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=ConstantEnergy(),
        token_spec=spec,
        config=QDiffusionConfig(num_candidates=2, disable_resample=disable_resample),
        freeze_proposal=False,
    )


def _state():
    tokens = torch.tensor([[0, 3, 5, 2]])
    return {
        "output_tokens": tokens,
        "output_scores": torch.zeros_like(tokens, dtype=torch.float32),
        "history": [],
    }


def test_disabled_resampling_is_not_called():
    model = _model(True)
    calls = []
    model._resample = lambda tokens, scores: calls.append(tokens.shape)

    out = model._decode_step(_state())

    assert calls == []
    assert len(out["history"]) == 1
    assert out["output_tokens"][0, 0].item() == 0
    assert out["output_tokens"][0, 3].item() == 2


def test_enabled_resampling_is_called_once():
    model = _model(False)
    calls = []
    model._resample = lambda tokens, scores: calls.append(tokens.shape)

    model._decode_step(_state())

    assert calls == [torch.Size([1, 4])]
