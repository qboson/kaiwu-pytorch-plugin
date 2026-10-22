"""Generation must keep fixed special tokens and decode every noisy position."""

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class DummyTokenizer:
    """Minimal tokenizer stub."""

    def batch_decode(self, tokens, skip_special_tokens=True):
        """Joins token ids into strings."""
        del skip_special_tokens
        return [" ".join(str(int(token)) for token in row) for row in tokens]


class Proposal(nn.Module):
    """Tiny proposal model."""

    def __init__(self, vocab_size=8, hidden_size=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        """Returns per-token logits."""
        del kwargs
        hidden = torch.tanh(self.hidden(self.embedding(input_ids)))
        return self.lm_head(hidden)


class DummyEnergy(EnergyModel):
    """Tiny energy model."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Returns one score per candidate."""
        del attention_mask
        return (
            candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)
            - noisy_tokens.to(torch.float32).sum(dim=1, keepdim=True)
        )


def _model():
    spec = SequenceTokenSpec(
        pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4, tokenizer=DummyTokenizer()
    )
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=DummyEnergy(),
        token_spec=spec,
        config=QDiffusionConfig(
            num_diffusion_timesteps=8,
            num_candidates=2,
            proposal_temperature=0.0,
            disable_resample=True,
        ),
        freeze_proposal=False,
    )


def test_generation_keeps_specials_and_clears_noise():
    torch.manual_seed(0)
    model = _model()
    tokens = torch.tensor([[1, 5, 6, 2, 0], [1, 6, 5, 2, 0]])

    out = model.generate(tokens, max_steps=2)

    assert out.shape == tokens.shape
    assert out.dtype == tokens.dtype
    special = tokens.eq(0) | tokens.eq(1) | tokens.eq(2)
    torch.testing.assert_close(out[special], tokens[special])
    assert not bool(out.eq(3).any())
