"""Pin the state payload returned by generate(return_state=True)."""

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)

EXPECTED_KEYS = {
    "output_tokens",
    "output_scores",
    "output_masks",
    "step",
    "max_steps",
    "history",
    "temperature",
    "partial_masks",
}


class _Tokenizer:  # pylint: disable=too-few-public-methods
    def batch_decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(str(int(t)) for t in row) for row in tokens]


class _Proposal(nn.Module):
    def __init__(self, vocab=8, hidden=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, **kwargs):
        del kwargs
        return self.head(self.embedding(input_ids))


class _Energy(EnergyModel):
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        return candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)


def _model() -> QDiffusion:
    return QDiffusion(
        proposal_model=_Proposal(),
        energy_model=_Energy(),
        token_spec=SequenceTokenSpec(
            pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=None, tokenizer=_Tokenizer()
        ),
        config=QDiffusionConfig(num_diffusion_timesteps=4, disable_resample=True),
        freeze_proposal=False,
    )


def test_return_state_payload_has_the_documented_fields() -> None:
    state = _model().generate(
        torch.tensor([[1, 5, 6, 2, 0]]), max_steps=0, return_state=True
    )

    assert isinstance(state, dict)
    assert set(state) == EXPECTED_KEYS
    assert state["max_steps"] == 0
    assert state["step"] == 0
    assert state["temperature"] == 1.0
    assert len(state["history"]) == 1
