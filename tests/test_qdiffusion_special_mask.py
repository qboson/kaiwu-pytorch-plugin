"""get_non_special_symbol_mask must keep reserved x_id positions fixed."""

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


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
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del attention_mask
        return candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)


def _model() -> QDiffusion:
    return QDiffusion(
        proposal_model=_Proposal(),
        energy_model=_Energy(),
        token_spec=SequenceTokenSpec(
            pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4, tokenizer=_Tokenizer()
        ),
        config=QDiffusionConfig(num_diffusion_timesteps=4, disable_resample=True),
        freeze_proposal=False,
    )


def test_reserved_x_tokens_are_not_editable() -> None:
    tokens = torch.tensor([[1, 4, 5, 2, 0]])
    mask = _model().get_non_special_symbol_mask(tokens)
    assert mask.tolist() == [[False, False, True, False, False]]


def test_partial_masks_still_freezes_positions() -> None:
    tokens = torch.tensor([[1, 4, 5, 6, 2]])
    partial = torch.tensor([[False, False, True, False, False]])
    mask = _model().get_non_special_symbol_mask(tokens, partial_masks=partial)
    assert mask.tolist() == [[False, False, False, True, False]]
