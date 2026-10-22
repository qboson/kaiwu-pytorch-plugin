"""Pin the initial masked decode state of QDiffusion."""

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


def test_editable_positions_are_replaced_with_the_mask_token() -> None:
    tokens = torch.tensor([[1, 5, 6, 2, 0]])
    output_tokens, output_scores = _model()._initialize_output_tokens(  # pylint: disable=protected-access
        tokens
    )

    assert output_tokens.tolist() == [[1, 3, 3, 2, 0]]
    assert output_scores.shape == tokens.shape
    assert torch.all(output_scores == 0)


def test_partial_masks_keep_their_tokens() -> None:
    tokens = torch.tensor([[1, 5, 6, 2, 0]])
    partial = torch.tensor([[False, True, False, False, False]])
    output_tokens, _ = _model()._initialize_output_tokens(  # pylint: disable=protected-access
        tokens, partial_masks=partial
    )

    assert output_tokens.tolist() == [[1, 5, 3, 2, 0]]
