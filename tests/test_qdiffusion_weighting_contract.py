"""Pin the per-timestep loss weighting of QDiffusion.objective."""

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
        config=QDiffusionConfig(num_diffusion_timesteps=8, disable_resample=True),
        freeze_proposal=False,
    )


def test_linear_weighting_decreases_with_the_timestep() -> None:
    weights = _model()._compute_loss_weight(  # pylint: disable=protected-access
        torch.tensor([1, 4, 8]), "linear"
    )
    expected = torch.tensor([[8.0], [5.0], [1.0]]) / 8.0
    torch.testing.assert_close(weights, expected)


def test_constant_weighting_is_one() -> None:
    weights = _model()._compute_loss_weight(  # pylint: disable=protected-access
        torch.tensor([1, 4, 8]), "constant"
    )
    torch.testing.assert_close(weights, torch.ones(3, 1))
