"""The cond strategy must only remask equal tokens with improved scores."""

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
        del noisy_tokens, attention_mask
        return torch.zeros(candidate_tokens.size(0), 1)


def _model():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=ConstantEnergy(),
        token_spec=spec,
        config=QDiffusionConfig(num_candidates=2),
        freeze_proposal=False,
    )


def _run(condition):
    model = _model()
    return model._reparam_decoding(
        torch.full((1, 4), 5, dtype=torch.long),
        torch.zeros(1, 4),
        torch.tensor([[5, 9, 5, 5]]),
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        f"reparam-{condition}-deterministic-linear",
        torch.zeros(1, 4, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
        t=2,
        max_step=5,
        noise=3,
    )


def test_cond_keeps_tokens_without_improved_scores():
    mask, tokens, scores = _run("cond")

    assert mask.tolist() == [[False, False, False, False]]
    assert tokens.tolist() == [[5, 5, 5, 5]]
    assert scores.tolist() == [[0.0, 0.0, 0.0, 0.0]]


def test_uncond_remasks_the_lowest_two_scores():
    mask, tokens, scores = _run("uncond")

    assert mask.tolist() == [[True, True, False, False]]
    assert tokens.tolist() == [[3, 3, 5, 5]]
    assert scores[:, :2].tolist() == [[float("-inf"), float("-inf")]]


def test_cond_remasks_equal_tokens_with_better_scores():
    model = _model()

    mask, tokens, scores = model._reparam_decoding(
        torch.full((1, 4), 5, dtype=torch.long),
        torch.full((1, 4), 0.5),
        torch.full((1, 4), 5, dtype=torch.long),
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "reparam-cond-deterministic-linear",
        torch.zeros(1, 4, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
        t=2,
        max_step=5,
        noise=3,
    )

    assert mask.tolist() == [[True, True, False, False]]
    assert tokens.tolist() == [[3, 3, 5, 5]]
    assert scores[:, :2].tolist() == [[float("-inf"), float("-inf")]]
