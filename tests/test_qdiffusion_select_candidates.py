"""Energy-based reranking must select the lowest-energy candidate."""

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


class FirstTokenEnergy(EnergyModel):
    """Returns the first candidate token id as its energy."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Returns the first token of each flattened candidate."""
        del noisy_tokens, attention_mask
        return candidate_tokens[:, :1].to(torch.float32)


def _model():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=FirstTokenEnergy(),
        token_spec=spec,
        config=QDiffusionConfig(num_candidates=3, energy_temperature=1e-3),
        freeze_proposal=False,
    )


def test_the_lowest_energy_candidate_is_selected():
    model = _model()
    noisy = torch.tensor([[5, 6, 7, 8]])
    candidates = torch.tensor(
        [
            [[0, 6, 7, 8], [5, 6, 7, 8], [9, 6, 7, 8]],
        ]
    )
    scores = torch.tensor(
        [
            [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]],
        ]
    )

    tokens, selected_scores = model._select_candidates(noisy, candidates, scores)

    assert tokens.shape == (1, 4)
    torch.testing.assert_close(tokens, candidates[0, 0].unsqueeze(0))
    torch.testing.assert_close(selected_scores, scores[0, 0].unsqueeze(0))
