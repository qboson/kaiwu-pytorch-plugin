"""_score_candidates must flatten per candidate and keep the padding mask."""

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

    def __init__(self, vocab_size=8, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        """Returns per-token logits."""
        del kwargs
        return self.lm_head(self.embedding(input_ids))


class RecordingEnergy(EnergyModel):
    """Records the flattened score call and returns the candidate token mean."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Stores the arguments and returns one energy per candidate."""
        self.calls.append(
            (noisy_tokens.clone(), candidate_tokens.clone(), attention_mask.clone())
        )
        return candidate_tokens.to(torch.float32).mean(dim=1, keepdim=True)


def _model(energy):
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=energy,
        token_spec=spec,
        config=QDiffusionConfig(num_candidates=3),
        freeze_proposal=False,
    )


def test_candidates_are_flattened_and_masked_once():
    energy = RecordingEnergy()
    model = _model(energy)
    noisy = torch.tensor([[5, 6, 7, 0], [7, 5, 6, 0]])
    candidates = torch.tensor(
        [
            [[5, 5, 7, 0], [6, 6, 6, 0], [7, 7, 5, 0]],
            [[5, 6, 7, 0], [7, 7, 7, 0], [5, 5, 6, 0]],
        ]
    )

    scores = model._score_candidates(noisy, candidates)

    assert len(energy.calls) == 1
    flat_noisy, flat_candidates, mask = energy.calls[0]
    assert flat_noisy.shape == (6, 4)
    assert flat_candidates.shape == (6, 4)
    expected_noisy = noisy.unsqueeze(1).expand(-1, 3, -1).reshape(6, 4)
    torch.testing.assert_close(flat_noisy, expected_noisy)
    torch.testing.assert_close(flat_candidates, candidates.reshape(6, 4))
    torch.testing.assert_close(mask, flat_candidates != 0)
    assert scores.shape == (2, 3)
    torch.testing.assert_close(scores, candidates.to(torch.float32).mean(dim=-1))
