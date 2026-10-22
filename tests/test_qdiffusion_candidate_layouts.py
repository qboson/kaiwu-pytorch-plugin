"""_reshape_candidates must normalize every supported candidate layout."""

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

    def __init__(self, vocab_size=8, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        """Returns per-token logits."""
        del kwargs
        return self.lm_head(self.embedding(input_ids))


class Energy(EnergyModel):
    """Energy model stub used only for construction."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Returns a constant score."""
        del noisy_tokens, candidate_tokens, attention_mask
        return torch.zeros(1, 1)


def _model():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=Energy(),
        token_spec=spec,
        config=QDiffusionConfig(num_candidates=2),
        freeze_proposal=False,
    )


def test_candidate_first_layout_is_permuted():
    model = _model()
    tensor = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

    reshaped = model._reshape_candidates(tensor, batch_size=3, num_candidates=2)

    assert reshaped.shape == (3, 2, 4)
    torch.testing.assert_close(reshaped, tensor.permute(1, 0, 2))


def test_batch_first_layout_is_kept():
    model = _model()
    tensor = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)

    reshaped = model._reshape_candidates(tensor, batch_size=3, num_candidates=2)

    assert reshaped.shape == (3, 2, 4)
    torch.testing.assert_close(reshaped, tensor)


def test_flat_layout_is_viewed():
    model = _model()
    tensor = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)

    reshaped = model._reshape_candidates(tensor, batch_size=3, num_candidates=2)

    assert reshaped.shape == (3, 2, 4)
    torch.testing.assert_close(reshaped, tensor.view(3, 2, 4))


def test_an_unknown_layout_is_rejected():
    model = _model()

    with pytest.raises(ValueError, match="Unexpected candidate tensor shape"):
        model._reshape_candidates(torch.zeros(5, 3, 4), batch_size=3, num_candidates=2)
