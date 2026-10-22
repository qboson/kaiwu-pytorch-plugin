"""objective must prefix energy-model diagnostics by phase."""

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


class StatsEnergy(EnergyModel):
    """Energy stub exposing diagnostics."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Returns one zero energy per candidate."""
        del noisy_tokens, attention_mask
        return candidate_tokens.to(torch.float32).mean(dim=1, keepdim=True)

    def get_last_stats(self):
        """Returns fixed diagnostics."""
        return {"foo": torch.tensor(2.0), "bar": torch.tensor(3.0)}


def _model():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=StatsEnergy(),
        token_spec=spec,
        config=QDiffusionConfig(num_candidates=2, num_diffusion_timesteps=8),
        freeze_proposal=False,
    )


def test_objective_prefixes_both_phases():
    model = _model()
    targets = torch.tensor([[1, 5, 6, 2, 0], [1, 6, 5, 2, 0]])

    outputs = model.objective({"targets": targets})

    for prefix in ("positive", "negative"):
        assert outputs[f"{prefix}_foo"].item() == 2.0
        assert outputs[f"{prefix}_bar"].item() == 3.0
