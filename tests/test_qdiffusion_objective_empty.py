"""QDiffusion.objective must reject an empty batch instead of emitting NaN."""

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
    def __init__(self, vocab=8, hidden=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.hidden = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, **kwargs):
        del kwargs
        return self.lm_head(torch.tanh(self.hidden(self.embedding(input_ids))))


class _Energy(EnergyModel):
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del attention_mask
        return (
            candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)
            - noisy_tokens.to(torch.float32).sum(dim=1, keepdim=True)
        )


def _model() -> QDiffusion:
    return QDiffusion(
        proposal_model=_Proposal(),
        energy_model=_Energy(),
        token_spec=SequenceTokenSpec(
            pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4, tokenizer=_Tokenizer()
        ),
        config=QDiffusionConfig(
            num_diffusion_timesteps=8, num_candidates=2, disable_resample=True
        ),
        freeze_proposal=False,
    )


def test_empty_targets_are_rejected() -> None:
    empty = torch.zeros((0, 5), dtype=torch.long)
    try:
        _model().objective({"targets": empty})
    except ValueError as exc:
        assert "at least one sample" in str(exc)
    else:  # pragma: no cover - exercised by the RED run
        raise AssertionError("empty targets were accepted")


def test_non_empty_targets_still_produce_finite_statistics() -> None:
    targets = torch.tensor([[1, 5, 6, 2, 0], [1, 6, 5, 2, 0]], dtype=torch.long)
    outputs = _model().objective({"targets": targets})
    assert outputs["logits"].shape == (2, 5, 8)
    assert torch.isfinite(outputs["positive_energy_mean"])
    assert torch.isfinite(outputs["negative_energy_mean"])
