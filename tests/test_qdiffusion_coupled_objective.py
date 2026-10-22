"""Pin the doubled batch produced by coupled sampling in objective()."""

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
        return (
            candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)
            - noisy_tokens.to(torch.float32).sum(dim=1, keepdim=True)
        )


def _model(use_coupled_sampling: bool) -> QDiffusion:
    return QDiffusion(
        proposal_model=_Proposal(),
        energy_model=_Energy(),
        token_spec=SequenceTokenSpec(
            pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=None, tokenizer=_Tokenizer()
        ),
        config=QDiffusionConfig(
            num_diffusion_timesteps=8,
            num_candidates=2,
            use_coupled_sampling=use_coupled_sampling,
            disable_resample=True,
        ),
        freeze_proposal=False,
    )


def test_coupled_sampling_doubles_the_batch() -> None:
    torch.manual_seed(0)
    targets = torch.tensor([[1, 5, 6, 2, 0], [1, 6, 5, 2, 0]])
    outputs = _model(True).objective({"targets": targets})

    assert outputs["logits"].shape == (4, 5, 8)
    assert outputs["targets"].shape == (4, 5)
    assert outputs["loss_mask"].shape == (4, 5)
    assert torch.equal(outputs["targets"][:2], targets)
    assert torch.equal(outputs["targets"][2:], targets)


def test_uncoupled_sampling_keeps_the_batch() -> None:
    torch.manual_seed(0)
    targets = torch.tensor([[1, 5, 6, 2, 0], [1, 6, 5, 2, 0]])
    outputs = _model(False).objective({"targets": targets})

    assert outputs["logits"].shape == (2, 5, 8)
    assert torch.equal(outputs["targets"], targets)
