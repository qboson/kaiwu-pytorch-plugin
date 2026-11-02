"""Regression tests for proposal gradients through ``QDiffusion.objective``.

Issue #292: ``freeze_proposal=False`` is a supported option, but ``objective``
unconditionally ran the proposal under ``torch.no_grad()`` and detached the
returned logits, so callers could never train the proposal through the public
training path.
"""

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class TinyProposal(nn.Module):
    """Small trainable proposal backbone used by both regression cases."""

    def __init__(self, vocab_size: int = 8, hidden_size: int = 12) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return self.head(torch.tanh(self.embedding(tokens)))


class TinyScorer(EnergyModel):
    """Scorer that keeps the energy path independent of the proposal graph."""

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del noisy_tokens, attention_mask
        return candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)


def build_model(*, freeze_proposal: bool) -> tuple[QDiffusion, TinyProposal]:
    """Builds the deterministic toy model shared by the regression tests."""
    proposal = TinyProposal()
    model = QDiffusion(
        proposal_model=proposal,
        energy_model=TinyScorer(),
        token_spec=SequenceTokenSpec(
            mask_id=3,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            x_id=4,
        ),
        config=QDiffusionConfig(
            num_diffusion_timesteps=8,
            num_candidates=1,
            proposal_temperature=0.0,
            disable_resample=True,
        ),
        freeze_proposal=freeze_proposal,
    )
    return model, proposal


def run_objective(model: QDiffusion) -> dict[str, torch.Tensor]:
    """Runs one objective step on a deterministic toy batch."""
    torch.manual_seed(0)
    targets = torch.tensor([[1, 5, 6, 7, 2, 0], [1, 7, 6, 5, 2, 0]])
    return model.objective({"targets": targets})


def test_trainable_proposal_objective_keeps_gradient_graph() -> None:
    """A trainable proposal must produce differentiable objective logits."""
    model, proposal = build_model(freeze_proposal=False)
    outputs = run_objective(model)

    logits = outputs["logits"]
    assert logits.requires_grad
    assert logits.grad_fn is not None

    logits.sum().backward()

    gradients = [
        parameter.grad
        for parameter in proposal.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_frozen_proposal_objective_stays_detached() -> None:
    """The default frozen path must keep its previous detached behaviour."""
    model, proposal = build_model(freeze_proposal=True)
    outputs = run_objective(model)

    assert all(not parameter.requires_grad for parameter in proposal.parameters())
    assert not outputs["logits"].requires_grad
    assert outputs["logits"].grad_fn is None
    assert outputs["energy_objective"].shape == (2, 1)
