"""Regression tests for the skeptical-remasking schedule boundaries."""

import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class DummyProposal(nn.Module):
    """Tiny proposal model with a moderate preference for one token."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 8)
        self.head = nn.Linear(8, 8)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[5] = 4.0

    def forward(self, tokens):
        return self.head(self.embedding(tokens))


class SumEnergyModel(EnergyModel):
    """Score candidates by their raw token-id sum."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del attention_mask
        return candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)


def build_model():
    spec = SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4)
    config = QDiffusionConfig(
        num_diffusion_timesteps=8,
        num_candidates=1,
        proposal_temperature=0.0,
        disable_resample=True,
    )
    return QDiffusion(
        DummyProposal(),
        SumEnergyModel(),
        spec,
        config=config,
        freeze_proposal=False,
    )


TOKENS = torch.tensor([[1, 5, 6, 2, 0]])


def test_step_beyond_max_steps_is_a_converged_noop():
    """External loops may keep stepping after convergence; it must not crash."""
    model = build_model()
    state = model.initialize_state(TOKENS, max_steps=2)
    state = model.step(state)
    state = model.step(state)
    assert not state["output_masks"].any()
    converged = state["output_tokens"].clone()

    extra = model.step(state)  # t = 3 > max_step = 2

    torch.testing.assert_close(extra["output_tokens"], converged)
    assert not extra["output_masks"].any()
    assert extra["step"] == 3


def test_step_with_zero_max_steps_decodes_in_one_step():
    """max_steps=0 plus a manual step must not divide by zero."""
    model = build_model()
    state = model.initialize_state(TOKENS, max_steps=0)
    assert state["output_masks"].any()

    stepped = model.step(state)  # t = 1, max_step = 0

    assert not stepped["output_masks"].any()
    assert not stepped["output_tokens"].eq(model.mask_id).any()
    # Structural tokens outside the editable region are preserved.
    torch.testing.assert_close(
        stepped["output_tokens"][:, [0, 3, 4]], TOKENS[:, [0, 3, 4]]
    )


def _reparam_inputs(seq_len: int):
    """Build deterministic inputs for direct schedule checks."""
    output_tokens = torch.full((1, seq_len), 3)
    output_scores = torch.zeros(1, seq_len)
    step_tokens = torch.full((1, seq_len), 5)
    # Distinct scores avoid tie handling in top-k selection.
    step_scores = -torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)
    still_noisy = torch.ones(1, seq_len, dtype=torch.bool)
    editable = torch.ones(1, seq_len, dtype=torch.bool)
    return (
        output_tokens,
        output_scores,
        step_tokens,
        step_scores,
        still_noisy,
        editable,
    )


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_reparam_decoding_clamps_schedule_past_max_step(schedule):
    """A step index past max_step decodes everything instead of crashing."""
    model = build_model()
    seq_len = 6
    (
        output_tokens,
        output_scores,
        step_tokens,
        step_scores,
        still_noisy,
        editable,
    ) = _reparam_inputs(seq_len)

    new_mask, tokens, _ = model._reparam_decoding(  # pylint: disable=protected-access
        output_tokens.clone(),
        output_scores.clone(),
        step_tokens,
        step_scores,
        f"reparam-uncond-deterministic-{schedule}",
        still_noisy,
        editable,
        t=10,
        max_step=4,
        noise=3,
    )

    assert not new_mask.any()
    torch.testing.assert_close(tokens, step_tokens)


def test_reparam_decoding_linear_schedule_unchanged_within_range():
    """Within [1, max_step] the linear cutoff stays exactly as scheduled."""
    model = build_model()
    seq_len = 6
    (
        output_tokens,
        output_scores,
        step_tokens,
        step_scores,
        still_noisy,
        editable,
    ) = _reparam_inputs(seq_len)

    new_mask, _, _ = model._reparam_decoding(  # pylint: disable=protected-access
        output_tokens.clone(),
        output_scores.clone(),
        step_tokens,
        step_scores,
        "reparam-uncond-deterministic-linear",
        still_noisy,
        editable,
        t=1,
        max_step=4,
        noise=3,
    )

    # rate = 1 - 1/4 = 0.75; cutoff = floor(6 * 0.75) = 4 kept-masked positions.
    assert int(new_mask.sum()) == 4


def test_reparam_decoding_cosine_schedule_unchanged_within_range():
    """Within [1, max_step] the cosine cutoff stays exactly as scheduled."""
    model = build_model()
    seq_len = 6
    (
        output_tokens,
        output_scores,
        step_tokens,
        step_scores,
        still_noisy,
        editable,
    ) = _reparam_inputs(seq_len)

    new_mask, _, _ = model._reparam_decoding(  # pylint: disable=protected-access
        output_tokens.clone(),
        output_scores.clone(),
        step_tokens,
        step_scores,
        "reparam-uncond-deterministic-cosine",
        still_noisy,
        editable,
        t=1,
        max_step=4,
        noise=3,
    )

    # rate = cos(pi/8); cutoff = floor(6 * cos(pi/8)) = 5 kept-masked positions.
    assert int(new_mask.sum()) == 5


if __name__ == "__main__":
    pytest.main([__file__])
