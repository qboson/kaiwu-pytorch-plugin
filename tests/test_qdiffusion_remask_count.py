"""Remasking must retain the scheduled number of tokens when scores tie."""

import math

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin import _qdiffusion_sampling as sampling
from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tied_scores_retain_exact_per_row_cutoff(monkeypatch, stochastic, dtype):
    """Ties cannot reduce a row's requested count, including zero rows."""
    scores = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [-2.0, -2.0, -2.0, 1.0, 1.0],
        [1.0, -3.0, -3.0, -3.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=dtype)
    original = scores.clone()
    cutoff = torch.tensor([[0], [1], [2], [3]])
    monkeypatch.setattr(sampling.torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.5))

    mask = sampling.topk_masking(scores, cutoff, stochastic=stochastic)

    assert mask.dtype == torch.bool
    assert mask.device == scores.device
    torch.testing.assert_close(mask.sum(-1, keepdim=True), cutoff)
    for row_scores, row_mask in zip(scores, mask):
        if row_mask.any() and (~row_mask).any():
            assert row_scores[row_mask].max() <= row_scores[~row_mask].min()
    torch.testing.assert_close(scores, original)


@pytest.mark.parametrize("cutoff", [0, 1, 2, 3, 4])
def test_distinct_scores_keep_lowest_positions(cutoff):
    """Exact counts preserve the existing lowest-score order without ties."""
    scores = torch.tensor([[-1.0, -4.0, -2.0, -3.0]])
    expected_masks = [
        [False, False, False, False],
        [False, True, False, False],
        [False, True, False, True],
        [False, True, True, True],
        [True, True, True, True],
    ]

    mask = sampling.topk_masking(scores, torch.tensor([[cutoff]]))

    torch.testing.assert_close(mask, torch.tensor([expected_masks[cutoff]]))


def test_stochastic_ranking_still_uses_gumbel_perturbation(monkeypatch):
    """The noisy minimum may differ from the unperturbed minimum."""
    scores = torch.tensor([[0.0, 0.2, 0.1, -0.2]])
    uniform = torch.tensor([[0.9, 0.1, 0.5, 0.8]])
    monkeypatch.setattr(sampling.torch, "rand_like", lambda tensor: uniform.expand_as(tensor))

    mask = sampling.topk_masking(scores, torch.tensor([[1]]), stochastic=True)

    torch.testing.assert_close(mask, torch.tensor([[False, True, False, False]]))


def test_empty_positions_allow_zero_cutoff():
    """An empty region has an empty mask rather than an indexing failure."""
    mask = sampling.topk_masking(torch.empty(2, 0), torch.zeros(2, 1, dtype=torch.long))

    assert mask.shape == (2, 0)
    assert mask.dtype == torch.bool


class ConfidentProposal(nn.Module):
    """Finite logits producing exactly tied zero log-probability scores on CPU."""

    def forward(self, tokens):
        logits = torch.full((*tokens.shape, 8), -100.0)
        logits[..., 5] = 100.0
        return logits


class ConstantEnergy(EnergyModel):
    """Choose between identical proposal reconstructions without an external SDK."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del noisy_tokens, attention_mask
        return torch.zeros(candidate_tokens.size(0), 1)


def make_generator(schedule, mode="deterministic"):
    """Keep these regressions focused on the public remasking schedule."""
    return QDiffusion(
        ConfidentProposal(),
        ConstantEnergy(),
        SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4),
        QDiffusionConfig(
            num_candidates=1,
            disable_resample=True,
            decoding_strategy=f"reparam-uncond-{mode}-{schedule}",
        ),
    )


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
@pytest.mark.parametrize("mode", ["deterministic", "stochastic0"])
def test_public_steps_follow_mask_counts_with_tied_confidence(schedule, mode):
    """Equal confidence cannot finish decoding before the configured schedule."""
    generator = make_generator(schedule, mode)
    targets = torch.tensor([[1, 5, 5, 5, 5, 2]])
    state = generator.initialize_state(targets, max_steps=4)
    torch.manual_seed(41)

    for step in range(1, 5):
        state = generator.step(state)
        rate = 1 - step / 4 if schedule == "linear" else math.cos(step / 4 * math.pi / 2)
        expected_count = math.floor(4 * rate)
        assert state["output_masks"].sum().item() == expected_count
        assert state["output_tokens"].eq(3).sum().item() == expected_count
        torch.testing.assert_close(state["output_tokens"][:, [0, 5]], targets[:, [0, 5]])

    torch.testing.assert_close(state["output_tokens"], targets)


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_public_batch_counts_exclude_fixed_and_special_positions(schedule):
    """Each row uses its own editable length while retaining fixed tokens."""
    generator = make_generator(schedule)
    targets = torch.tensor([
        [1, 5, 6, 5, 5, 2, 0, 0],
        [1, 5, 5, 2, 0, 0, 0, 0],
        [1, 6, 2, 0, 0, 0, 0, 0],
    ])
    partial_masks = torch.zeros_like(targets, dtype=torch.bool)
    partial_masks[0, 2] = True
    partial_masks[2, 1] = True
    editable = targets.ne(0) & targets.ne(1) & targets.ne(2) & ~partial_masks
    state = generator.initialize_state(targets, partial_masks=partial_masks, max_steps=4)
    torch.manual_seed(43)

    for step in range(1, 5):
        state = generator.step(state)
        rate = 1 - step / 4 if schedule == "linear" else math.cos(step / 4 * math.pi / 2)
        expected_counts = (editable.sum(-1, keepdim=True).float() * rate).long()
        torch.testing.assert_close(state["output_masks"].sum(-1, keepdim=True), expected_counts)
        assert not (state["output_masks"] & ~editable).any()
        torch.testing.assert_close(state["output_tokens"][~editable], targets[~editable])

    torch.testing.assert_close(state["output_tokens"], targets)


def test_public_cosine_step_can_remask_all_editable_positions():
    """A cosine rate rounded to one remains a valid full-length cutoff."""
    generator = make_generator("cosine")
    targets = torch.tensor([[5, 5, 5, 5]])
    state = generator.initialize_state(targets, max_steps=100000)
    torch.manual_seed(47)

    state = generator.step(state)

    assert state["output_masks"].all()
    assert state["output_tokens"].eq(3).all()
