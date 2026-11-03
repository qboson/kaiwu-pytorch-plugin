"""Candidate scoring must respect sequence structure and fixed context."""

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class ConstantProposal(nn.Module):
    """Prefer one ordinary token while recording proposal conditioning inputs."""

    def __init__(self, preferred_token=5):
        super().__init__()
        logits = torch.full((8,), -100.0)
        logits[5] = 100.0
        logits[preferred_token] = 200.0
        self.register_buffer("logits", logits)
        self.inputs = []

    def forward(self, tokens):
        self.inputs.append(tokens.detach().clone())
        return self.logits.expand(*tokens.shape, -1)


class RecordingEnergy(EnergyModel):
    """A differentiable energy oracle whose padding is controlled by attention."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.02, dtype=torch.float64))
        self.calls = []

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        self.calls.append((
            noisy_tokens.detach().clone(),
            candidate_tokens.detach().clone(),
            attention_mask.detach().clone(),
        ))
        return (candidate_tokens * attention_mask).to(self.weight.dtype).sum(
            -1, keepdim=True
        ) * self.weight


def make_generator(**config_kwargs):
    """Build a CPU generator without a tokenizer, checkpoint, or solver."""
    proposal = ConstantProposal()
    energy = RecordingEnergy()
    generator = QDiffusion(
        proposal,
        energy,
        SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4),
        QDiffusionConfig(disable_resample=True, **config_kwargs),
    )
    return generator, proposal, energy


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("num_candidates", [1, 3])
def test_objective_scores_structurally_valid_negatives_and_gradients(coupled, num_candidates):
    """Negative energies use the same BOS/EOS/PAD layout as the targets."""
    generator, proposal, energy = make_generator(
        use_coupled_sampling=coupled, num_candidates=num_candidates
    )
    targets = torch.tensor([[1, 6, 2, 0, 0], [1, 7, 6, 2, 0]])
    torch.manual_seed(7)

    outputs = generator.objective({"targets": targets})
    outputs["energy_objective"].mean().backward()

    expanded_targets = targets.repeat(2 if coupled else 1, 1)
    expected_negatives = torch.tensor([[1, 5, 2, 0, 0], [1, 5, 5, 2, 0]]).repeat(
        2 if coupled else 1, 1
    )
    scored_negatives = energy.calls[1][1]
    expected_flat = expected_negatives.repeat_interleave(num_candidates, dim=0)
    torch.testing.assert_close(scored_negatives, expected_flat)
    torch.testing.assert_close(energy.calls[1][2], expected_flat.ne(0))
    torch.testing.assert_close(
        outputs["logits"], proposal.logits.expand(*expanded_targets.shape, -1)
    )
    positive_sums = expanded_targets.sum(-1).to(torch.float64)
    negative_sums = expected_negatives.sum(-1).to(torch.float64)
    expected_gradient = (
        positive_sums * torch.sigmoid(0.02 * positive_sums)
        - negative_sums * torch.sigmoid(-0.02 * negative_sums)
    ).mean()
    torch.testing.assert_close(energy.weight.grad, expected_gradient)


@pytest.mark.parametrize("coupled", [False, True])
def test_objective_energy_and_gradient_are_invariant_to_extra_padding(coupled):
    """Batch padding cannot supply an energy discriminator's negative signal."""
    results = []
    for extra_padding in (0, 4):
        generator, _, energy = make_generator(use_coupled_sampling=coupled, num_candidates=2)
        targets = torch.tensor([[1, 6, 2, 0] + [0] * extra_padding])
        torch.manual_seed(13)
        outputs = generator.objective({"targets": targets})
        outputs["energy_objective"].mean().backward()
        results.append((outputs["energy_objective"], energy.weight.grad))

    torch.testing.assert_close(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1])


@pytest.mark.parametrize("preferred_special", [0, 1, 2, 3, 4])
def test_objective_excludes_special_proposals_only_from_editable_positions(preferred_special):
    """Raw logits stay available while editable negative tokens stay ordinary."""
    generator, _, energy = make_generator(num_candidates=2)
    generator.proposal_model = ConstantProposal(preferred_token=preferred_special)
    targets = torch.tensor([[1, 6, 2, 0]])
    torch.manual_seed(19)

    outputs = generator.objective({"targets": targets})

    torch.testing.assert_close(
        energy.calls[1][1], torch.tensor([[1, 5, 2, 0], [1, 5, 2, 0]])
    )
    assert outputs["logits"][..., preferred_special].eq(200.0).all()


@pytest.mark.parametrize("partial_fixed", [False, True])
def test_generate_reranks_reconstructions_that_preserve_fixed_context(monkeypatch, partial_fixed):
    """Candidate preference follows the sequence that generation actually emits."""
    generator, _, energy = make_generator(num_candidates=2, energy_temperature=1e-6)
    targets = torch.tensor([[1, 7, 5, 2, 0]]) if partial_fixed else torch.tensor([[1, 5, 2, 0, 0]])
    partial_masks = torch.tensor([[False, True, False, False, False]]) if partial_fixed else None
    candidates = torch.tensor([[[6, 6, 5, 6, 6], [5, 5, 6, 5, 5]]]) if partial_fixed else torch.tensor(
        [[[6, 5, 6, 6, 6], [5, 6, 5, 5, 5]]]
    )
    original_candidates = candidates.clone()
    monkeypatch.setattr(
        generator, "_sample_candidates",
        lambda logits, count: (candidates, torch.zeros_like(candidates, dtype=logits.dtype)),
    )

    generated = generator.generate(targets, partial_masks=partial_masks, max_steps=1)

    torch.testing.assert_close(generated, targets)
    expected_scored = torch.cat((targets, targets.clone()), dim=0)
    expected_scored[1, 2 if partial_fixed else 1] = 6
    torch.testing.assert_close(energy.calls[0][1], expected_scored)
    torch.testing.assert_close(energy.calls[0][2], expected_scored.ne(0))
    torch.testing.assert_close(candidates, original_candidates)


def test_generate_scoring_handles_uneven_padded_batches():
    """Each flattened candidate retains the originating sample's padding mask."""
    generator, _, energy = make_generator(num_candidates=3)
    targets = torch.tensor([[1, 5, 2, 0, 0], [1, 5, 5, 2, 0]])
    torch.manual_seed(23)

    generated = generator.generate(targets, max_steps=1)

    torch.testing.assert_close(generated, targets)
    expected_candidates = targets.repeat_interleave(3, dim=0)
    torch.testing.assert_close(energy.calls[0][1], expected_candidates)
    torch.testing.assert_close(energy.calls[0][2], expected_candidates.ne(0))
    implicit_energy = generator.energy(energy.calls[0][0], expected_candidates)
    explicit_energy = generator.energy(
        energy.calls[0][0], expected_candidates, expected_candidates.ne(0)
    )
    torch.testing.assert_close(implicit_energy, explicit_energy)


def test_repetition_resampling_keeps_fixed_proposal_context(monkeypatch):
    """Resampling must retain structural tokens and fixed repeated ordinary tokens."""
    generator, proposal, energy = make_generator(num_candidates=1, resample_ratio=0.25)
    generator.config.disable_resample = False
    targets = torch.tensor([[1, 6, 6, 5, 5, 2, 0, 0, 0, 0]])
    partial_masks = torch.tensor([[False, True, True, False, False, False, False, False, False, False]])
    candidates = torch.full((1, 1, targets.size(1)), 6)
    scores = torch.zeros_like(candidates, dtype=torch.float32)
    monkeypatch.setattr(generator, "_sample_candidates", lambda logits, count: (candidates, scores))

    generated = generator.generate(targets, max_steps=1, partial_masks=partial_masks)

    torch.testing.assert_close(generated, targets)
    assert len(proposal.inputs) == 2
    expected_resample_input = targets.clone()
    expected_resample_input[:, 3:5] = 3
    torch.testing.assert_close(proposal.inputs[1], expected_resample_input)
    expected_scored = targets.clone()
    expected_scored[:, 3:5] = 6
    torch.testing.assert_close(energy.calls[0][1], expected_scored)


def test_padding_only_repetition_does_not_trigger_resampling():
    """A repeated fixed PAD suffix is not editable repetition collapse."""
    generator, proposal, _ = make_generator(num_candidates=1, resample_ratio=0.25)
    generator.config.disable_resample = False
    targets = torch.tensor([[1, 5, 2, 0, 0, 0, 0, 0]])
    torch.manual_seed(29)

    generated = generator.generate(targets, max_steps=1)

    torch.testing.assert_close(generated, targets)
    assert len(proposal.inputs) == 1


@pytest.mark.parametrize("ordinary_fixed", [False, True])
def test_generate_all_fixed_sequences_score_unchanged_context(ordinary_fixed):
    """An empty editable region has no sampled content or repetition to replace."""
    generator, proposal, energy = make_generator(num_candidates=2)
    generator.config.disable_resample = False
    targets = torch.tensor([[1, 5, 5, 2, 0]]) if ordinary_fixed else torch.tensor([[1, 2, 0, 0, 0]])
    partial_masks = torch.ones_like(targets, dtype=torch.bool) if ordinary_fixed else None

    generated = generator.generate(targets, max_steps=1, partial_masks=partial_masks)

    torch.testing.assert_close(generated, targets)
    torch.testing.assert_close(energy.calls[0][1], targets.repeat_interleave(2, dim=0))
    assert len(proposal.inputs) == 1
