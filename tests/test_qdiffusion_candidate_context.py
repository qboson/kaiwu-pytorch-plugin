"""Candidate scoring must respect sequence structure and fixed context."""

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin._qdiffusion_sampling import stochastic_sample_from_categorical_n
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


class ConstrainedProposal(nn.Module):
    """Return usable content logits only at editable positions in the toy input."""

    def __init__(self, invalid_content_logit=None):
        super().__init__()
        logits = torch.full((8, 8), -torch.inf)
        logits[:, 5] = 0.0 if invalid_content_logit is None else invalid_content_logit
        if invalid_content_logit is not None and invalid_content_logit != -torch.inf:
            logits[:, 6] = -2.0
        context_ids = [0, 1, 2, 7]
        logits[context_ids] = -torch.inf
        logits[context_ids, 0] = 0.0
        self.register_buffer("logits", logits)

    def forward(self, tokens):
        return self.logits[tokens]


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


@pytest.mark.parametrize("coupled", (False, True))
@pytest.mark.parametrize("temperature", (0.0, 1.0))
def test_objective_handles_context_with_only_special_logits(coupled, temperature):
    generator, _, energy = make_generator(
        num_candidates=2, proposal_temperature=temperature, use_coupled_sampling=coupled,
    )
    proposal = ConstrainedProposal()
    generator.proposal_model = proposal
    targets = torch.tensor([[1, 6, 2, 0, 0], [1, 6, 6, 2, 0]])
    torch.manual_seed(31)

    outputs = generator.objective({"targets": targets})

    noisy, candidates, attention = energy.calls[1]
    expected_noisy = energy.calls[0][0].repeat_interleave(2, dim=0)
    expected_candidates = torch.where(
        generator.get_non_special_symbol_mask(expected_noisy), 5, expected_noisy,
    )
    torch.testing.assert_close(noisy, expected_noisy)
    torch.testing.assert_close(candidates, expected_candidates)
    torch.testing.assert_close(attention, expected_candidates.ne(0))
    torch.testing.assert_close(outputs["logits"], proposal(energy.calls[0][0]))
    outputs["energy_objective"].mean().backward()
    assert torch.isfinite(energy.weight.grad)


@pytest.mark.parametrize("temperature", (0.0, 1.0))
@pytest.mark.parametrize("cast_overflow", (False, True))
def test_step_preserves_context_tokens_and_scores(temperature, cast_overflow):
    generator, _, energy = make_generator(num_candidates=2, proposal_temperature=temperature)
    proposal = ConstrainedProposal()
    if cast_overflow:
        proposal.logits = proposal.logits.to(torch.float64)
        proposal.logits[[0, 1, 2, 7], 5] = torch.finfo(torch.float64).max
    generator.proposal_model = proposal
    inputs = torch.tensor([[1, 7, 6, 2, 0], [1, 7, 2, 0, 0]])
    partial_masks = inputs.eq(7)
    state = generator.initialize_state(inputs, partial_masks=partial_masks, max_steps=1)
    editable = generator.get_non_special_symbol_mask(inputs, partial_masks)
    state["output_scores"] = torch.arange(inputs.numel(), dtype=torch.float32).reshape_as(inputs)
    initial_scores = state["output_scores"].clone()
    expected_tokens = torch.where(editable, 5, inputs)
    torch.manual_seed(37)

    result = generator.step(state)

    torch.testing.assert_close(result["output_tokens"], expected_tokens)
    torch.testing.assert_close(result["output_scores"][~editable], initial_scores[~editable])
    assert torch.isfinite(result["output_scores"][editable]).all()
    torch.testing.assert_close(state["output_scores"], initial_scores)
    torch.testing.assert_close(energy.calls[0][1], expected_tokens.repeat_interleave(2, dim=0))


@pytest.mark.parametrize("operation", ("objective", "generate"))
@pytest.mark.parametrize("temperature", (0.0, 1.0))
@pytest.mark.parametrize("invalid_logit", (-torch.inf, torch.nan, torch.inf))
def test_invalid_editable_logits_fail_before_energy_scoring(operation, temperature, invalid_logit):
    generator, _, energy = make_generator(num_candidates=2, proposal_temperature=temperature)
    generator.proposal_model = ConstrainedProposal(invalid_content_logit=invalid_logit)
    inputs = torch.tensor([[1, 6, 2, 0]])
    torch.manual_seed(43)

    with pytest.raises(ValueError, match="editable proposal position"):
        if operation == "objective":
            generator.objective({"targets": inputs})
        else:
            generator.generate(inputs, max_steps=1)
    assert not energy.calls


@pytest.mark.parametrize("temperature", (0.0, 1.0))
def test_fixed_logits_do_not_change_editable_random_draws_or_gradients(temperature):
    logits = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(47)).requires_grad_()
    editable = torch.tensor([[False, True, True, False], [False, True, False, False]])
    torch.manual_seed(53)
    original_tokens, original_scores = stochastic_sample_from_categorical_n(logits, temperature, n=3)
    original_rng = torch.get_rng_state()
    original_grad, = torch.autograd.grad(original_scores[:, editable].sum(), logits)

    torch.manual_seed(53)
    tokens, scores = stochastic_sample_from_categorical_n(
        logits, temperature, n=3, editable_token_mask=editable,
    )
    grad, = torch.autograd.grad(scores[:, editable].sum(), logits)
    torch.testing.assert_close(tokens[:, editable], original_tokens[:, editable])
    torch.testing.assert_close(scores[:, editable], original_scores[:, editable], rtol=0, atol=0)
    torch.testing.assert_close(grad, original_grad, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), original_rng)


@pytest.mark.parametrize("temperature", (0.0, 1.0))
def test_fixed_rows_ignore_nonfinite_fp16_perturbations(temperature, monkeypatch):
    logits = torch.tensor([[[2.0, 0.0], [1.0, 1.0]]], dtype=torch.float16, requires_grad=True)
    editable = torch.tensor([[True, False]])

    def draw_uniform(expanded_logits):
        uniforms = torch.full_like(expanded_logits, 0.5)
        uniforms[..., 1, :] = 0.0  # In FP16, the epsilon in the Gumbel draw rounds to zero.
        return uniforms

    monkeypatch.setattr(torch, "rand_like", draw_uniform)
    _, scores = stochastic_sample_from_categorical_n(
        logits, temperature, noise_scale=0.0, n=2, editable_token_mask=editable,
    )
    grad, = torch.autograd.grad(scores[:, editable].sum(), logits)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(grad).all()
    assert grad[:, 1].eq(0).all()


@pytest.mark.parametrize("invalid_logit", (-torch.inf, torch.nan, torch.inf))
def test_repetition_repair_rejects_invalid_logits_at_resampled_positions(invalid_logit):
    generator, _, _ = make_generator(resample_ratio=0.25)
    generator.proposal_model = ConstrainedProposal(invalid_content_logit=invalid_logit)
    tokens = torch.tensor([[1, 6, 6, 2, 0]])
    scores = torch.zeros_like(tokens, dtype=torch.float32)
    editable = generator.get_non_special_symbol_mask(tokens)

    with pytest.raises(ValueError, match="editable proposal position"):
        generator._resample(tokens, scores, editable)
    torch.testing.assert_close(tokens, torch.tensor([[1, 6, 6, 2, 0]]))
    assert scores.eq(0).all()


def test_repetition_repair_ignores_logits_outside_the_replacement_mask():
    generator, _, _ = make_generator(resample_ratio=0.25)
    proposal = ConstrainedProposal()
    proposal.logits[5, 5] = torch.nan
    generator.proposal_model = proposal
    tokens = torch.tensor([[1, 6, 6, 5, 2, 0]])
    scores = torch.arange(tokens.numel(), dtype=torch.float32).reshape_as(tokens)
    editable = generator.get_non_special_symbol_mask(tokens)

    generator._resample(tokens, scores, editable)

    torch.testing.assert_close(tokens, torch.tensor([[1, 5, 5, 5, 2, 0]]))
    torch.testing.assert_close(scores, torch.tensor([[0., 0., 0., 3., 4., 5.]]))


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
        lambda logits, count, **kwargs: (
            candidates, torch.zeros_like(candidates, dtype=logits.dtype)
        ),
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
    monkeypatch.setattr(
        generator, "_sample_candidates", lambda logits, count, **kwargs: (candidates, scores)
    )

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
