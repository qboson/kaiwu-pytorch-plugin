"""Backward and seeded-reference checks for proposal scores in decode steps."""

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class Proposal(nn.Module):
    """Embedding-based proposal; an optional private copy provides an oracle."""

    def __init__(self, copy_inputs=False):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8)
        self.copy_inputs = copy_inputs
        self.calls = 0
        with torch.no_grad():
            # Repeated legal tokens force the repetition-repair branch.
            self.head.bias[5] = 4.0

    def forward(self, tokens):
        self.calls += 1
        if self.copy_inputs:
            tokens = tokens.clone()
        return self.head(self.embedding(tokens))


class Score(EnergyModel):
    """Deterministic reranking without an external solver or pretrained model."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del noisy_tokens
        return (candidate_tokens.float() * attention_mask).sum(-1, keepdim=True)


def make_model(resample=False, candidates=2, copy_inputs=False, temperature=0.0):
    """Build a trainable proposal with the real core decoding implementation."""
    torch.manual_seed(503)
    return QDiffusion(
        Proposal(copy_inputs), Score(),
        SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4),
        QDiffusionConfig(
            num_candidates=candidates,
            proposal_temperature=temperature,
            proposal_noise_scale=0.1,
            disable_resample=not resample,
            resample_ratio=0.1,
        ),
        freeze_proposal=False,
    ).eval()


def decode(model, steps, seed=809):
    """Run caller-controlled decoding, including padding and fixed context."""
    tokens = torch.tensor([[1, 5, 6, 7, 5, 2, 0], [1, 6, 5, 6, 2, 0, 0]])
    fixed = torch.zeros_like(tokens, dtype=torch.bool)
    fixed[0, 2] = True
    torch.manual_seed(seed)
    state = model.initialize_state(tokens, partial_masks=fixed, max_steps=steps)
    original_state = state
    original_tokens = state["output_tokens"].clone()
    for _ in range(steps):
        state = model.step(state)
    assert original_state["step"] == 0
    assert torch.equal(original_state["output_tokens"], original_tokens)
    assert state["output_tokens"][0, 2] == tokens[0, 2]
    return state, torch.random.get_rng_state()


@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("resample", [False, True])
@pytest.mark.parametrize("candidates", [1, 2])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_decode_score_backward_matches_private_input_reference(steps, resample, candidates, temperature):
    """Scores must backpropagate without changing outputs or the RNG stream."""
    model = make_model(resample, candidates, temperature=temperature)
    reference = copy.deepcopy(model)
    reference.proposal_model.copy_inputs = True
    expected, expected_rng = decode(reference, steps)
    expected["output_scores"].sum().backward()

    actual, actual_rng = decode(model, steps)
    actual["output_scores"].sum().backward()

    assert torch.equal(actual_rng, expected_rng)
    for key in ("output_tokens", "output_scores", "output_masks"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for actual_frame, expected_frame in zip(actual["history"], expected["history"]):
        assert torch.equal(actual_frame, expected_frame)
    for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0)
    assert any(torch.count_nonzero(p.grad) for p in model.parameters())
    if resample:
        assert model.proposal_model.calls > steps


@pytest.mark.parametrize("resample", [False, True])
def test_decode_score_gradient_matches_finite_difference(resample):
    """Independent numeric derivative with locally stable categorical choices."""
    model = make_model(resample)
    state, _ = decode(model, 2)
    state["output_scores"].sum().backward()
    parameter = model.proposal_model.head.bias
    analytic = parameter.grad[5].item()
    original = parameter[5].item()
    epsilon = 0.002
    values = []
    tokens = []
    with torch.no_grad():
        for offset in (epsilon, -epsilon):
            parameter[5] = original + offset
            output, _ = decode(model, 2)
            values.append(output["output_scores"].sum().item())
            tokens.append(output["output_tokens"])
        parameter[5] = original
    assert torch.equal(tokens[0], tokens[1])
    numerical = (values[0] - values[1]) / (2 * epsilon)
    assert abs(analytic) > 1e-4
    assert analytic == pytest.approx(numerical, rel=0.01, abs=0.001)


@pytest.mark.parametrize("resample", [False, True])
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_no_grad_decoding_keeps_existing_behavior(resample, context):
    """The fix does not opt inference callers into recording gradients."""
    model = make_model(resample)
    with context():
        state, _ = decode(model, 2)
    assert not state["output_scores"].requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("resample", [False, True])
def test_optimizer_step_improves_selected_score_loss(resample):
    """A real update can optimize the locally fixed selected-token score."""
    model = make_model(resample)
    optimizer = torch.optim.SGD(model.proposal_model.parameters(), lr=0.001)
    state, _ = decode(model, 2)
    loss = -state["output_scores"].sum()
    initial_loss = loss.item()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        updated, _ = decode(model, 2)
    assert torch.equal(state["output_tokens"], updated["output_tokens"])
    assert -updated["output_scores"].sum().item() < initial_loss
