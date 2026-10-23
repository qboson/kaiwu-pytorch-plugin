"""Generation parity and retained token-storage checks for QDiffusion history."""

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


class TokenEnergy(EnergyModel):
    """Small trainable energy scorer without an external solver."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 1)

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del noisy_tokens
        return (self.embedding(candidate_tokens).squeeze(-1) * attention_mask).sum(
            dim=-1, keepdim=True
        )


def make_model(resample=False):
    """Use real tensor sampling, learned proposal logits and candidate scoring."""
    torch.manual_seed(73)
    return QDiffusion(
        nn.Sequential(nn.Embedding(8, 5), nn.Linear(5, 8)),
        TokenEnergy(),
        SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4),
        config=QDiffusionConfig(
            num_candidates=2,
            proposal_temperature=0.7,
            disable_resample=not resample,
            resample_ratio=0.0,
        ),
    ).eval()


def inputs():
    """Include unequal padding lengths and a fixed ordinary token."""
    tokens = torch.tensor([[1, 5, 6, 7, 5, 2, 0], [1, 6, 5, 6, 2, 0, 0]])
    fixed = torch.zeros_like(tokens, dtype=torch.bool)
    fixed[0, 2] = True
    return tokens, fixed


@pytest.mark.parametrize("steps", [0, 1, 6])
@pytest.mark.parametrize("resample", [False, True])
def test_token_only_generation_matches_full_history(steps, resample):
    """History retention must not change samples or consume random draws."""
    model = make_model(resample)
    tokens, fixed = inputs()
    original = tokens.clone()
    torch.manual_seed(891)
    with torch.no_grad():
        full = model.generate(tokens, max_steps=steps, partial_masks=fixed, return_state=True)
    rng_after_full = torch.random.get_rng_state()
    torch.manual_seed(891)
    with torch.no_grad():
        final = model.generate(tokens, max_steps=steps, partial_masks=fixed)
    assert torch.equal(final, full["output_tokens"])
    assert torch.equal(torch.random.get_rng_state(), rng_after_full)
    assert len(full["history"]) == steps + 1
    assert full["history"][-1].shape == final.shape
    assert torch.equal(tokens, original)
    assert final[0, 2] == original[0, 2]
    special = tokens.eq(0) | tokens.eq(1) | tokens.eq(2)
    assert torch.equal(final[special], tokens[special])


@pytest.mark.parametrize("steps", [1, 4, 12])
def test_token_only_generation_does_not_retain_token_snapshots(monkeypatch, steps):
    """Observe actual live history storage during the public generation loop."""
    model = make_model()
    tokens, fixed = inputs()
    retained_bytes = []
    original_step = model.step

    def observe_step(state, **kwargs):
        next_state = original_step(state, **kwargs)
        snapshots = next_state["history"] or []
        retained_bytes.append(sum(t.numel() * t.element_size() for t in snapshots))
        return next_state

    monkeypatch.setattr(model, "step", observe_step)
    with torch.no_grad():
        model.generate(tokens, max_steps=steps, partial_masks=fixed)
    assert retained_bytes == [0] * steps


@pytest.mark.parametrize("record_history", [False, True])
def test_external_loop_preserves_scores_and_previous_states(record_history):
    """External loops can disable snapshots while retaining full decode state."""
    model = make_model()
    tokens, fixed = inputs()
    torch.manual_seed(109)
    with torch.no_grad():
        reference = model.initialize_state(tokens, fixed, max_steps=4)
        for _ in range(4):
            reference = model.step(reference)
    torch.manual_seed(109)
    with torch.no_grad():
        initial = model.initialize_state(tokens, fixed, max_steps=4, record_history=record_history)
        previous = initial
        for step in range(4):
            current = model.step(previous)
            if record_history:
                assert len(previous["history"]) == step + 1
                assert len(current["history"]) == step + 2
                assert current["history"] is not previous["history"]
            else:
                assert previous["history"] is None
                assert current["history"] is None
            previous = current
    assert initial["step"] == 0
    for key in ("output_tokens", "output_scores", "output_masks"):
        assert torch.equal(current[key], reference[key])
    if record_history:
        for actual, expected in zip(current["history"], reference["history"]):
            assert torch.equal(actual, expected)
        assert current["history"][-1].data_ptr() != current["output_tokens"].data_ptr()


def test_resampling_runs_without_recording_history(monkeypatch):
    """Exercise the additional proposal calls used for repeated-token repair."""
    model = make_model(resample=True)
    tokens, fixed = inputs()
    calls = []
    forward = model.proposal_model.forward

    def observe_forward(*args, **kwargs):
        calls.append(1)
        return forward(*args, **kwargs)

    monkeypatch.setattr(model.proposal_model, "forward", observe_forward)
    with torch.no_grad():
        state = model.initialize_state(tokens, fixed, max_steps=4, record_history=False)
        for _ in range(4):
            state = model.step(state)
    assert len(calls) > 4
    assert state["history"] is None
