"""Decoding fully fixed inputs must not invoke proposal or energy backends."""

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel, QDiffusion, QDiffusionConfig, SequenceTokenSpec,
)


class CountingProposal(nn.Module):
    """Real embedding logits with a known best editable token."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 8)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[:, 5] = 8
            self.embedding.weight[:, 6] = 1
        self.calls = 0

    def forward(self, input_ids):
        """Count actual proposal evaluations."""
        self.calls += 1
        return self.embedding(input_ids)


class CountingEnergy(EnergyModel):
    """A finite deterministic scorer counting candidate evaluations."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Score the sampled candidates without a licensed backend."""
        del noisy_tokens
        self.calls += 1
        return (candidate_tokens.float() * attention_mask).sum(1, keepdim=True)


def _model(strategy="reparam-uncond-deterministic-linear", disable_resample=False):
    return QDiffusion(
        CountingProposal(), CountingEnergy(),
        SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4),
        config=QDiffusionConfig(num_candidates=3, proposal_noise_scale=0,
                               disable_resample=disable_resample,
                               decoding_strategy=strategy),
    )


def _fixed_input(partial):
    if partial:
        tokens = torch.tensor([[1, 7, 6, 2, 0], [1, 5, 7, 2, 0]])
        return tokens, torch.ones_like(tokens, dtype=torch.bool)
    return torch.tensor([[1, 2, 0, 0, 0], [1, 0, 0, 2, 0]]), None


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("return_state", [False, True])
def test_generate_fixed_inputs_skips_models_and_preserves_history(partial, return_state):
    """Both public return forms retain fixed tokens without consuming proposal RNG."""
    model = _model()
    tokens, masks = _fixed_input(partial)
    original = tokens.clone()
    rng = torch.random.get_rng_state().clone()

    result = model.generate(tokens, partial_masks=masks, max_steps=3,
                            return_state=return_state)

    assert model.proposal_model.calls == model.energy_model.calls == 0
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(tokens, original)
    if return_state:
        assert result["step"] == result["max_steps"] == 3
        assert not result["output_masks"].any()
        assert torch.equal(result["output_scores"], torch.zeros_like(tokens).float())
        history = result["history"]
        assert len(history) == 4
        assert all(torch.equal(item, original) for item in history)
        assert len({item.data_ptr() for item in history}) == 4
        assert result["output_tokens"].data_ptr() not in {item.data_ptr() for item in history}
        result = result["output_tokens"]
    assert torch.equal(result, original)
    assert result.data_ptr() != tokens.data_ptr()


def test_external_step_preserves_scores_and_does_not_mutate_input_state():
    """A stored partial mask works for incremental callers too."""
    model = _model()
    tokens, masks = _fixed_input(True)
    state = model.initialize_state(tokens, partial_masks=masks, max_steps=2)
    state["output_scores"].fill_(0.75)

    updated = model.step(state)

    assert model.proposal_model.calls == model.energy_model.calls == 0
    assert updated["step"] == 1 and state["step"] == 0
    assert len(state["history"]) == 1 and len(updated["history"]) == 2
    assert torch.equal(updated["output_tokens"], tokens)
    assert torch.equal(updated["output_scores"], state["output_scores"])
    assert updated["output_scores"].data_ptr() != state["output_scores"].data_ptr()
    assert updated["history"] is not state["history"]


@pytest.mark.parametrize("strategy, error", [
    ("malformed", ValueError),
    ("reparam-uncond-deterministic-unknown", NotImplementedError),
    ("reparam-uncond-unknown-linear", NotImplementedError),
    ("reparam-unknown-deterministic-linear", NotImplementedError),
])
def test_fixed_input_still_validates_decoding_strategy(strategy, error):
    """Skipping backends must not skip the ordinary state transition validation."""
    tokens, masks = _fixed_input(True)
    with pytest.raises(error):
        _model(strategy).generate(tokens, partial_masks=masks, max_steps=1)


def test_mixed_batch_still_runs_models_and_decodes_editable_positions():
    """One fully fixed row must not short-circuit the rest of its batch."""
    model = _model(disable_resample=True)
    tokens, masks = _fixed_input(True)
    masks[1, 1:3] = False

    state = model.generate(tokens, partial_masks=masks, max_steps=1, return_state=True)

    assert model.proposal_model.calls == model.energy_model.calls == 1
    expected = tokens.clone()
    expected[1, 1:3] = 5
    assert torch.equal(state["output_tokens"], expected)
    expected_scores = torch.zeros_like(tokens).float()
    expected_scores[1, 1:3] = torch.tensor([8., 1., 0.]).log_softmax(0)[0]
    torch.testing.assert_close(state["output_scores"], expected_scores)
    assert not state["output_masks"].any()
    assert state["step"] == 1 and len(state["history"]) == 2
