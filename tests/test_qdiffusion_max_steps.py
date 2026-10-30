"""QDiffusion decode steps must be a non-negative integer."""

from __future__ import annotations

import pytest
import torch

from kaiwu.torch_plugin import EnergyModel, QDiffusion, QDiffusionConfig
from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec


class _Proposal(torch.nn.Module):
    def forward(self, tokens):
        return torch.zeros(tokens.size(0), tokens.size(1), 8)


class _Energy(EnergyModel):
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del candidate_tokens, attention_mask
        return torch.zeros(noisy_tokens.size(0), 1)


def _generator() -> QDiffusion:
    spec = SequenceTokenSpec(
        mask_id=4, pad_id=3, bos_id=0, eos_id=1, x_id=2, tokenizer=None
    )
    return QDiffusion(
        _Proposal(), _Energy(), spec, QDiffusionConfig(num_diffusion_timesteps=4)
    )


TOKENS = torch.tensor([[0, 2, 5, 1]])


def test_negative_max_steps_is_rejected_by_generate() -> None:
    with pytest.raises(ValueError, match="max_steps must be a non-negative integer"):
        _generator().generate(TOKENS, max_steps=-1)


def test_negative_max_steps_is_rejected_by_initialize_state() -> None:
    with pytest.raises(ValueError, match="max_steps must be a non-negative integer"):
        _generator().initialize_state(TOKENS, max_steps=-1)


@pytest.mark.parametrize("value", [1.5, True])
def test_non_integer_max_steps_is_rejected(value) -> None:
    with pytest.raises(TypeError, match="max_steps must be a non-negative integer"):
        _generator().initialize_state(TOKENS, max_steps=value)


def test_zero_steps_returns_the_initial_state() -> None:
    state = _generator().initialize_state(TOKENS, max_steps=0)
    assert state["step"] == 0
    assert int(state["output_tokens"][0, 2]) == 4  # mask token
    output = _generator().generate(TOKENS, max_steps=0)
    assert output.shape == TOKENS.shape


def test_positive_steps_still_run() -> None:
    output = _generator().generate(TOKENS, max_steps=2)
    assert output.shape == TOKENS.shape
    assert int(output[0, 2]) != 4
