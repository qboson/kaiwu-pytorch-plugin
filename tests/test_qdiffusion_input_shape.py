"""QDiffusion decode inputs must be 2-D token tensors with aligned masks."""

from __future__ import annotations

import pytest
import torch

from kaiwu.torch_plugin import EnergyModel, QDiffusion, QDiffusionConfig
from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec

BATCH = 2
SEQ_LEN = 4


class _Proposal(torch.nn.Module):  # pylint: disable=too-few-public-methods
    def forward(self, tokens):
        return torch.zeros(tokens.size(0), tokens.size(1), 8)


class _ZeroEnergy(EnergyModel):  # pylint: disable=too-few-public-methods
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del candidate_tokens, attention_mask
        return torch.zeros(noisy_tokens.size(0), 1)


def _generator() -> QDiffusion:
    spec = SequenceTokenSpec(
        mask_id=4, pad_id=3, bos_id=0, eos_id=1, x_id=2, tokenizer=None
    )
    return QDiffusion(
        _Proposal(),
        _ZeroEnergy(),
        spec,
        QDiffusionConfig(num_diffusion_timesteps=4, resample_ratio=1.0),
    )


def _tokens() -> torch.Tensor:
    return torch.tensor([[0, 2, 5, 1], [0, 2, 6, 1]])


def test_generate_rejects_one_dimensional_tokens() -> None:
    generator = _generator()
    with pytest.raises(ValueError, match="input_tokens must have shape"):
        generator.generate(torch.tensor([0, 2, 5, 1]), max_steps=2)


def test_initialize_state_rejects_one_dimensional_tokens() -> None:
    generator = _generator()
    with pytest.raises(ValueError, match="input_tokens must have shape"):
        generator.initialize_state(torch.tensor([0, 2, 5, 1]), max_steps=2)


def test_non_tensor_tokens_are_rejected() -> None:
    generator = _generator()
    with pytest.raises(TypeError, match="input_tokens must be a torch.Tensor"):
        generator.initialize_state([[0, 2, 5, 1]], max_steps=2)


def test_partial_masks_must_match_the_token_shape() -> None:
    generator = _generator()
    with pytest.raises(ValueError, match="partial_masks must have the same shape"):
        generator.initialize_state(
            _tokens(), partial_masks=torch.zeros(3, SEQ_LEN, dtype=torch.bool), max_steps=2
        )


def test_partial_masks_must_be_two_dimensional() -> None:
    generator = _generator()
    with pytest.raises(ValueError, match="partial_masks must have shape"):
        generator.initialize_state(
            _tokens(), partial_masks=torch.zeros(1, BATCH, SEQ_LEN, dtype=torch.bool), max_steps=2
        )


def test_step_rejects_mismatched_partial_masks() -> None:
    generator = _generator()
    state = generator.initialize_state(_tokens(), max_steps=2)
    with pytest.raises(ValueError, match="partial_masks must have the same shape"):
        generator.step(state, partial_masks=torch.zeros(3, SEQ_LEN, dtype=torch.bool))


def test_two_dimensional_inputs_still_work() -> None:
    generator = _generator()
    tokens = _tokens()
    masks = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    state = generator.initialize_state(tokens, partial_masks=masks, max_steps=2)
    assert state["output_tokens"].shape == (BATCH, SEQ_LEN)
    output = generator.generate(tokens, partial_masks=masks, max_steps=2)
    assert output.shape == (BATCH, SEQ_LEN)
