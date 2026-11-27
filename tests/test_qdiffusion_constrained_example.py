"""Contract tests for the constrained Q-Diffusion example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "example"
    / "qdiffusion_constrained"
    / "constrained_generate.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "qdiffusion_constrained_example", EXAMPLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = _load_example()
INPUT_TOKENS = torch.tensor([[0, 2, 5, 6, 7, 1]])
FIXED = torch.tensor([[False, False, True, True, False, False]])
EDITABLE = ~FIXED & INPUT_TOKENS.ne(0) & INPUT_TOKENS.ne(1) & INPUT_TOKENS.ne(3)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fixed_positions_survive_generation(seed: int) -> None:
    generator = example.build_generator(seed)
    output = generator.generate(INPUT_TOKENS, max_steps=8, partial_masks=FIXED)
    assert torch.equal(output[FIXED], INPUT_TOKENS[FIXED])


@pytest.mark.parametrize("seed", [0, 1])
def test_editable_positions_never_become_special_tokens(seed: int) -> None:
    generator = example.build_generator(seed)
    output = generator.generate(INPUT_TOKENS, max_steps=8, partial_masks=FIXED)
    for token in output[EDITABLE]:
        assert int(token) not in (
            example.BOS_ID,
            example.EOS_ID,
            example.PAD_ID,
            example.MASK_ID,
        )


def test_initialize_state_keeps_fixed_tokens_and_masks_editable_ones() -> None:
    generator = example.build_generator(0)
    state = generator.initialize_state(
        INPUT_TOKENS, partial_masks=FIXED, max_steps=8
    )
    output_tokens = state["output_tokens"]
    assert int(output_tokens[0, 2]) == 5
    assert int(output_tokens[0, 3]) == 6
    assert int(output_tokens[0, 4]) == example.MASK_ID
    assert int(output_tokens[0, 0]) == example.BOS_ID


def test_energy_model_is_consulted_for_reranking() -> None:
    class CountingEnergy(example.PreferManySevens):  # pylint: disable=too-few-public-methods
        """Energy model that records how often it is consulted."""

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
            self.calls += 1
            return super().score_conditioned(
                noisy_tokens, candidate_tokens, attention_mask
            )

    generator = example.build_generator(0)
    energy_model = CountingEnergy()
    generator.energy_model = energy_model
    generator.generate(INPUT_TOKENS, max_steps=4, partial_masks=FIXED)
    assert energy_model.calls > 0
