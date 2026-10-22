"""Contract tests for the manual Q-Diffusion decode example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "example"
    / "qdiffusion_manual_decode"
    / "manual_decode.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "qdiffusion_manual_decode_example", EXAMPLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = _load_example()
INPUT_TOKENS = torch.tensor([[0, 2, 5, 6, 7, 1]])
FIXED = torch.tensor([[False, False, True, True, False, False]])


def _state(generator, max_steps: int = example.MAX_STEPS):
    return generator.initialize_state(
        INPUT_TOKENS, partial_masks=FIXED, max_steps=max_steps
    )


def test_step_advances_the_state_and_history() -> None:
    generator = example.build_generator()
    state = _state(generator)
    assert state["step"] == 0
    assert len(state["history"]) == 1
    next_state = generator.step(state)
    assert next_state["step"] == 1
    assert len(next_state["history"]) == 2
    assert next_state["output_tokens"].shape == INPUT_TOKENS.shape


def test_step_does_not_mutate_the_previous_state() -> None:
    generator = example.build_generator()
    state = _state(generator)
    before = state["output_tokens"].clone()
    generator.step(state)
    assert state["step"] == 0
    assert torch.equal(state["output_tokens"], before)


def test_fixed_positions_survive_every_manual_step() -> None:
    generator = example.build_generator()
    state = _state(generator)
    for _ in range(example.MAX_STEPS):
        state = generator.step(state)
        assert torch.equal(state["output_tokens"][FIXED], INPUT_TOKENS[FIXED])


def test_manual_loop_matches_generate_under_the_same_seed() -> None:
    generator = example.build_generator()
    torch.manual_seed(0)
    manual = _state(generator)
    for _ in range(example.MAX_STEPS):
        manual = generator.step(manual)

    torch.manual_seed(0)
    generated = generator.generate(
        INPUT_TOKENS, partial_masks=FIXED, max_steps=example.MAX_STEPS
    )
    assert torch.equal(manual["output_tokens"], generated)


def test_decoding_resolves_every_mask_token() -> None:
    generator = example.build_generator()
    state = _state(generator)
    for _ in range(example.MAX_STEPS):
        state = generator.step(state)
    assert example.masked_positions(state["output_tokens"]) == 0
