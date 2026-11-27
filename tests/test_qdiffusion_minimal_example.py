# -*- coding: utf-8 -*-
"""Tests for the minimal Q-Diffusion example with a custom energy model."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

EXAMPLE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../example/qdiffusion_minimal")
)
if EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLE_ROOT)

from qdiffusion_minimal import (  # noqa: E402  # pylint: disable=wrong-import-position
    BOS_ID,
    EOS_ID,
    MASK_ID,
    PAD_ID,
    ORDINARY_TOKENS,
    build_generator,
    generate_demo,
)


def test_generation_decodes_every_mask() -> None:
    """Generated sequences keep their specials and contain no mask tokens."""
    result = generate_demo(steps=4, seed=0)
    tokens = result["tokens"]

    assert len(tokens) == 6
    assert tokens[0] == BOS_ID and tokens[-1] == EOS_ID
    assert MASK_ID not in tokens and PAD_ID not in tokens
    assert set(tokens[1:-1]).issubset(set(ORDINARY_TOKENS))


def test_energy_model_scores_once_per_step() -> None:
    """The custom energy model is consulted by the reranking step."""
    result = generate_demo(steps=4, seed=0)
    assert result["energy_calls"] == 4


def test_generation_is_deterministic_per_seed() -> None:
    """The same seed reproduces the same token sequence."""
    first = generate_demo(steps=3, seed=11)
    second = generate_demo(steps=3, seed=11)
    assert first["tokens"] == second["tokens"]


def test_partial_masks_are_preserved() -> None:
    """Positions marked as fixed survive generation unchanged."""
    torch.manual_seed(0)
    generator = build_generator(seed=0)
    tokens = torch.tensor([[BOS_ID, 3, MASK_ID, MASK_ID, 5, EOS_ID]])
    partial = torch.zeros_like(tokens, dtype=torch.bool)
    partial[0, 1] = True
    partial[0, 4] = True

    output = generator.generate(tokens, max_steps=3, partial_masks=partial)

    # the two fixed positions keep their observed values ...
    assert int(output[0, 1]) == 3 and int(output[0, 4]) == 5
    # ... while the editable mask positions are decoded
    assert MASK_ID not in output[0, 2:4].tolist()


def test_steps_must_be_positive() -> None:
    """A non-positive step count fails early."""
    with pytest.raises(ValueError):
        generate_demo(steps=0)


def test_cli_entry_point_passes() -> None:
    """The documented command reports a PASS and exits successfully."""
    script = os.path.join(EXAMPLE_ROOT, "qdiffusion_minimal.py")
    proc = subprocess.run(
        [sys.executable, script, "--steps", "3"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=EXAMPLE_ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "PASS" in proc.stdout
    assert "energy-model calls" in proc.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
