"""Contract test for ``FIDCalculator.compute`` and ``reset``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
HELPERS = REPO / "example" / "qvae_mnist" / "utils" / "helpers.py"


def _load_helpers():
    spec = importlib.util.spec_from_file_location(
        "qvae_mnist_helpers_fid_uut", HELPERS
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeFid:
    def __init__(self) -> None:
        self.reset_calls = 0

    def compute(self):
        return torch.tensor(1.25)

    def reset(self):
        self.reset_calls += 1


def _calculator(module):
    calculator = object.__new__(module.FIDCalculator)
    calculator.fid = _FakeFid()
    return calculator


def test_compute_returns_a_python_float():
    module = _load_helpers()
    calculator = _calculator(module)

    score = calculator.compute()

    assert isinstance(score, float)
    assert score == 1.25


def test_reset_delegates_to_the_underlying_fid_state():
    module = _load_helpers()
    calculator = _calculator(module)

    calculator.reset()

    assert calculator.fid.reset_calls == 1
