"""Contract test for ``dplm.utils.runtime.summarize_trainable_parameters``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "example" / "qdiffusion" / "dplm" / "utils" / "runtime.py"


def _load_runtime():
    spec = importlib.util.spec_from_file_location(
        "dplm_runtime_summary_uut", RUNTIME
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proposal = nn.Linear(3, 2)  # 8 parameters, trainable
        self.energy = nn.Linear(4, 5)  # 25 parameters, trainable
        self.frozen = nn.Parameter(torch.zeros(7), requires_grad=False)


def test_counts_total_and_trainable_parameters():
    runtime = _load_runtime()
    generator = _Generator()

    summary = runtime.summarize_trainable_parameters(generator)

    assert summary == {"total_parameters": 40, "trainable_parameters": 33}


def test_reports_zero_trainable_parameters_for_a_frozen_generator():
    runtime = _load_runtime()
    generator = _Generator()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)

    summary = runtime.summarize_trainable_parameters(generator)

    assert summary == {"total_parameters": 40, "trainable_parameters": 0}
