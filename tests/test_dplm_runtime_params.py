"""Parameter-count contract for the DPLM runtime helpers."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from utils.runtime import summarize_trainable_parameters  # noqa: E402


def build_generator(freeze_bias=False):
    """Builds a two-layer module whose bias can be frozen."""
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 1),
    )
    if freeze_bias:
        for module in model:
            if isinstance(module, torch.nn.Linear):
                module.bias.requires_grad_(False)
    return model


def test_counts_every_parameter_element():
    """total_parameters sums numel across all parameter tensors."""
    counts = summarize_trainable_parameters(build_generator())
    # (2 * 3 + 3) + (3 * 1 + 1) = 9 + 4
    assert counts == {"total_parameters": 13, "trainable_parameters": 13}


def test_frozen_parameters_leave_the_trainable_count():
    """require_grad=False parameters still count as total but not trainable."""
    counts = summarize_trainable_parameters(build_generator(freeze_bias=True))
    # total stays 13; the two biases (3 + 1) are frozen
    assert counts == {"total_parameters": 13, "trainable_parameters": 9}


def test_module_without_parameters_reports_zero():
    """A parameter-free generator reports two zero counters."""
    counts = summarize_trainable_parameters(torch.nn.Sequential())
    assert counts == {"total_parameters": 0, "trainable_parameters": 0}
