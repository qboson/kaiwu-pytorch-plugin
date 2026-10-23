"""Seed contract for reproducible DPLM generation runs."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from utils.runtime import seed_torch  # noqa: E402


def test_same_seed_replays_the_same_draws():
    """Re-seeding with the same value reproduces the next CPU draws."""
    seed_torch(1234)
    first = torch.rand(4)
    seed_torch(1234)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_different_seeds_do_not_share_a_stream():
    """Different seeds must produce different random streams."""
    seed_torch(1)
    first = torch.rand(4)
    seed_torch(2)
    second = torch.rand(4)
    assert not torch.equal(first, second)
