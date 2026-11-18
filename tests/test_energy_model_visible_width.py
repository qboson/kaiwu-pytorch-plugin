# -*- coding: utf-8 -*-
"""EnergyModel visible logits must match the configured BM visible width."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin import EnergyModel  # noqa: E402  pylint: disable=wrong-import-position


class _DummySampler:
    def solve(self, ising_matrix):
        return np.ones((1, ising_matrix.shape[0]), dtype=np.float32)


@pytest.mark.parametrize("width", [2, 4])
def test_wrong_visible_width_is_rejected(width: int) -> None:
    """A width mismatch currently conditions on the wrong BM units silently."""
    model = EnergyModel(
        bm_num_visible=3,
        bm_num_hidden=2,
        sampler=_DummySampler(),
    )

    with pytest.raises(ValueError, match="bm_num_visible"):
        model.score_visible_logits(torch.randn(1, width))


def test_correct_visible_width_still_scores() -> None:
    model = EnergyModel(
        bm_num_visible=3,
        bm_num_hidden=2,
        sampler=_DummySampler(),
    )

    energy = model.score_visible_logits(torch.randn(2, 3))

    assert energy.shape == (2, 1)
    assert torch.isfinite(energy).all()
