# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""The quick-start RBM example must actually apply its optimizer step.

``example/run_rbm.py`` computes the objective and calls ``backward()``; without
``opt_rbm.step()`` the parameters never move, which teaches the wrong training
loop. The solver is replaced with a deterministic stand-in so the test runs
without a Kaiwu license.
"""

import os
from pathlib import Path
import runpy

import numpy as np
import pytest
import torch
from unittest.mock import patch

os.environ.setdefault("KPP_STATS_ENABLED", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "example" / "run_rbm.py"


class DummySampler:
    """Return one all-ones spin configuration for any Ising matrix."""

    def solve(self, matrix):
        """Mimic the Kaiwu solver contract used by ``sample()``."""
        nodes = int(np.asarray(matrix).shape[0])
        return np.ones((1, nodes), dtype=np.float32)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_run_rbm_applies_optimizer_step():
    """At least one parameter changes after running the example."""
    import kaiwu.torch_plugin as plugin

    real_model = plugin.RestrictedBoltzmannMachine
    captured = {}

    def factory(*args, **kwargs):
        model = real_model(*args, **kwargs)
        captured["model"] = model
        captured["before"] = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
        }
        return model

    with (
        patch("kaiwu.classical.SimulatedAnnealingOptimizer", DummySampler),
        patch("kaiwu.torch_plugin.RestrictedBoltzmannMachine", factory),
    ):
        runpy.run_path(str(SCRIPT), run_name="__main__")

    model = captured["model"]
    before = captured["before"]
    after = {name: value.detach().clone() for name, value in model.named_parameters()}
    changed = [
        name for name in before if not torch.equal(before[name], after[name])
    ]
    assert changed, "run_rbm.py finished without updating any parameter"

