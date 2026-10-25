"""Contract test for ``QDiffusion._collect_energy_model_stats``."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from kaiwu.torch_plugin.qdiffusion import QDiffusion


def test_returns_empty_when_the_energy_model_has_no_stats_hook():
    module = SimpleNamespace(energy_model=SimpleNamespace())

    assert QDiffusion._collect_energy_model_stats(module) == {}


def test_returns_empty_when_the_stats_hook_is_not_callable():
    module = SimpleNamespace(
        energy_model=SimpleNamespace(get_last_stats={"not": "callable"})
    )

    assert QDiffusion._collect_energy_model_stats(module) == {}


def test_returns_the_energy_model_statistics():
    statistics = {"visible_on_ratio": torch.tensor(0.5)}
    module = SimpleNamespace(
        energy_model=SimpleNamespace(get_last_stats=lambda: statistics)
    )

    assert QDiffusion._collect_energy_model_stats(module) is statistics
