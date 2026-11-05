# -*- coding: utf-8 -*-
"""DPLM comparison reports must reject empty aligned sets."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_UTILS_DIR = (
    Path(__file__).resolve().parents[1] / "example" / "qdiffusion" / "dplm" / "utils"
)


def _load_metrics_module():
    """Load ``utils/metrics.py`` without importing the heavy utils package."""
    package = types.ModuleType("dplm_utils_under_test")
    package.__path__ = [str(_UTILS_DIR)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.metrics", _UTILS_DIR / "metrics.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


metrics = _load_metrics_module()


def test_empty_aligned_sets_are_rejected() -> None:
    """Returning all-zero metrics invents a valid comparison from no pairs."""
    with pytest.raises(ValueError, match="aligned"):
        metrics.compare_generation_sets([], [], [])


def test_one_aligned_triple_still_reports_changes() -> None:
    result = metrics.compare_generation_sets(
        [("h", "A")],
        [("h", "A")],
        [("h", "C")],
    )

    assert result["changed_sequences"] == 1
    assert result["changed_fraction"] == 1.0
    assert result["baseline_identity_mean"] == 1.0
    assert result["guided_identity_mean"] == 0.0
