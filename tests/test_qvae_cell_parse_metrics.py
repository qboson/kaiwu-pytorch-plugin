# -*- coding: utf-8 -*-
"""The qvae_cell --metrics parser must expand `all` consistently."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

QVAE_CELL_ROOT = Path(__file__).resolve().parents[1] / "example" / "qvae_cell"


def _load_benchmark_module():
    # The parser is importable without the optional single-cell stack.
    sys.modules.setdefault("anndata", types.ModuleType("anndata"))
    sys.modules.setdefault("scanpy", types.ModuleType("scanpy"))
    spec = importlib.util.spec_from_file_location(
        "benchmark_under_test", QVAE_CELL_ROOT / "evaluate_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


def test_all_combined_with_a_group_runs_every_group() -> None:
    """`all,clustering` currently returns ['all', 'clustering'] and skips 3 groups."""
    assert benchmark.parse_metrics("all,clustering") == list(
        benchmark.VALID_METRICS
    )


def test_all_alone_still_runs_every_group() -> None:
    assert benchmark.parse_metrics("all") == list(benchmark.VALID_METRICS)


def test_duplicate_groups_are_removed() -> None:
    assert benchmark.parse_metrics("clustering,classification,clustering") == [
        "clustering",
        "classification",
    ]


def test_unknown_group_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        benchmark.parse_metrics("all,not-a-metric")
