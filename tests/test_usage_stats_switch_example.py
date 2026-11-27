"""Contract tests for the usage-statistics switch example."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kaiwu.torch_plugin import usage_stats
from kaiwu.torch_plugin import (
    disable_usage_stats,
    enable_usage_stats,
    is_usage_stats_enabled,
)

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "example"
    / "usage_stats_switch"
    / "usage_stats_switch.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "usage_stats_switch_example", EXAMPLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = _load_example()


@pytest.fixture(autouse=True)
def _reset_switch(monkeypatch):
    monkeypatch.setattr(usage_stats, "_stats_enabled", None)
    monkeypatch.setattr(usage_stats, "_env_stats_cache", None)
    yield


def _child(value: str | None) -> str:
    env = dict(os.environ)
    env.pop("KPP_STATS_ENABLED", None)
    if value is not None:
        env["KPP_STATS_ENABLED"] = value
    completed = subprocess.run(
        [sys.executable, "-c", example.CHILD_PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def test_runtime_switch_toggles() -> None:
    disable_usage_stats()
    assert is_usage_stats_enabled() is False
    enable_usage_stats()
    assert is_usage_stats_enabled() is True


def test_default_is_enabled_in_a_fresh_process() -> None:
    assert _child(None) == "enabled"


def test_environment_variable_disables_statistics() -> None:
    assert _child("false") == "disabled"


def test_environment_variable_accepts_documented_true_values() -> None:
    for value in ("true", "1", "yes", "on"):
        assert _child(value) == "enabled"


def test_example_helper_matches_the_library() -> None:
    enable_usage_stats()
    assert example.state() == "enabled"
    assert example.child_state("false") == "disabled"
