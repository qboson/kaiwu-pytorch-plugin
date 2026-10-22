"""build_sampler must map every documented sampler name to its optimizer."""

import sys
import types
from pathlib import Path

import pytest
from kaiwu.classical import SimulatedAnnealingOptimizer

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"


@pytest.fixture
def etth1(monkeypatch):
    """Imports etth1_common with a stubbed pandas module."""
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda path: None
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("etth1_common", None)
    import etth1_common

    yield etth1_common
    sys.modules.pop("etth1_common", None)


def test_the_anderson_baseline_needs_no_sampler(etth1):
    assert etth1.build_sampler("anderson") is None


def test_simulated_annealing_is_returned(etth1):
    sampler = etth1.build_sampler("sa", seed=3)

    assert isinstance(sampler, SimulatedAnnealingOptimizer)


def test_unknown_sampler_names_are_rejected(etth1):
    with pytest.raises(ValueError, match="Unsupported sampler type"):
        etth1.build_sampler("qpu")
