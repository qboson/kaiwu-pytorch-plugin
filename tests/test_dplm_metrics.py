"""Regression tests for the DPLM sequence-quality metrics."""

import importlib.util
import math
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


def test_jsd_is_zero_for_two_empty_distributions():
    assert metrics.jsd_from_counts({}, {}) == 0.0


def test_jsd_is_maximal_when_one_distribution_is_empty():
    assert math.isclose(
        metrics.jsd_from_counts({}, {"ABC": 3}), math.log(2.0), rel_tol=1e-12
    )
    assert math.isclose(
        metrics.jsd_from_counts({"ABC": 3}, {}), math.log(2.0), rel_tol=1e-12
    )


def test_jsd_matches_known_value_for_disjoint_supports():
    # Disjoint supports always give ln 2, regardless of the counts.
    assert math.isclose(
        metrics.jsd_from_counts({"A": 10}, {"B": 1}), math.log(2.0), rel_tol=1e-12
    )


def test_jsd_short_sequences_are_not_reported_as_perfect_match():
    """k-mer distributions are empty for sequences shorter than k."""
    reference = metrics.kmer_distribution(["AB", "AB"], 3)
    candidate = metrics.kmer_distribution(["ABC"], 3)
    assert reference == {}
    assert metrics.jsd_from_counts(reference, candidate) > 0.0


def test_jsd_still_zero_for_identical_distributions():
    counts = {"A": 2, "B": 1}
    assert metrics.jsd_from_counts(counts, dict(counts)) == pytest.approx(0.0, abs=1e-12)
