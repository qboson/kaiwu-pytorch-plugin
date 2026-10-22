"""Aligned quality metrics may not silently discard missing sequences."""
import importlib
from pathlib import Path
import pytest


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("example.qdiffusion.dplm.utils.metrics")


@pytest.mark.parametrize("reference_count,generated_count", [(2, 1), (1, 2), (0, 1), (1, 0)])
def test_quality_rejects_missing_or_extra_records(metrics, reference_count, generated_count):
    records = [("a", "AAA"), ("b", "CCC")]
    with pytest.raises(ValueError, match="equal|aligned"):
        metrics.evaluate_generation_quality(records[:reference_count], records[:generated_count], label="bad")


@pytest.mark.parametrize("counts", [(2, 1, 2), (2, 2, 1), (1, 2, 2), (0, 0, 0)])
def test_comparison_requires_three_nonempty_aligned_sets(metrics, counts):
    records = [("a", "AAA"), ("b", "CCC")]
    with pytest.raises(ValueError, match="equal|aligned"):
        metrics.compare_generation_sets(*(records[:count] for count in counts))


def test_equal_count_metrics_remain_positional(metrics):
    references = [("r1", "AAA"), ("r2", "CCC")]
    baseline = [("b1", "AAA"), ("b2", "CCC")]
    guided = [("g1", "AAA"), ("g2", "DDD")]
    result = metrics.compare_generation_sets(references, baseline, guided)
    assert result["changed_fraction"] == 0.5
    assert result["baseline_identity_mean"] == 1.
    assert result["guided_identity_mean"] == 0.5
    assert metrics.evaluate_generation_quality(references, baseline, label="ok").length_match_ratio == 1.
