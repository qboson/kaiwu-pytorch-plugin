"""Comparison reports use the same decoded-sequence normalization as quality scores."""

import importlib

import pytest

pytest.importorskip("transformers")
pytest.importorskip("omegaconf")

metrics = importlib.import_module("example.qdiffusion.dplm.utils.metrics")


def test_comparison_ignores_tokenizer_spacing():
    """Equivalent token text must not be reported as a changed generation."""
    reference = [("first", "ACDE"), ("second", "F G H")]
    baseline = [("first", " A C D E "), ("second", "FGH")]
    guided = [("first", "ACDE"), ("second", " F G H ")]
    result = metrics.compare_generation_sets(reference, baseline, guided)
    assert result == {
        "changed_sequences": 0,
        "changed_fraction": 0.0,
        "baseline_identity_mean": 1.0,
        "guided_identity_mean": 1.0,
    }
    for name, records in (("baseline", baseline), ("guided", guided)):
        summary = metrics.evaluate_generation_quality(reference, records, label=name)
        assert result[f"{name}_identity_mean"] == summary.identity_to_reference_mean
    assert baseline[0][1] == " A C D E "


def test_comparison_retains_real_residue_changes():
    """Whitespace normalization does not hide sequence changes or alter pair order."""
    reference = [("a", "ACDE"), ("b", "FGHI")]
    baseline = [("a", "ACDE"), ("b", "FGHI")]
    guided = [("a", "A C D F"), ("b", " F G H I ")]
    result = metrics.compare_generation_sets(reference, baseline, guided)
    assert result["changed_sequences"] == 1
    assert result["changed_fraction"] == 0.5
    assert result["baseline_identity_mean"] == 1.0
    assert result["guided_identity_mean"] == 0.875


def test_compact_sequence_comparison_is_unchanged():
    """Keep the existing shared-prefix identity convention for clean input."""
    reference = [("a", "ACDE")]
    result = metrics.compare_generation_sets(reference, [("b", "ACDF")], [("c", "AC")])
    assert result["baseline_identity_mean"] == 0.75
    assert result["guided_identity_mean"] == 1.0
    assert result["changed_sequences"] == 1
