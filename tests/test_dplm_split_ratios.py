"""Regression tests for DPLM ``split_train_val_test`` ratio validation.

``split_train_val_test`` computed split sizes with ``max(1, int(n * ratio))``
and never validated the ratios: negative values silently shrank the intended
split back to one record, and NaN crashed later with an unrelated message.
"""

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DPLM_DIR = REPO_ROOT / "example" / "qdiffusion" / "dplm"


def _load_workflow_helpers(monkeypatch):
    """Loads workflows/workflow_helpers.py with sibling imports stubbed."""
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, **kwargs: iterable
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_stub)

    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [str(DPLM_DIR / "utils")]
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    for name, attrs in {
        "utils.io": (
            "normalize_decoded_sequence",
            "save_markdown",
            "write_fasta_records",
            "write_tsv_rows",
        ),
        "utils.metrics": ("QualitySummary",),
        "utils.runtime": ("encode_sequence", "seed_torch"),
    }.items():
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, object())
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "workflow_helpers_under_test", DPLM_DIR / "workflows" / "workflow_helpers.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _records(count):
    return [(f"seq_{index}", "ACDEFG") for index in range(count)]


@pytest.mark.parametrize("val_ratio", [-0.2, -1.0])
def test_negative_ratios_are_rejected(monkeypatch, val_ratio):
    helpers = _load_workflow_helpers(monkeypatch)

    with pytest.raises(ValueError, match="val_ratio"):
        helpers.split_train_val_test(
            _records(10), val_ratio=val_ratio, test_ratio=0.2, seed=0
        )


@pytest.mark.parametrize("test_ratio", [-0.2, -1.0])
def test_negative_test_ratios_are_rejected(monkeypatch, test_ratio):
    helpers = _load_workflow_helpers(monkeypatch)

    with pytest.raises(ValueError, match="test_ratio"):
        helpers.split_train_val_test(
            _records(10), val_ratio=0.2, test_ratio=test_ratio, seed=0
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, 1.0, 1.5])
def test_non_fraction_ratios_are_rejected(monkeypatch, bad):
    helpers = _load_workflow_helpers(monkeypatch)

    with pytest.raises(ValueError, match="ratio"):
        helpers.split_train_val_test(
            _records(10), val_ratio=bad, test_ratio=0.2, seed=0
        )
    with pytest.raises(ValueError, match="ratio"):
        helpers.split_train_val_test(
            _records(10), val_ratio=0.2, test_ratio=bad, seed=0
        )


def test_ratios_summing_to_one_are_rejected(monkeypatch):
    helpers = _load_workflow_helpers(monkeypatch)

    with pytest.raises(ValueError):
        helpers.split_train_val_test(
            _records(20), val_ratio=0.5, test_ratio=0.5, seed=0
        )


def test_valid_ratios_keep_the_documented_split_semantics(monkeypatch):
    helpers = _load_workflow_helpers(monkeypatch)
    records = _records(20)

    train, val, test = helpers.split_train_val_test(
        records, val_ratio=0.2, test_ratio=0.1, seed=123
    )

    assert len(val) == 4
    assert len(test) == 2
    assert len(train) == 14
    # No record duplicated or lost.
    combined = train + val + test
    assert sorted(record[0] for record in combined) == sorted(
        record[0] for record in records
    )


def test_splits_remain_deterministic_per_seed(monkeypatch):
    helpers = _load_workflow_helpers(monkeypatch)
    records = _records(12)

    first = helpers.split_train_val_test(records, val_ratio=0.25, test_ratio=0.25, seed=7)
    second = helpers.split_train_val_test(records, val_ratio=0.25, test_ratio=0.25, seed=7)

    assert first == second
    assert all(math.isfinite(index) for index in (0, 1, 2))
