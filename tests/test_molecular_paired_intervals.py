"""Matched result intervals keep every trajectory within its reference cluster."""

import csv
import gzip
import importlib
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def report_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "example" / "molecular_design" / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    names = ("verify_results", "paired_result_intervals")
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield importlib.import_module("paired_result_intervals")
    for name in names:
        sys.modules.pop(name, None)


def rows():
    common = dict(
        target_prop="qed", target_band_index=0, condition_seed=42, phase="explore",
        target_low=0.2, target_high=0.5, sampling_regime="single_call",
        solver_returned_reads=10,
    )
    pairs = [
        ("a", 0, (True, True), (True, False)),
        ("a", 1, (True, True), (False, False)),
        ("b", 0, (False, False), (True, True)),
    ]
    return [
        dict(common, condition=f"qed_b0_{ref}_s42", reference_id=ref, global_step=step,
             solver=solver, valid=valid, target_hit=hit)
        for ref, step, cim, sqa in pairs
        for solver, (valid, hit) in (("cim", cim), ("sqa", sqa))
    ]


def write_rows(path, records):
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def test_cli_reports_cluster_intervals_without_solver_dependencies(
    report_module, monkeypatch, tmp_path, capsys,
):
    path = tmp_path / "steps.csv.gz"
    write_rows(path, rows())
    monkeypatch.setattr(sys, "argv", [
        "paired_result_intervals.py", "--records", str(path), "--property", "qed",
        "--regime", "single_call", "--seed", "5",
    ])
    report_module.main()
    result = json.loads(capsys.readouterr().out)
    assert result["references"] == 2 and result["paired_steps"] == 3
    assert result["validity"]["cim_minus_sqa_pp"] == 0
    assert result["target_hit"]["cim_minus_sqa_pp"] == pytest.approx(100 / 3)
    assert result["target_hit"]["interval_95_pp"] == [-100, 100]

    write_rows(path, list(reversed(rows())))
    reordered = report_module.compare_records(
        path, seed=5, target_property="qed", sampling_regime="single_call"
    )
    assert reordered == result


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "budget"])
def test_rejects_incomparable_record_pairs(report_module, tmp_path, invalid):
    path = tmp_path / "steps.csv.gz"
    records = rows()
    if invalid == "missing":
        records.pop()
    elif invalid == "duplicate":
        records.append(records[0])
    else:
        records[1]["solver_returned_reads"] = 50
    write_rows(path, records)
    with pytest.raises(ValueError, match="Missing|Duplicate|budgets"):
        report_module.compare_records(path)
