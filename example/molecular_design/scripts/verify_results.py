#!/usr/bin/env python3
"""Verify the published CIM/SQA/SA selected-step results without solver dependencies."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def truth(value):
    require(value.lower() in ("true", "false"), "Invalid boolean")
    return value.lower() == "true"


def verify():
    result_dir = ROOT / "results"
    provenance = json.loads((result_dir / "provenance.json").read_text())
    for name, expected in provenance["published_sha256"].items():
        require(sha(ROOT / name) == expected, "Published file changed: " + name)
    groups, conditions, matched = {}, {}, {}
    with gzip.open(result_dir / "selected_steps.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            solver = row["solver"]
            require(solver in ("cim", "sqa", "sa"), "Unknown solver")
            prop, segment = row["target_prop"], row["sampling_regime"]
            require(prop in ("qed", "logp"), "Unknown property")
            require(segment in ("historical_5_calls", "single_call"), "Unknown segment")
            step = int(row["global_step"])
            condition = row["condition"]
            require(condition == f'{prop}_b{row["target_band_index"]}_{row["reference_id"]}_s42',
                    "Condition identifier differs")
            seen = conditions.setdefault((solver, condition), set())
            require(step not in seen, "Duplicate step")
            seen.add(step)
            valid, hit = truth(row["valid"]), truth(row["target_hit"])
            expected_hit = valid and float(row["target_low"]) <= float(row[prop]) < float(row["target_high"])
            require(hit == expected_hit, "Target hit differs from recorded property")
            reads = int(row["solver_returned_reads"])
            require(reads in (10, 50, 100), "Unexpected candidate budget")
            if solver == "sa":
                require(reads == 50, "SA candidate budget differs")
            if solver in ("cim", "sqa") and segment == "single_call":
                require(reads == 10, "Single-call candidate budget differs")
            identity = (prop, row["target_band_index"], row["reference_id"],
                        row["condition_seed"], row["phase"], row["target_low"], row["target_high"], segment)
            matched.setdefault((condition, step), {})[solver] = (identity, reads)
            for part in ("all", segment):
                for property_group in ("all", prop):
                    counts = groups.setdefault((part, property_group), Counter())
                    counts[solver + "_steps"] += 1
                    counts[solver + "_valid"] += valid
                    counts[solver + "_hits"] += hit
                    counts[solver + "_reads"] += reads
    require(len(conditions) == 750 and all(v == set(range(100)) for v in conditions.values()),
            "Expected 250 complete conditions per solver")
    require(len(matched) == 25000, "Expected 25000 matched steps")
    for values in matched.values():
        require(set(values) == {"cim", "sqa", "sa"}, "Missing solver")
        require(values["cim"][0] == values["sqa"][0] == values["sa"][0], "Conditions differ")
        require(values["cim"][1] == values["sqa"][1], "CIM/SQA candidate budgets differ")
    with (result_dir / "comparison.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    require(len(summary) == 9, "Expected nine summary groups")
    seen_groups = set()
    for row in summary:
        key = (row["segment"], row["property"])
        require(key not in seen_groups, "Duplicate summary group")
        seen_groups.add(key)
        counts = groups[key]
        steps = int(row["steps"])
        for solver in ("cim", "sqa", "sa"):
            require(counts[solver + "_steps"] == steps, "Step count mismatch")
            for metric in ("valid", "hits", "reads"):
                field = solver + "_" + metric
                require(counts[field] == int(row[field]), "Count mismatch: " + field)
            for metric, rate in (("valid", "valid_pct"), ("hits", "hit_pct")):
                require(abs(100 * counts[solver + "_" + metric] / steps
                            - float(row[solver + "_" + rate])) < 1e-9, "Rate mismatch")
    require(seen_groups == set(groups), "Missing summary group")
    return {"verified": True, "conditions_per_solver": 250, "steps_per_solver": 25000,
            "selected_records": 75000, "cim_sqa_candidates_each": groups[("all", "all")]["cim_reads"],
            "summary_groups": 9,
            "scope": "Selected CSV records and summary statistics; no re-sampling or chemistry recomputation."}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
