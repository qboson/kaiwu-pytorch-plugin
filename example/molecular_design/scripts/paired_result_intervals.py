#!/usr/bin/env python3
"""Summarize paired differences in recorded CIM and SQA results."""

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import random
from statistics import quantiles

from verify_results import ROOT, truth


def _pair_records(records, target_property, sampling_regime):
    """Read comparable solver rows keyed by condition and step."""
    paired = defaultdict(dict)
    with gzip.open(records, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            solver = row["solver"]
            if solver not in {"cim", "sqa"}:
                continue
            if target_property is not None and row["target_prop"] != target_property:
                continue
            if sampling_regime is not None and row["sampling_regime"] != sampling_regime:
                continue
            key = (row["condition"], int(row["global_step"]))
            if solver in paired[key]:
                raise ValueError(f"Duplicate {solver} record for {key}")
            identity = (
                row["reference_id"], row["target_prop"], row["target_band_index"],
                row["condition_seed"], row["phase"], row["target_low"], row["target_high"],
                row["sampling_regime"], int(row["solver_returned_reads"]),
            )
            valid, hit = truth(row["valid"]), truth(row["target_hit"])
            if identity[-1] <= 0 or (hit and not valid):
                raise ValueError(f"Invalid read count or hit flag for {key}")
            paired[key][solver] = (identity, valid, hit)
    return paired


def _reference_counts(paired):
    """Accumulate valid/hit differences together for each reference."""
    totals = defaultdict(lambda: [0, 0, 0])
    for key, pair in paired.items():
        if set(pair) != {"cim", "sqa"}:
            raise ValueError(f"Missing CIM or SQA record for {key}")
        cim_identity, cim_valid, cim_hit = pair["cim"]
        sqa_identity, sqa_valid, sqa_hit = pair["sqa"]
        if cim_identity != sqa_identity:
            raise ValueError(f"Pair metadata or candidate budgets differ for {key}")
        cluster = totals[cim_identity[0]]
        cluster[0] += int(cim_valid) - int(sqa_valid)
        cluster[1] += int(cim_hit) - int(sqa_hit)
        cluster[2] += 1
    return [totals[reference] for reference in sorted(totals)]


def _difference_percentages(clusters):
    valid, hits, steps = map(sum, zip(*clusters))
    return 100 * valid / steps, 100 * hits / steps


def compare_records(records, *, repeats=2000, seed=42, target_property=None, sampling_regime=None):
    """Return pooled rate differences and paired reference-bootstrap intervals."""
    if repeats < 2:
        raise ValueError("Use at least two bootstrap repetitions")
    pairs = _pair_records(records, target_property, sampling_regime)
    clusters = _reference_counts(pairs)
    if len(clusters) < 2:
        raise ValueError("Paired intervals require at least two references")
    rng = random.Random(seed)
    samples = [
        _difference_percentages(rng.choices(clusters, k=len(clusters)))
        for _ in range(repeats)
    ]
    report = {
        "references": len(clusters),
        "paired_steps": len(pairs),
        "target_property": target_property,
        "sampling_regime": sampling_regime,
        "bootstrap_repetitions": repeats,
        "seed": seed,
    }
    for name, point, simulated in zip(
        ("validity", "target_hit"), _difference_percentages(clusters), zip(*samples)
    ):
        cut_points = quantiles(simulated, n=40, method="inclusive")
        report[name] = {
            "cim_minus_sqa_pp": point,
            "interval_95_pp": [cut_points[0], cut_points[-1]],
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "results" / "selected_steps.csv.gz")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--property", choices=("qed", "logp"))
    parser.add_argument("--regime", choices=("historical_5_calls", "single_call"))
    args = parser.parse_args()
    report = compare_records(
        args.records, repeats=args.repeats, seed=args.seed, target_property=args.property,
        sampling_regime=args.regime,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
