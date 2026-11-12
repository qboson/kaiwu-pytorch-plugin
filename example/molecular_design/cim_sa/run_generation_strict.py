#!/usr/bin/env python3
"""Audited checkpoint-reuse generation over the fixed 25-reference manifest."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parent
sys.path.insert(0, str(SOURCE_ROOT))

TABLE2_TARGETS = {
    "qed": [
        (0.526, 0.615), (0.673, 0.721), (0.721, 0.759),
        (0.759, 0.790), (0.819, 0.847),
    ],
    "logp": [
        (0.522, 1.295), (1.799, 2.222), (2.222, 2.583),
        (2.583, 2.945), (3.313, 3.688),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict 25-reference checkpoint-reuse generation"
    )
    parser.add_argument("--prop", default="qed", choices=("qed", "logp", "sas"))
    parser.add_argument(
        "--solver", default="sqa",
        choices=("kaiwu", "kaiwu_cim", "neal", "sqa"),
    )
    parser.add_argument(
        "--checkpoint", default=str(PROJECT_ROOT / "checkpoints/checkpoint_qc.pt")
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--reference-manifest", default="references_25.json",
        help="Immutable 25-reference JSON manifest",
    )
    parser.add_argument(
        "--reference-ids", default="all",
        help="'all' or comma-separated stable IDs such as R00,R01",
    )
    parser.add_argument(
        "--bands", default="all",
        help="'all' or comma-separated zero-based target-band indices",
    )
    parser.add_argument("--qubo-mode", default="strict",
                        choices=("strict", "compensated"))
    parser.add_argument("--sweeps", type=int, default=None)
    parser.add_argument("--reads", type=int, default=None)
    parser.add_argument("--n-explore", type=int, default=None)
    parser.add_argument("--n-optimize", type=int, default=None)
    parser.add_argument(
        "--valency-penalty", type=float, default=None,
        help=(
            "Fixed lambda multiplying the paper Eq. (2) structural QUBO; "
            "the paper does not report its numeric value"
        ),
    )
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--lambda-mode", choices=("fixed", "cim_precision"), default="fixed",
                        help="cim_precision is a hardware-adapted protocol, not fixed-lambda reproduction")
    parser.add_argument("--cim-penalty-ratio", type=float, default=4.0)
    parser.add_argument(
        "--exploration-file", default=None,
        help="Optional JSON mapping 'prop:band_index:reference_id' to bitstrings",
    )
    parser.add_argument(
        "--require-v2-checkpoint", action="store_true",
        help="Reject legacy checkpoints without repaired-training provenance",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_condition_seed(
    base_seed: int, prop: str, band_index: int, reference_id: str
) -> int:
    """Seed is independent of sharding order and solver backend."""
    prop_code = {"qed": 1, "logp": 2, "sas": 3}[prop]
    ref_code = int(reference_id[1:])
    return int(base_seed + prop_code * 1_000_000 + band_index * 1_000 + ref_code)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def atomic_csv(rows: List[Dict], path: Path, columns: List[str] | None = None) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    os.environ["QC_QUBO_SOLVER"] = args.solver

    from qc_mol_design.config import CFG, ATOM_FEATURE_SCHEMA
    from qc_mol_design.data import (
        load_zinc12k, split_dataset, load_reference_manifest,
        select_references, manifest_sha256,
    )
    from qc_mol_design.models import build_fingerprint_model, CRBM
    from qc_mol_design.optimization import MolecularDesigner
    from qc_mol_design.optimization.designer import _count_rings
    from qc_mol_design.evaluation import (
        summarise_generation, generation_quality_metrics,
    )

    CFG.data_path = str(PROJECT_ROOT / "zinc250k.csv")
    CFG.target_prop = args.prop
    CFG.qubo_mode = args.qubo_mode
    CFG.lambda_mode = args.lambda_mode
    CFG.cim_penalty_ratio = args.cim_penalty_ratio
    if args.sweeps is not None:
        CFG.sa_sweeps = args.sweeps
    if args.reads is not None:
        CFG.sa_reads = args.reads
    if args.n_explore is not None:
        CFG.n_explore = args.n_explore
    if args.n_optimize is not None:
        CFG.n_optimize = args.n_optimize
    if args.valency_penalty is not None:
        if args.valency_penalty <= 0:
            raise ValueError("--valency-penalty must be positive")
        CFG.valency_penalty = args.valency_penalty

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.reference_manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    checkpoint_path = Path(args.checkpoint)

    data = load_zinc12k()
    train_df, _valid_df, test_df = split_dataset(data, seed=CFG.seed)
    all_references = load_reference_manifest(
        manifest_path, test_df=test_df, expected_count=25
    )
    requested_ids = None
    if args.reference_ids != "all":
        requested_ids = [
            value.strip() for value in args.reference_ids.split(",")
            if value.strip()
        ]
    references = select_references(all_references, requested_ids)

    all_bands = TABLE2_TARGETS.get(args.prop, [CFG.target_range])
    if args.bands == "all":
        band_indices = list(range(len(all_bands)))
    else:
        band_indices = [
            int(value) for value in args.bands.split(",") if value.strip()
        ]
    invalid_bands = [
        index for index in band_indices
        if index < 0 or index >= len(all_bands)
    ]
    if invalid_bands:
        raise ValueError(f"invalid band indices: {invalid_bands}")

    checkpoint = torch.load(checkpoint_path, map_location=CFG.device, weights_only=False)
    checkpoint_version = int(checkpoint.get("format_version", 1))
    if args.require_v2_checkpoint and checkpoint_version < 2:
        raise ValueError(
            "legacy checkpoint rejected: rerun repaired training or omit "
            "--require-v2-checkpoint for a provisional generation-only analysis"
        )
    if checkpoint_version < 3 or checkpoint.get("feature_schema") != ATOM_FEATURE_SCHEMA:
        raise ValueError(
            "checkpoint is incompatible with the repaired strict training "
            "protocol: retrain a format-v3 checkpoint with nine-heavy-atom "
            "features before generating strict results"
        )

    gcn = build_fingerprint_model()
    crbm = CRBM().to(CFG.device)
    gcn.load_state_dict(checkpoint["gcn"])
    crbm.load_state_dict(checkpoint["crbm"])
    gcn.eval()
    crbm.eval()
    designer = MolecularDesigner(gcn, crbm)

    exploration = {}
    exploration_hash = None
    if args.exploration_file:
        exploration_path = Path(args.exploration_file)
        exploration = json.loads(exploration_path.read_text(encoding="utf-8"))
        exploration_hash = file_sha256(exploration_path)

    all_raw: List[Dict] = []
    all_trajectory: List[Dict] = []
    all_solver_reads: List[Dict] = []
    all_unique_per_reference: List[Dict] = []
    reference_rates: List[Dict] = []
    table2_rows: List[Dict] = []
    condition_meta: List[Dict] = []
    train_smiles = set(train_df["smiles"].astype(str))
    started = time.time()

    def flush(include_solver_reads: bool = False) -> None:
        atomic_csv(all_raw, save_dir / "samples_raw.csv")
        atomic_csv(all_trajectory, save_dir / "trajectory.csv")
        if include_solver_reads:
            atomic_csv(all_solver_reads, save_dir / "solver_reads.csv")
        valid_raw = [row for row in all_raw if bool(row.get("valid"))]
        atomic_csv(valid_raw, save_dir / "generated_molecules.csv")
        unique_global: List[Dict] = []
        seen_smiles = set()
        for row in all_unique_per_reference:
            smiles = row.get("smiles")
            if smiles and smiles not in seen_smiles:
                seen_smiles.add(smiles)
                unique_global.append(row)
        atomic_csv(unique_global, save_dir / "samples_unique.csv")
        atomic_csv(reference_rates, save_dir / "reference_success_rates.csv")
        atomic_csv(table2_rows, save_dir / "table2_generation_stats.csv")

        quality = generation_quality_metrics(all_raw, train_smiles) if all_raw else {}
        meta = {
            "protocol_version": 5,
            "tuning_protocol": "matched_integer_inputs_validity_selection_v1",
            "hardware_adapted_objective": args.lambda_mode == "cim_precision",
            "property": args.prop,
            "solver": args.solver,
            "cim": {
                "project_no": os.environ.get("KAIWU_CIM_PROJECT_NO"),
                "run_id": os.environ.get("KAIWU_CIM_RUN_ID"),
                "bit_width": os.environ.get("KAIWU_CIM_BIT_WIDTH"),
                "task_mode": os.environ.get("KAIWU_CIM_TASK_MODE"),
                "local_solver_fallback": False,
                "precision_policy": ("balanced_structure_exact_v1" if args.lambda_mode == "cim_precision"
                                     else "uniform_scale_round_full_matrix"),
                "hardware_steps": sum(bool(row.get("used_cim")) for row in all_raw),
                "reads_semantics": "platform-controlled pool in optimization mode; see actual counts",
            } if args.solver == "kaiwu_cim" else None,
            "qubo_mode": args.qubo_mode,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "checkpoint_format_version": checkpoint_version,
            "legacy_checkpoint": checkpoint_version < 2,
            "reference_manifest": str(manifest_path),
            "reference_manifest_sha256": manifest_sha256(manifest_path),
            "manifest_reference_count": len(all_references),
            "selected_reference_ids": [ref.reference_id for ref in references],
            "selected_reference_count": len(references),
            "band_indices": band_indices,
            "range_semantics": "lower-inclusive, upper-exclusive",
            "seed_protocol": (
                "condition=base+property*1000000+band*1000+reference_number; "
                "solver_seed=condition_seed+global_step"
            ),
            "structural_qubo": "paper_Eq2_cross_pair_1_same_side_pair_2",
            "invalid_policy": (
                "audit in samples_raw; discard before free-energy evaluation "
                "and surrogate dataset D"
            ),
            "base_seed": args.run_seed,
            "exploration_file_sha256": exploration_hash,
            "params": {
                "max_feasibility_batches": int(os.environ.get("QC_MAX_FEASIBILITY_BATCHES", "1")),
                "adaptive_budget_policy": "Only acquire another batch when all candidates so far are invalid; record actual reads",
                "spin_encoding": os.environ.get("QC_SPIN_ENCODING", "none"),
                "candidate_selection_policy": os.environ.get("QC_CANDIDATE_SELECTION", "energy"),
                "solver_repeats": int(os.environ.get("QC_SOLVER_REPEATS", "1")),
                "both_solvers_use_14bit_integer_matrix": True,
                "n_explore": CFG.n_explore,
                "n_optimize": CFG.n_optimize,
                "reads": CFG.sa_reads,
                "sweeps": CFG.sa_sweeps,
                "valency_penalty": CFG.valency_penalty if args.lambda_mode == "fixed" else None,
                "cim_penalty_ratio": CFG.cim_penalty_ratio if args.lambda_mode == "cim_precision" else None,
                "lambda_mode": CFG.lambda_mode,
                "lambda_requirement": ("lambda = ratio * ||beta||_inf; 1 when beta=0"
                                       if args.lambda_mode == "cim_precision" else "fixed lambda >> ||beta||_inf"),
                "lambda_numeric_source": (
                    "hardware precision calibration; changes relative objective weight"
                    if args.lambda_mode == "cim_precision" else
                    "reproduction choice; paper does not report a value"
                ),
                "penalty_scale": CFG.penalty_scale,
                "compensated_terms_enabled": args.qubo_mode == "compensated",
            },
            "reference_records": [asdict(ref) for ref in references],
            "conditions": condition_meta,
            "counts": {
                "raw": len(all_raw),
                "trajectory": len(all_trajectory),
                "solver_reads": len(all_solver_reads),
                "valid": len(valid_raw),
                "unique_global": len(unique_global),
            },
            "quality": quality,
            "elapsed_seconds": round(time.time() - started, 3),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "rdkit": package_version("rdkit"),
                "kaiwu": package_version("kaiwu"),
                "dwave_samplers": package_version("dwave-samplers"),
                "device": str(CFG.device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        }
        temporary = save_dir / "run_meta.json.tmp"
        temporary.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(temporary, save_dir / "run_meta.json")

    print(
        f"[STRICT] {len(references)}/25 references, "
        f"{len(band_indices)} bands, {CFG.n_explore}+{CFG.n_optimize} steps"
    )
    for band_index in band_indices:
        target_range: Tuple[float, float] = all_bands[band_index]
        lo, hi = target_range
        label = f"{args.prop}=[{lo:.3f},{hi:.3f})"
        band_raw: List[Dict] = []
        band_started = time.time()

        for reference in references:
            condition_seed = stable_condition_seed(
                args.run_seed, args.prop, band_index, reference.reference_id
            )
            CFG.seed = condition_seed
            set_seed(condition_seed)
            key = f"{args.prop}:{band_index}:{reference.reference_id}"
            forced = exploration.get(key)
            result = designer.design(
                ref_smiles=reference.smiles,
                target_prop=args.prop,
                target_range=target_range,
                reference_id=reference.reference_id,
                qubo_mode=args.qubo_mode,
                exploration_bitstrings=forced,
            )
            for collection in (
                result.samples_raw, result.trajectory, result.samples_unique,
                result.solver_reads,
            ):
                for row in collection:
                    row["target_band"] = label
                    row["target_band_index"] = band_index
                    row["solver"] = args.solver
                    row["seed"] = condition_seed
                    row["condition_seed"] = condition_seed
                    row["reference_test_position"] = reference.test_position
                    row["reference_size_stratum"] = reference.size_stratum

            all_raw.extend(result.samples_raw)
            band_raw.extend(result.samples_raw)
            all_trajectory.extend(result.trajectory)
            all_solver_reads.extend(result.solver_reads)
            all_unique_per_reference.extend(result.samples_unique)
            ref_stats = summarise_generation(
                result.samples_raw, args.prop, target_range, train_df
            )
            trajectory_summary = (
                result.trajectory[-1] if result.trajectory else {
                    "best_free_energy": None,
                    "auc_best_free_energy_normalized_time": None,
                    "auc_normalized_best_fe_regret": None,
                }
            )
            reference_rates.append({
                "reference_id": reference.reference_id,
                "target_band": label,
                "target_band_index": band_index,
                "condition_seed": condition_seed,
                "n_heavy_atoms": reference.n_heavy_atoms,
                "n_qubo_vars": reference.n_qubo_vars,
                "best_free_energy": trajectory_summary["best_free_energy"],
                "auc_best_free_energy_normalized_time": trajectory_summary[
                    "auc_best_free_energy_normalized_time"
                ],
                "auc_normalized_best_fe_regret": trajectory_summary[
                    "auc_normalized_best_fe_regret"
                ],
                "solver_wall_seconds": sum(
                    float(row["solve_seconds"]) for row in result.trajectory
                ),
                **ref_stats,
            })
            condition_meta.append({
                "reference_id": reference.reference_id,
                "target_band_index": band_index,
                "seed": condition_seed,
                "selected_candidates": len(result.samples_raw),
                "solver_reads": len(result.solver_reads),
                "trajectory_steps": len(result.trajectory),
                "valid_unique_per_reference": len(result.samples_unique),
            })
            # Persist the full read pool after every completed condition.
            # This keeps long SQA shards crash-recoverable for analysis.
            flush(include_solver_reads=True)

        band_stats = summarise_generation(
            band_raw, args.prop, target_range, train_df
        )
        table2_rows.append({
            "target_band": label,
            "target_band_index": band_index,
            **band_stats,
            "ring_pct_valid": 100.0 * sum(
                _count_rings(str(row.get("smiles", ""))) > 0
                for row in band_raw if row.get("valid")
            ) / max(sum(bool(row.get("valid")) for row in band_raw), 1),
            "seconds": round(time.time() - band_started, 3),
        })
        flush(include_solver_reads=True)
        print(
            f"[BAND] {label}: sampled={band_stats['n_sampled']} "
            f"valid={band_stats['n_valid']} hit={band_stats['n_in_range']} "
            f"success/all={band_stats['success_all_pct']:.2f}%"
        )

    flush(include_solver_reads=True)
    print(
        f"[DONE] raw={len(all_raw)} trajectory={len(all_trajectory)} "
        f"outputs={save_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
