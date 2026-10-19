"""Matched SA/CIM candidate budgets and audited reuse of exploration pools."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

from qc_mol_design.config import CFG
from . import cim_hardware
from .qubo import solve_qubo, _decode_cim_pool
from .spin_encoding import encode_prepared, physical_matrix


def prioritize_valid(evaluated, valid, policy):
    """Stable energy order inside each validity tier; never uses target hits."""
    if policy == "energy":
        return evaluated
    if policy != "valid_energy":
        raise ValueError("Unknown candidate selection policy")
    classified = [(bool(valid(item)), item) for item in evaluated]
    return [item for is_valid, item in classified if is_valid] + [
        item for is_valid, item in classified if not is_valid]


def reuse_exploration(Q, prepared, context):
    root = os.environ.get("QC_EXPLORATION_REUSE_DIR")
    if not root or int(context.rsplit("_", 1)[1]) >= 10:
        return None
    matches = []
    for path in (Path(root) / "calls").glob("*/metadata.json"):
        meta = json.loads(path.read_text())
        if meta.get("used_cim") and meta["context"] == context:
            matches.append((path, meta))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one recorded exploration hardware pool")
    path, meta = matches[0]
    saved = np.load(path.with_name("input.npz"))
    np.testing.assert_array_equal(saved["qubo"], Q)
    np.testing.assert_array_equal(saved["ising"], prepared["ising"])
    raw = np.load(path.with_name("spins.npy"), allow_pickle=False)
    samples = _decode_cim_pool(raw, Q, len(Q), len(Q)+1)
    meta = {**meta, "cache_reused": True, "reused_from": str(path)}
    return samples, meta


def previous_batch(Q, prepared, context, repeats, encoding, seed):
    """Resume a whole matching batch under its original IDs, including pending calls."""
    root = os.environ.get("QC_FIRST_BATCH_REUSE_DIR")
    if not root:
        return None
    records = {}
    for path in (Path(root) / "calls").glob("*/metadata.json"):
        meta = json.loads(path.read_text())
        records[meta["context"]] = (path, meta)
    namespaces = set()
    for repeat in range(repeats):
        key = context + f"_rep{repeat}"
        if key not in records:
            return None
        path, meta = records[key]
        old = np.load(path.with_name("input.npz"))
        encoded = encode_prepared(prepared, encoding, seed + repeat * 100000)
        if not (np.array_equal(old["qubo"], Q) and
                np.array_equal(old["ising"], physical_matrix(encoded["ising"], encoded))):
            return None
        namespaces.add(meta["run_id"])
    if len(namespaces) != 1:
        raise RuntimeError("Mixed run IDs in reusable batch")
    return root, namespaces.pop()


def solve_tuning(Q, num_reads=10, num_sweeps=1000, initial_states=None, cim_prepared=None,
                 valid_sample=None):
    backend = os.environ["QC_QUBO_SOLVER"]
    if backend == "sqa":
        from .sqa_solver import solve_budgeted
        return solve_budgeted(Q, num_reads, num_sweeps, cim_prepared)
    repeats = int(os.environ.get("QC_SOLVER_REPEATS", "1"))
    encoding = os.environ.get("QC_SPIN_ENCODING", "none")
    max_batches = int(os.environ.get("QC_MAX_FEASIBILITY_BATCHES", "1"))
    if backend not in ("kaiwu", "kaiwu_cim") or repeats not in (1, 3, 5):
        raise ValueError("Tuning supports SA/CIM with 1, 3 or 5 repeats")
    if num_reads != 10 or cim_prepared is None:
        raise ValueError("Both arms require 10 reads per repeat and the same integer encoding")
    if not 1 <= max_batches <= 3 or (max_batches > 1 and valid_sample is None):
        raise ValueError("Adaptive feasibility sampling requires a validator and at most three batches")
    if backend == "kaiwu":
        original_seed = CFG.seed
        samples = []
        try:
            for batch in range(max_batches):
                for repeat in range(repeats):
                    CFG.seed = original_seed + batch * 1000000 + repeat * 100000
                    pool = solve_qubo(Q, num_reads, num_sweeps,
                                      initial_states=initial_states,
                                      cim_prepared=encode_prepared(cim_prepared, encoding, CFG.seed))
                    if len(pool) != num_reads:
                        raise RuntimeError("SA candidate budget mismatch")
                    samples.extend(pool)
                if max_batches == 1 or any(valid_sample(sample) for sample in samples):
                    break
        finally:
            CFG.seed = original_seed
        return samples
    context = os.environ["KAIWU_CIM_TASK_CONTEXT"]
    if os.environ.get("QC_HISTORY_PREFIX_FILE"):
        if repeats != 1 or max_batches != 1:
            raise RuntimeError("Continuation requires exactly one call and no extra sampling")
        from .history_prefix import replay
        historical = replay(Q, cim_prepared, context, CFG.seed, encoding)
        if historical is not None:
            return historical
    cim_hardware.initialize_license()
    samples, calls = [], []
    original_seed = CFG.seed
    for batch in range(max_batches):
        prior = previous_batch(Q, cim_prepared, context, repeats, encoding, original_seed) if batch == 0 else None
        cached = (reuse_exploration(Q, cim_prepared, context)
                  if batch == 0 and repeats == 1 and encoding == "none" else None)
        if cached is not None:
            results = [cached]
        else:
            def run(repeat):
                options = {"root_override": prior[0], "namespace_override": prior[1]} if prior else {}
                suffix = f"_rep{repeat}" if batch == 0 else f"_batch{batch}_rep{repeat}"
                pool, meta = cim_hardware.solve_hardware(
                    Q, num_reads=num_reads,
                    prepared=encode_prepared(cim_prepared, encoding, original_seed + batch * 1000000 + repeat * 100000),
                    context_override=context + suffix, return_metadata=True, **options)
                if prior:
                    meta = {**meta, "reused_batch_from": prior[0]}
                return pool, meta
            with ThreadPoolExecutor(max_workers=repeats) as executor:
                results = list(executor.map(run, range(repeats)))
        if any(len(pool) != num_reads for pool, _ in results):
            raise RuntimeError("CIM candidate budget mismatch")
        samples.extend(sample for pool, _ in results for sample in pool)
        calls.extend(meta for _, meta in results)
        if max_batches == 1 or any(valid_sample(sample) for sample in samples):
            break
    cim_hardware.LAST_CALL = {
        **calls[0], "task_names": [m["task_name"] for m in calls],
        "hardware_calls": len(calls), "returned_reads": len(samples),
        "cache_reused": all(m["cache_reused"] for m in calls),
        "solver_repeats": repeats, "feasibility_batches": len(calls) // repeats,
        "history_replayed": False, "sampling_regime": "single_call",
    }
    cache_dir = Path(os.environ["KAIWU_CIM_CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "step_calls.jsonl").open("a") as handle:
        handle.write(json.dumps({"context": context, "calls": calls}) + "\n")
    return samples
