"""Continuous-time PIMC SQA on the same 14-bit Ising encoding as CIM."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from qc_mol_design.config import CFG
from .spin_encoding import encode_prepared, physical_matrix, logical_spins
from .qubo import _decode_cim_pool

LAST_CALL = {}
_BUDGETS = {}
_SAMPLER = None


def energy_model(matrix):
    import dimod
    matrix = np.asarray(matrix)
    if not (matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
            and np.array_equal(matrix, matrix.T)
            and np.all(np.diag(matrix) == 0)
            and np.all(matrix == np.rint(matrix)) and abs(matrix).max() <= 8191):
        raise ValueError("Expected the symmetric zero-diagonal 14-bit CIM matrix")
    h = {i: 0.0 for i in range(len(matrix))}
    J = {(i, j): -2.0 * float(matrix[i, j])
         for i in range(len(matrix)) for j in range(i + 1, len(matrix)) if matrix[i, j]}
    # dimod sums each pair once; Kaiwu uses -s.T @ matrix @ s.
    return dimod.BinaryQuadraticModel.from_ising(h, J)


def annealing_fields(bqm, sweeps):
    from dwave.samplers.sa.sampler import _default_ising_beta_range
    if sweeps < 2:
        raise ValueError("At least two sweeps required for the SQA schedule")
    beta_range = _default_ising_beta_range(bqm.linear, bqm.quadratic)
    hp = np.geomspace(*beta_range, sweeps)
    hd = np.geomspace(2.0, 0.001, sweeps)
    hd[-1] = 0.0
    return hp, hd


def sample_physical(matrix, reads, sweeps, seed):
    global _SAMPLER
    from dwave.samplers import PathIntegralAnnealingSampler
    if _SAMPLER is None:
        _SAMPLER = PathIntegralAnnealingSampler()
    bqm = energy_model(matrix)
    hp, hd = annealing_fields(bqm, sweeps)
    response = _SAMPLER.sample(
        bqm, num_reads=reads, num_sweeps=sweeps, num_sweeps_per_beta=1,
        beta_schedule_type="custom", Hp_field=hp, Hd_field=hd, Gamma=1.0,
        project_states=(True, True), seed=int(seed))
    order = [list(response.variables).index(i) for i in range(len(matrix))]
    raw = np.asarray(response.record.sample[:, order], dtype=np.int8)
    energies = -np.einsum("bi,ij,bj->b", raw.astype(float), matrix, raw.astype(float))
    np.testing.assert_allclose(energies, response.record.energy, rtol=1e-12, atol=1e-6)
    spins = np.repeat(raw, response.record.num_occurrences.astype(int), axis=0)
    if spins.shape != (reads, len(matrix)) or not np.isin(spins, (-1, 1)).all():
        raise RuntimeError("SQA returned an invalid pool")
    return spins, hp, hd


def solve_budgeted(Q, num_reads, num_sweeps, cim_prepared):
    global LAST_CALL
    if os.environ.get("QC_QUBO_SOLVER") != "sqa":
        raise RuntimeError("This adapter only executes SQA")
    if num_reads != 10 or num_sweeps != 1000 or cim_prepared is None:
        raise ValueError("Frozen SQA configuration requires 10 reads and 1000 sweeps")
    if os.environ.get("QC_SPIN_ENCODING") != "gauge":
        raise ValueError("CIM-matched SQA requires gauge encoding")
    path = Path(os.environ["QC_SQA_BUDGET_FILE"])
    if str(path) not in _BUDGETS:
        _BUDGETS[str(path)] = json.loads(path.read_text())
    budget = _BUDGETS[str(path)]
    step = int(os.environ["QC_SQA_STEP"])
    record = budget["steps"][step]
    if record["global_step"] != step or record["solver_seed"] != CFG.seed:
        raise ValueError("Budget condition/step/seed mismatch")
    repeats, batches = record["solver_repeats"], record["feasibility_batches"]
    if repeats not in (1, 5) or batches not in (1, 2) or record["candidate_reads"] != repeats * batches * 10:
        raise ValueError("Invalid frozen candidate budget")
    root = Path(os.environ["QC_SQA_CACHE_DIR"])
    call_root = root / "calls"
    call_root.mkdir(parents=True, exist_ok=True)
    samples, call_records = [], []
    for batch in range(batches):
        for repeat in range(repeats):
            seed = int(CFG.seed + batch * 1000000 + repeat * 100000)
            prepared = encode_prepared(cim_prepared, "gauge", seed)
            matrix = physical_matrix(prepared["ising"], prepared)
            cache = call_root / f"step{step:03d}_batch{batch}_repeat{repeat}.npz"
            metadata = cache.with_suffix(".json")
            started = time.monotonic()
            cached = cache.exists() and metadata.exists()
            if cached:
                saved_meta = json.loads(metadata.read_text())
                if saved_meta["seed"] != seed or saved_meta["sweeps"] != num_sweeps:
                    raise RuntimeError("Cached SQA parameters changed")
                with np.load(cache, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["qubo"], Q)
                    np.testing.assert_array_equal(saved["physical_ising"], matrix)
                    spins = saved["spins"].copy()
            else:
                spins, hp, hd = sample_physical(matrix, num_reads, num_sweeps, seed)
                temporary = cache.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, qubo=Q, physical_ising=matrix, spins=spins,
                                        Hp_field=hp, Hd_field=hd,
                                        spin_gauge=prepared["spin_gauge"],
                                        spin_permutation=prepared["spin_permutation"])
                temporary.replace(cache)
                saved_meta = {
                    "solver": "dwave.samplers.PathIntegralAnnealingSampler",
                    "used_cim": False, "transverse_field_enabled": True,
                    "path_integral": "continuous_time",
                    "seed": seed, "reads": num_reads, "sweeps": num_sweeps,
                    "Gamma": 1.0, "Hp_start": float(hp[0]), "Hp_end": float(hp[-1]),
                    "Hd_start": float(hd[0]), "Hd_end": float(hd[-1]),
                    "Hd_before_last": float(hd[-2]),
                    "physical_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                    "seconds": time.monotonic() - started,
                }
                temporary = metadata.with_suffix(".tmp")
                temporary.write_text(json.dumps(saved_meta, indent=2) + "\n")
                temporary.replace(metadata)
            pool = _decode_cim_pool(logical_spins(spins, prepared), Q, len(Q), len(Q) + 1)
            if len(pool) != num_reads:
                raise RuntimeError("SQA candidate count mismatch")
            samples.extend(pool)
            call_records.append({"file": cache.name, "seed": seed, "cache_reused": cached})
    if len(samples) != record["candidate_reads"]:
        raise RuntimeError("CIM-matched candidate budget violated")
    LAST_CALL = {
        "used_cim": False, "used_sqa": True, "sqa_calls": len(call_records),
        "sqa_transverse_field_enabled": True, "sqa_path_integral": "continuous_time",
        "sqa_num_sweeps": num_sweeps, "sqa_cache_reused": all(r["cache_reused"] for r in call_records),
        "solver_repeats": repeats, "feasibility_batches": batches,
        "sampling_regime": record["sampling_regime"],
        "candidate_budget_source": "frozen_observed_cim_budget",
        "history_replayed": False, "cim_hardware_calls": 0,
        "cim_precision_policy": cim_prepared["metadata"]["precision_policy"],
        "cim_structural_integer_exact": cim_prepared["metadata"]["structural_integer_exact"],
        "cim_penalty_ratio": cim_prepared["metadata"]["penalty_ratio"],
    }
    with (root / "step_calls.jsonl").open("a") as handle:
        handle.write(json.dumps({"step": step, "calls": call_records,
                                 "candidate_reads": len(samples)}) + "\n")
    return samples
