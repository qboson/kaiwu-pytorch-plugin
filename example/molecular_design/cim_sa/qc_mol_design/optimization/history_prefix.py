"""Read-only replay of frozen hardware pools; never calls a solver."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import cim_hardware
from .qubo import _decode_cim_pool
from .spin_encoding import encode_prepared, logical_spins, physical_matrix


@lru_cache(maxsize=1)
def history():
    path = os.environ.get("QC_HISTORY_PREFIX_FILE")
    return json.loads(Path(path).read_text()) if path else None


def replay(Q, prepared, context, seed, encoding):
    saved = history()
    if saved is None:
        return None
    step = int(context.rsplit("_", 1)[1])
    if step >= saved["prefix_steps"]:
        return None
    row = saved["rows"][step]
    if row["global_step"] != step or row["solver_seed"] != seed:
        raise RuntimeError("Historical step or seed mismatch")
    samples, metadata = [], []
    for index, task in enumerate(row["cim_task_names"]):
        record = saved["calls"][task]
        directory = Path(os.environ["QC_HISTORY_PROJECT_ROOT"]) / record["path"]
        meta_bytes = (directory / "metadata.json").read_bytes()
        raw_bytes = (directory / "spins.npy").read_bytes()
        if hashlib.sha256(meta_bytes).hexdigest() != record["metadata_sha256"] or hashlib.sha256(raw_bytes).hexdigest() != record["spins_sha256"]:
            raise RuntimeError("Historical hardware cache changed")
        meta = json.loads(meta_bytes)
        batch, repeat = divmod(index, 5)
        suffix = f"_rep{repeat}" if batch == 0 else f"_batch{batch}_rep{repeat}"
        if meta["context"] != context + suffix:
            raise RuntimeError("Historical call order mismatch")
        encoded = encode_prepared(prepared, encoding, seed + batch * 1000000 + repeat * 100000)
        with np.load(directory / "input.npz", allow_pickle=False) as old:
            np.testing.assert_array_equal(old["qubo"], Q)
            np.testing.assert_array_equal(old["ising"], physical_matrix(encoded["ising"], encoded))
        raw = np.load(directory / "spins.npy", allow_pickle=False)
        pool = _decode_cim_pool(logical_spins(raw, encoded), Q, len(Q), len(encoded["ising"]))
        if len(pool) != 10:
            raise RuntimeError("Historical pool size changed")
        samples.extend(pool)
        metadata.append(meta)
    if len(samples) != row["solver_returned_reads"]:
        raise RuntimeError("Historical budget mismatch")
    cim_hardware.LAST_CALL = {**metadata[0], "task_names": row["cim_task_names"],
        "hardware_calls": len(metadata), "returned_reads": len(samples), "cache_reused": True,
        "solver_repeats": 5, "feasibility_batches": len(metadata) // 5,
        "history_replayed": True, "sampling_regime": "historical_5_calls"}
    return samples


def validate_selected(row):
    saved = history()
    if saved is None or row["global_step"] >= saved["prefix_steps"]:
        return
    expected = saved["rows"][row["global_step"]]
    for key in ("A_bitstring", "valid", "target_hit", "solver_seed", "solver_returned_reads"):
        if row[key] != expected[key]:
            raise RuntimeError("Historical selection changed: " + key)
    if expected.get("free_energy") is not None and not np.isclose(row["free_energy"], expected["free_energy"], rtol=1e-7, atol=1e-7):
        raise RuntimeError("Historical model feedback changed")
