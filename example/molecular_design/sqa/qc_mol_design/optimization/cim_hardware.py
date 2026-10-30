"""Audited, resumable full-matrix CIM execution. Never substitutes a local solver."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from .spin_encoding import physical_matrix, logical_spins

LAST_CALL = {}
_LICENSE_READY = False


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def initialize_license():
    """Keep this project's license separate from other experiments' licenses."""
    global _LICENSE_READY
    import kaiwu as kw
    if _LICENSE_READY:
        return kw
    private = Path(os.environ["KAIWU_CREDENTIALS_FILE"])
    credentials = json.loads(private.read_text())
    import kaiwu.license._license_utils as license_utils
    import kaiwu.cim._optimizer_adapter as adapter
    license_path = private.with_name("cim_license.lic")
    license_utils.LICENSE_FILE_PATH = str(license_path)
    adapter.LICENSE_FILE_PATH = str(license_path)
    try:
        payload, code = license_utils._read_license()
        matches = (str(payload["user_id"]) == str(credentials["user_id"])
                   and code == credentials["sdk_code"])
    except (ValueError, OSError):
        matches = False
    if not matches:
        kw.license.init(str(credentials["user_id"]), credentials["sdk_code"])
    license_path.chmod(0o600)
    _LICENSE_READY = True
    return kw


def quantize_ising(matrix, bit_width):
    """Uniform positive rescaling and rounding; record the approximation."""
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Ising matrix must be square")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
        raise ValueError("Ising matrix must be finite and symmetric")
    np.fill_diagonal(matrix, 0)
    if not 2 <= bit_width <= 16:
        raise ValueError("unsupported coefficient bit width")
    limit = 2 ** (bit_width - 1) - 1
    magnitude = float(np.abs(matrix).max())
    scale = limit / magnitude if magnitude else 1.0
    quantized = np.rint(matrix * scale).astype(np.int64)
    error = quantized / scale - matrix
    nonzero = matrix != 0
    stats = {
        "bit_width": bit_width, "scale": scale,
        "ising_max_abs": magnitude,
        "quantization_max_abs_error": float(np.abs(error).max()),
        "quantization_relative_frobenius_error": float(
            np.linalg.norm(error) / max(float(np.linalg.norm(matrix)), 1e-300)),
        "nonzero_coefficients": int(nonzero.sum()),
        "coefficients_rounded_to_zero": int((nonzero & (quantized == 0)).sum()),
        "precision_policy": "uniform_scale_round_full_matrix",
    }
    return quantized, stats


def solve_hardware(Q, num_reads=50, prepared=None, context_override=None, return_metadata=False,
                   namespace_override=None, root_override=None):
    """Persist input, task identity and actual hardware spin pool at every step."""
    global LAST_CALL
    LAST_CALL = {}
    kw = initialize_license()
    project = os.environ["KAIWU_CIM_PROJECT_NO"]
    namespace = namespace_override or os.environ["KAIWU_CIM_RUN_ID"]
    context = context_override or os.environ["KAIWU_CIM_TASK_CONTEXT"]
    mode = os.environ.get("KAIWU_CIM_TASK_MODE", "quota")
    bits = int(os.environ.get("KAIWU_CIM_BIT_WIDTH", "14"))
    if mode not in ("sample", "quota"):
        raise ValueError("CIM task mode must be sample or quota")
    if mode == "sample" and not 10 <= num_reads <= 2000:
        raise ValueError("CIM sample count must be between 10 and 2000")
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or not np.isfinite(Q).all():
        raise ValueError("QUBO must be finite and square")
    converter = getattr(kw.qubo, "qubo_matrix_to_ising_matrix", None)
    if converter is None:
        converter = kw.core.qubo_matrix_to_ising_matrix
    ising, offset = converter(Q)
    if prepared is None:
        submitted, precision = quantize_ising(ising, bits)
    else:
        submitted = np.asarray(prepared["ising"], dtype=np.int64)
        precision = dict(prepared["metadata"])
        if (submitted.shape != ising.shape or not np.array_equal(submitted, submitted.T)
                or abs(submitted).max() > 2 ** (bits - 1) - 1
                or np.any(submitted.diagonal()) or precision["bit_width"] != bits):
            raise ValueError("Invalid prepared CIM matrix")
        expected_error = submitted / precision["scale"] - ising
        if not np.isclose(abs(expected_error).max(), precision["quantization_max_abs_error"], atol=1e-9):
            raise ValueError("Prepared matrix does not match the supplied QUBO")
    logical_submitted = submitted.copy()
    if prepared is not None:
        submitted = physical_matrix(logical_submitted, prepared)
    matrix_sha = hashlib.sha256(Q.tobytes()).hexdigest()
    identity = json.dumps([project, namespace, context, mode, num_reads,
                           bits, matrix_sha], separators=(",", ":"))
    if prepared is not None:
        identity += json.dumps([precision["precision_policy"],
                               hashlib.sha256(submitted.tobytes()).hexdigest()])
    task_name = "qcm_" + hashlib.sha256(identity.encode()).hexdigest()[:32]
    root = Path(root_override or os.environ["KAIWU_CIM_CACHE_DIR"])
    call_dir = root / "calls" / task_name
    call_dir.mkdir(parents=True, exist_ok=True)
    sdk_dir = root / "sdk"
    sdk_dir.mkdir(parents=True, exist_ok=True)
    kw.common.CheckpointManager.save_dir = str(sdk_dir)
    meta_path = call_dir / "metadata.json"
    raw_path = call_dir / "spins.npy"
    metadata = {
        "task_name": task_name, "project_no": project, "run_id": namespace,
        "context": context, "qubo_sha256": matrix_sha,
        "ising_sha256": hashlib.sha256(submitted.tobytes()).hexdigest(),
        "qubo_variables": len(Q), "ising_spins": len(submitted),
        "task_mode": mode, "requested_reads": num_reads,
        "ising_offset": float(offset), "used_cim": False,
        "local_solver_fallback": False, **precision,
    }
    started = time.time()
    if meta_path.exists():
        saved = json.loads(meta_path.read_text())
        if saved["qubo_sha256"] != matrix_sha or saved["context"] != context:
            raise RuntimeError("CIM cache identity mismatch")
        metadata.update(saved)
    else:
        metadata.update(status="prepared", created_at=time.time())
        extra = ({"structural_integer": prepared["structural_integer"],
                  "guidance_before_rounding": prepared["guidance_before_rounding"],
                  "logical_ising": logical_submitted,
                  "spin_permutation": prepared.get("spin_permutation", np.arange(len(submitted))),
                  "spin_gauge": prepared.get("spin_gauge", np.ones(len(submitted), dtype=int))}
                 if prepared is not None else {})
        np.savez_compressed(call_dir / "input.npz", qubo=Q, ising=submitted, **extra)
        atomic_json(meta_path, metadata)

    def record(event, **fields):
        metadata.update(status=event, updated_at=time.time(), **fields)
        atomic_json(meta_path, metadata)
        with (root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"task_name": task_name, "context": context,
                                     "event": event, "time": time.time(), **fields}) + "\n")

    class ObservedCIM(kw.cim.CIMOptimizer):
        def _on_matrix_change(self):
            pass

        def _create_task(self, *args, **kwargs):
            record("submitting")
            result = super()._create_task(*args, **kwargs)
            record("submitted")
            return result

        def _handle_task_status(self, task_data, name):
            public = {k: v for k, v in task_data.items()
                      if k not in {"result", "authorization", "sdk_code"}
                      and isinstance(v, (str, int, float, bool, type(None)))
                      and "http" not in str(v)}
            record("cloud_" + str(task_data.get("code")), cloud_status=public)
            return super()._handle_task_status(task_data, name)

    try:
        if raw_path.exists():
            raw = np.load(raw_path, allow_pickle=False)
            cached = True
        else:
            optimizer = ObservedCIM(
                task_name=task_name, project_no=project, task_mode=mode,
                sample_number=num_reads, sample_sort_mode=2,
                wait=False, interval=10,
            )
            deadline = time.monotonic() + float(os.environ.get("KAIWU_CIM_TIMEOUT", "86400"))
            network_failures = 0
            while True:
                try:
                    raw = optimizer.solve(submitted.astype(np.float64))
                    network_failures = 0
                except Exception as error:
                    transient = any(token in str(error) for token in (
                        "Timeout", "timed out", "Connection", "Network error",
                        "429", "502", "503", "504", "Too Many Requests"))
                    if not transient or network_failures >= 8 or time.monotonic() >= deadline:
                        raise
                    network_failures += 1
                    record("network_retry", retry=network_failures,
                           error_type=type(error).__name__)
                    time.sleep(min(60, 2 ** network_failures + 3))
                    continue
                if raw is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("CIM task still pending; resume with the same task identity")
                time.sleep(10)
            raw = np.asarray(raw)
            cached = False
        from qc_mol_design.optimization.qubo import _decode_cim_pool
        canonical_raw = logical_spins(raw, prepared) if prepared is not None else raw
        decoded = _decode_cim_pool(canonical_raw, Q, len(Q), len(submitted))
        if not cached:
            temporary = raw_path.with_suffix(".tmp.npy")
            np.save(temporary, raw, allow_pickle=False)
            temporary.replace(raw_path)
        record("completed", used_cim=True, returned_reads=len(decoded),
               wall_seconds=time.time() - started)
        call_result = {**metadata, "cache_reused": cached}
        LAST_CALL = call_result
        print(f"[CIM] {context} task={task_name} reads={len(decoded)} "
              f"cached={cached} seconds={time.time()-started:.1f}", flush=True)
        return (decoded, call_result) if return_metadata else decoded
    except Exception as error:
        # Do not serialize exception text: upload errors can contain signed URLs.
        if "资源不足" in str(error):
            failure = "insufficient_cloud_resources"
        elif "Verification failed" in str(error):
            failure = "cloud_validation_failed"
        else:
            failure = "solver_error"
        record("failed", error_type=type(error).__name__, failure_reason=failure)
        raise
