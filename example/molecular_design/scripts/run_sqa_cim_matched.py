#!/usr/bin/env python3
"""Run SQA trajectories under the frozen, observed CIM candidate budget."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sqa.json"
BUDGETS = ROOT / "configs/budgets"
SOURCE = ROOT / "sqa"
OUT = ROOT / "results/new_sqa"
DATA = None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def worker(condition, scope, n_opt):
    dest = OUT / scope / "conditions" / condition
    dest.mkdir(parents=True, exist_ok=True)
    prop, band, ref, seed = condition.split("_")
    for key in list(os.environ):
        if key.startswith(("KAIWU_", "QC_HISTORY_", "QC_CIM_", "QC_EXPLORATION_", "QC_FIRST_BATCH_")):
            del os.environ[key]
    os.environ.update(QC_QUBO_SOLVER="sqa", QC_SOLVER_REPEATS="1", QC_MAX_FEASIBILITY_BATCHES="1",
                      QC_CANDIDATE_SELECTION="valid_energy", QC_SPIN_ENCODING="gauge",
                      QC_SQA_BUDGET_FILE=str(BUDGETS / (condition + ".json")),
                      QC_SQA_CACHE_DIR=str(dest / "sqa"))
    from qc_mol_design.config import CFG
    import qc_mol_design.data as data_module
    CFG.seed = 42
    data_module.load_zinc12k = lambda *a, **k: DATA.copy(deep=False)
    sys.argv = [str(SOURCE / "run_generation_strict.py"), "--prop", prop, "--solver", "sqa",
        "--checkpoint", str(ROOT / "checkpoints/checkpoint_qc.pt"),
        "--reference-manifest", str(ROOT / "references_25.json"), "--reference-ids", ref,
        "--bands", band[1:], "--qubo-mode", "strict", "--lambda-mode", "cim_precision",
        "--cim-penalty-ratio", "4", "--n-explore", "10", "--n-optimize", str(n_opt),
        "--reads", "10", "--sweeps", "1000", "--run-seed", "42", "--save-dir", str(dest)]
    with (dest / "run.log").open("a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
            return {"condition": condition, "ok": True}
        except BaseException:
            traceback.print_exc()
            return {"condition": condition, "ok": False}


def main():
    global DATA, OUT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", choices=("pilot", "full"), default="full")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--detach", action="store_true")
    p.add_argument("--output-root", type=Path, default=OUT)
    args = p.parse_args()
    OUT = args.output_root.resolve()
    if not 1 <= args.workers <= 160:
        raise ValueError("workers must be 1..160")
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    base = OUT / args.scope
    base.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--scope", args.scope,
                   "--workers", str(args.workers), "--output-root", str(OUT)]
        with (base / "controller.log").open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": "42"},
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({"pid": child.pid, "scope": args.scope, "output": str(base)}))
        return
    lock = (base / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((CONFIG).read_text())
    conditions = config["conditions"]
    n_opt = 90
    if args.scope == "pilot":
        conditions = ["qed_b0_R00_s42", "logp_b4_R24_s42"]
        n_opt = 2
    paths = sorted(SOURCE.rglob("*.py")) + sorted((BUDGETS).glob("*.json"))
    paths += [CONFIG, Path(__file__).resolve()]
    packages = {name: importlib.metadata.version(name)
                for name in ("dwave-samplers", "dimod", "numpy", "torch", "rdkit")}
    inputs = json.loads((ROOT / "inputs.json").read_text())
    for item in ("dataset", "checkpoint", "references"):
        path = ROOT / inputs[item]["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}; see README.md")
        if sha(path) != inputs[item]["sha256"]:
            raise ValueError(f"Input checksum differs from the recorded experiment: {path}")
    manifest = {"config": config, "scope": args.scope, "n_explore": 10, "n_optimize": n_opt,
                "conditions": conditions, "checkpoint_sha256": sha(ROOT / config["checkpoint"]),
                "references_sha256": sha(ROOT / config["reference_manifest"]),
                "dataset_sha256": sha(ROOT / "zinc250k.csv"),
                "files_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
                "data_loading": "Validate once in parent, fork read-only frame; same loader and split per child",
                "no_cim_calls": True, "no_sa_execution": True, "package_versions": packages}
    if (base / "manifest.json").exists() and json.loads((base / "manifest.json").read_text()) != manifest:
        raise RuntimeError("Frozen SQA inputs or source changed; refuse mixed results")
    save(base / "manifest.json", manifest)
    steps = 10 + n_opt
    done, pending = [], []
    for condition in conditions:
        meta_path = base / "conditions" / condition / "run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            budget = json.loads((BUDGETS / (condition + ".json")).read_text())
            expected_reads = sum(r["candidate_reads"] for r in budget["steps"][:steps])
            if (meta["counts"]["raw"] == steps and meta["counts"]["solver_reads"] == expected_reads
                    and meta["checkpoint_sha256"] == manifest["checkpoint_sha256"]
                    and meta["protocol_version"] == 6):
                done.append(condition)
                continue
        pending.append(condition)
    # Schedule large molecules first to reduce a long tail.
    pending.sort(key=lambda c: (-int(c.split("_")[2][1:]), c))
    if not pending:
        save(base / "progress.json", {"status": "complete", "completed_conditions": len(done),
                                     "completed_steps": len(done) * steps, "total_conditions": len(conditions)})
        return
    sys.path.insert(0, str(SOURCE))
    import torch
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from qc_mol_design.data import load_zinc12k
    save(base / "progress.json", {"status": "loading_inputs", "pid": os.getpid(),
                                 "completed_conditions": len(done), "total_conditions": len(conditions)})
    DATA = load_zinc12k(path=str(ROOT / "zinc250k.csv"))
    started = time.time()
    failures, active = [], {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(pending)),
                             mp_context=multiprocessing.get_context("fork")) as pool:
        active = {pool.submit(worker, c, args.scope, n_opt): c for c in pending}
        while active:
            finished, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in finished:
                condition = active.pop(future)
                try:
                    result = future.result()
                except BaseException as error:
                    result = {"condition": condition, "ok": False, "error": repr(error)}
                if result["ok"]:
                    done.append(condition)
                else:
                    failures.append(result)
                print(json.dumps(result), flush=True)
            observed = 0
            for path in (base / "conditions").glob("*/sqa/selected_steps.jsonl"):
                try:
                    observed += len({json.loads(line)["global_step"] for line in path.read_text().splitlines()
                                     if line.endswith("}")})
                except (OSError, json.JSONDecodeError):
                    pass
            save(base / "progress.json", {"status": "running" if active else ("failed" if failures else "complete"),
                "pid": os.getpid(), "workers": args.workers, "total_conditions": len(conditions),
                "completed_conditions": len(done), "active_or_queued_conditions": len(active),
                "completed_steps": observed, "planned_steps": len(conditions) * steps,
                "failures": failures, "elapsed_seconds": time.time() - started, "updated_at": time.time(),
                "no_cim_calls": True})
    if failures:
        raise RuntimeError("Some SQA conditions failed: " + str(len(failures)))
    print("SQA_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
