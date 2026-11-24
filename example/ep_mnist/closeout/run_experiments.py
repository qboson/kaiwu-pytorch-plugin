"""Reproduce the archived MNIST experiment matrix in fresh output directories."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ENTRY = HERE.parent / "train_mnist.py"


def job_list(data_dir, only="all"):
    """Return the historical tasks with explicit data and model settings."""
    common = [str(ENTRY), "--data-dir", str(data_dir), "--split-seed", "0",
              "--seed", "0", "--n-val", "5000", "--batch-size", "20",
              "--lr", "0.05", "--hidden", "128", "--input-scale", "2"]
    jobs = []
    if only in ("all", "bp"):
        jobs.append(("bp", [str(HERE / "bp_mnist.py"), str(data_dir)]))
    if only == "all":
        jobs.append(("probe", common + ["--limit-train", "200", "--limit-eval", "100",
                                        "--epochs", "1", "--beta", "2",
                                        "--sa-iterations", "50", "--validation-only"]))
    if only in ("all", "ep-full"):
        jobs.append(("ep-full", common + ["--epochs", "3", "--beta", "2",
                                          "--sa-iterations", "50"]))
    if only in ("all", "sweep"):
        for beta, budget in ((2, 50), (1, 50), (4, 50), (2, 20), (2, 100)):
            for seed in range(3):
                options = common.copy()
                options[options.index("--seed") + 1] = str(seed)
                jobs.append((f"beta-{beta}_sa-{budget}_seed-{seed}", options + [
                    "--limit-train", "2000", "--limit-eval", "1000", "--epochs", "3",
                    "--validation-only", "--beta", str(beta), "--sa-iterations", str(budget)]))
    return jobs


def command_for(name, options, output):
    """Build a child command without invoking a shell."""
    destination = str(output / name)
    tail = [destination] if name == "bp" else ["--output-dir", destination]
    return [sys.executable, "-B", "-u", *options, *tail]


def run_jobs(jobs, output):
    """Record every job and propagate both training and report failures."""
    output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "PYTHONIOENCODING": "utf-8"}
    status = {name: {"state": "pending"} for name, _ in jobs}
    report_failed = False

    def save_status():
        pending = output / "status.tmp"
        pending.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        for attempt in range(20):
            try:
                pending.replace(output / "status.json")
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.25)

    for name, options in jobs:
        command = command_for(name, options, output)
        status[name].update(state="running", started=datetime.now(timezone.utc).isoformat(),
                            command=command)
        save_status()
        started = time.perf_counter()
        with (output / f"{name}.log").open("wb") as log:
            try:
                code = subprocess.run(command, env=environment, stdout=log,
                                      stderr=subprocess.STDOUT, check=False).returncode
            except OSError as error:
                log.write(str(error).encode("utf-8"))
                code = 1
        status[name].update(state="passed" if code == 0 else "failed",
                            returncode=code, seconds=time.perf_counter() - started)
        save_status()
        with (output / "summarize.log").open("ab") as log:
            try:
                result = subprocess.run(
                    [sys.executable, "-B", str(HERE / "summarize.py"), str(output)],
                    env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
                report_code = result.returncode
            except OSError as error:
                log.write(str(error).encode("utf-8"))
                report_code = 1
        status[name]["summary_returncode"] = report_code
        report_failed |= report_code != 0
        save_status()
    return int(report_failed or any(row["state"] == "failed" for row in status.values()))


def main(argv=None):
    """Print commands without side effects or execute selected experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", choices=("all", "bp", "ep-full", "sweep"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    jobs = job_list(args.data_dir.resolve(), args.only)
    if args.dry_run:
        print(json.dumps({"environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                          "jobs": [{"name": name, "command": command_for(name, options, output)}
                                   for name, options in jobs]}, indent=2))
        return 0
    if output.exists():
        parser.error("--output-dir must be a new directory")
    return run_jobs(jobs, output)


if __name__ == "__main__":
    sys.exit(main())
