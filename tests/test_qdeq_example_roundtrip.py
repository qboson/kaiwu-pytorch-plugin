"""The CPU example must train, save, and restore predictions in another process.

This uses synthetic CSV data and one optimizer step, not an ETTh1 benchmark.
The real Anderson solver, CSV reader, training loop and serialization all run.
"""

import csv
import importlib.util
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"
SEED = 20261029


def _run_example(script, data_path, checkpoint, work_dir, *extra):
    """Run a public CLI with real dependencies and a bounded CPU budget."""
    result = subprocess.run(
        [
            sys.executable, str(EXAMPLE_DIR / script),
            "--data-path", str(data_path), "--checkpoint", str(checkpoint),
            "--sampler", "anderson", "--seed", str(SEED), *extra,
        ],
        cwd=work_dir,
        env={
            **os.environ, "KPP_STATS_ENABLED": "false", "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUTF8": "1",
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _test_metrics(output, prefix):
    """Read the reported test metrics without pinning benchmark scores."""
    match = re.search(rf"^{re.escape(prefix)}test mse (\S+) mae (\S+)", output, re.MULTILINE)
    assert match is not None, output
    metrics = tuple(float(value) for value in match.groups())
    assert all(math.isfinite(value) for value in metrics), output
    return metrics


def test_anderson_training_checkpoint_is_consumed_by_forecast_cli(tmp_path):
    """One real update survives the train/save/load/forecast CLI boundary."""
    data_path = tmp_path / "synthetic.csv"
    # The example has a fixed 12/4/4-month split; one variable keeps it small.
    with data_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "OT"])
        writer.writerows((index, math.sin(index / 24)) for index in range(14400))
    checkpoint_path = tmp_path / "checkpoints" / "model.pt"
    training_output = _run_example(
        "simple_train_example.py", data_path, checkpoint_path, tmp_path,
        "--epochs", "1", "--max-train-batches", "1",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["sampler"] == "anderson"
    state = checkpoint["state_dict"]
    assert state and all(torch.isfinite(value).all() for value in state.values())

    spec = importlib.util.spec_from_file_location(
        "roundtrip_etth1", EXAMPLE_DIR / "etth1_common.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    # Reproduce initialization without changing the test runner's RNG state.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(SEED)
        initial = example.DEQForecaster(96, 96, None, example.build_config())
    assert not torch.equal(state["embed.weight"], initial.embed.weight), "No parameter update saved"

    forecast_output = _run_example(
        "simple_forecast_example.py", data_path, checkpoint_path, tmp_path,
    )
    # Both CLIs print four decimal places. A fresh process must restore the
    # trained model rather than evaluate independently initialized weights.
    assert _test_metrics(forecast_output, "anderson: ") == pytest.approx(
        _test_metrics(training_output, ""), rel=0, abs=1e-4,
    )
