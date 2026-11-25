"""Exercise the MNIST CLI with synthetic data and a deterministic sampler."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


class SyntheticMNIST:
    """Tiny images; accessing the official test split is a test failure."""

    def __init__(self, _path, train, download):
        assert train, "validation-only runs must not load the test split"
        self.data = (torch.arange(8 * 784) % 256).to(torch.uint8).reshape(8, 28, 28)
        self.targets = torch.arange(8)

    def __len__(self):
        return len(self.targets)


class CandidateSampler:
    """Return two valid candidates without a licensed solve or random state."""

    def __init__(self, **settings):
        self.settings = settings

    def solve(self, matrix):
        rows = np.ones((2, len(matrix)))
        rows[1, ::2] = -1
        return rows


@pytest.fixture
def entry(monkeypatch):
    """Load the real entrypoint and replace only data acquisition and sampling."""
    pytest.importorskip("torchvision", reason="install example/ep_mnist/requirements.txt")
    path = Path(__file__).resolve().parents[1] / "example/ep_mnist/train_mnist.py"
    spec = importlib.util.spec_from_file_location("ep_mnist_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "MNIST", SyntheticMNIST)
    monkeypatch.setattr(module, "SimulatedAnnealingOptimizer", CandidateSampler)
    return module


@pytest.mark.parametrize("conv", [False, True])
def test_training_and_checkpoint_evaluation(entry, monkeypatch, tmp_path, conv):
    """Restore saved configuration and metrics, even with conflicting CLI options."""
    output = tmp_path / "train"
    options = ["train_mnist", "--output-dir", str(output), "--validation-only",
               "--n-val", "2", "--limit-train", "4", "--limit-eval", "2",
               "--hidden", "2", "--channels", "1", "--epochs", "1",
               "--batch-size", "2", "--seed", "7", "--split-seed", "3"]
    if conv:
        options.append("--conv")
    monkeypatch.setattr(sys, "argv", options)
    entry.main()
    summary = json.loads((output / ("summary_conv.json" if conv else "summary.json")).read_text())
    assert (summary["n_train"], summary["n_val"], summary["n_test"]) == (4, 2, 0)
    assert "test_accuracy" not in summary
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    assert checkpoint["config"]["split_seed"] == 3
    evaluation = tmp_path / "evaluate"
    monkeypatch.setattr(sys, "argv", ["train_mnist", "--evaluate", str(output / "checkpoint.pt"),
                                      "--output-dir", str(evaluation), "--hidden", "99",
                                      "--seed", "99", "--split-seed", "99"])
    entry.main()
    restored = json.loads((evaluation / "evaluation.json").read_text())
    assert restored["epoch"] == 1
    assert "test_accuracy" not in restored
    for metric in ("train_accuracy", "val_accuracy", "train_error", "val_error"):
        assert restored[metric] == summary["history"][-1][metric]
    assert not (output / "checkpoint.tmp").exists()


@pytest.mark.parametrize("flag,value", [
    ("--epochs", "0"), ("--batch-size", "-1"), ("--beta", "nan"),
    ("--lr", "inf"), ("--input-scale", "0"), ("--split-seed", "-1"),
])
def test_invalid_arguments_create_no_output(entry, monkeypatch, tmp_path, flag, value):
    """Reject invalid inputs before creating output or loading data."""
    output = tmp_path / "invalid"
    monkeypatch.setattr(sys, "argv", ["train_mnist", "--output-dir", str(output), flag, value])
    with pytest.raises(SystemExit):
        entry.main()
    assert not output.exists()


def test_training_refuses_existing_output(entry, monkeypatch, tmp_path):
    """The training CLI protects prior results before doing any work."""
    marker = tmp_path / "keep.txt"
    marker.write_text("original")
    monkeypatch.setattr(sys, "argv", ["train_mnist", "--output-dir", str(tmp_path)])
    with pytest.raises(FileExistsError):
        entry.main()
    assert marker.read_text() == "original"
