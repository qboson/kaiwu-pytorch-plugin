"""FID-evaluation helper contract for the QVAE-MNIST example."""

import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

_spec = importlib.util.spec_from_file_location(
    "qvae_mnist_helpers_fid",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "helpers.py"),
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def test_fid_result_is_written_with_training_metadata(tmp_path, monkeypatch):
    """The computed score is returned and persisted with the run metadata."""
    calls = {}

    def fake_fid(fake_imgs, real_imgs, device, batch_size):
        calls["batch_size"] = batch_size
        calls["device"] = device
        return 3.5

    monkeypatch.setattr(helpers, "compute_fid_in_batches", fake_fid)
    trainer = SimpleNamespace(
        config=SimpleNamespace(num_epochs=7),
        model=SimpleNamespace(_latent_dimensions=256),
    )
    score = helpers.evaluate_qvae_fid(
        trainer,
        torch.zeros(2, 784),
        torch.zeros(2, 784),
        device="cpu",
        save_path=str(tmp_path),
        batch_size=32,
    )
    assert score == 3.5
    assert calls == {"batch_size": 32, "device": "cpu"}
    lines = (tmp_path / "fid_results.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "FID分数: 3.5000"
    assert lines[1] == "训练轮次: 7"
    assert lines[2] == "潜在维度: 256"
    assert lines[3].startswith("评估时间: ")
