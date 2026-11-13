"""The bm_generation saver must create a fresh checkpoint directory."""

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_saver():
    spec = importlib.util.spec_from_file_location(
        "bm_generation_saver", ROOT / "example" / "bm_generation" / "saver.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_save_info_creates_a_missing_directory(tmp_path):
    saver = load_saver().Saver(log_path=str(tmp_path / "log"))
    checkpoint_dir = tmp_path / "checkpoints" / "nested"

    saver.save_info(torch.nn.Linear(2, 2), str(checkpoint_dir), 0, 0.0)

    assert (checkpoint_dir / "rbm_model0.pth").is_file()
