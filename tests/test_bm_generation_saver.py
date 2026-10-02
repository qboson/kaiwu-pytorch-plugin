"""Regression test for Saver.save_info creating its output directory."""

import os
import sys

import torch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/bm_generation"))
)
from saver import Saver  # noqa: E402  pylint: disable=wrong-import-position


def test_save_info_creates_missing_directory(tmp_path):
    saver = Saver(log_path=str(tmp_path / "log"))
    save_dir = tmp_path / "fresh_run" / "checkpoints"
    model = torch.nn.Linear(2, 2)

    saver.save_info(model, str(save_dir), 0, 0.0)

    saved = save_dir / "rbm_model0.pth"
    assert saved.is_file()
    assert isinstance(torch.load(saved, weights_only=False), torch.nn.Linear)
