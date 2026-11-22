"""Regression coverage for the ETTh1 training example's MSE."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.mark.parametrize(("batch_limit", "train_mse"), [(1, 0.0), (2, 3.0)])
def test_train_mse_counts_processed_target_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    batch_limit: int,
    train_mse: float,
) -> None:
    example = Path(__file__).parents[1] / "example" / "qdeq" / "simple"
    monkeypatch.syspath_prepend(str(example))
    import simple_train_example

    class ConstantForecaster(nn.Module):
        def __init__(self, *_args: object) -> None:
            super().__init__()
            self.offset = nn.Parameter(torch.tensor(0.0))
            self.deq = SimpleNamespace(
                get_last_stats=lambda: {"forward_nstep": 1, "backward_nstep": 1}
            )

        def forward(self, history: torch.Tensor) -> torch.Tensor:
            return history + self.offset

    histories = torch.zeros(3, 1, 1)
    targets = torch.tensor([0.0, 0.0, 3.0]).reshape(3, 1, 1)
    loader = DataLoader(TensorDataset(histories, targets), batch_size=2)
    loaders = {"train": loader, "val": loader, "test": loader}
    monkeypatch.setattr(simple_train_example, "build_loaders", lambda _: loaders)
    monkeypatch.setattr(simple_train_example, "DEQForecaster", ConstantForecaster)
    checkpoint = tmp_path / "model.pt"
    argv = [
        "simple_train_example.py", "--sampler", "anderson", "--epochs", "1",
        "--lr", "0", "--checkpoint", str(checkpoint),
        "--max-train-batches", str(batch_limit),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    simple_train_example.main()

    assert f"train mse {train_mse:.4f} | val mse 3.0000" in capsys.readouterr().out
    assert checkpoint.is_file()
