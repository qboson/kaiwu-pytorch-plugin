"""Regression coverage for the QDEQ training example checkpoint."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def train_example(monkeypatch, tmp_path):
    """Exercise the training entry point with one small CPU training batch."""
    example = Path(__file__).resolve().parents[1] / "example/qdeq/simple"
    monkeypatch.syspath_prepend(str(example))
    import simple_train_example as train

    checkpoint = tmp_path / "forecast.pt"
    args = SimpleNamespace(
        seed=0, data_path=None, sampler="anderson", checkpoint=checkpoint,
        lr=1e-3, epochs=1, max_train_batches=1,
    )
    model = torch.nn.Linear(1, 1)
    model.deq = SimpleNamespace(
        get_last_stats=lambda: {"forward_nstep": 1, "backward_nstep": 1}
    )
    batch = (torch.ones(2, 1), torch.zeros(2, 1))
    monkeypatch.setattr(train, "parse_args", lambda: args)
    monkeypatch.setattr(train, "build_loaders", lambda _: {"train": [batch], "val": [], "test": []})
    monkeypatch.setattr(train, "build_sampler", lambda *_: None)
    monkeypatch.setattr(train, "DEQForecaster", lambda *_: model)
    monkeypatch.setattr(train, "evaluate", lambda *_: {"mse": 0, "mae": 0, "steps": 1})
    return train, checkpoint


def test_checkpoint_replacement_preserves_loadable_format(train_example):
    train, checkpoint = train_example
    torch.save({"state_dict": {}, "sampler": "sa"}, checkpoint)

    train.main()

    saved = torch.load(checkpoint, weights_only=True)
    assert saved["sampler"] == "anderson"
    assert set(saved["state_dict"]) == {"weight", "bias"}
    assert list(checkpoint.parent.iterdir()) == [checkpoint]


def test_partial_write_keeps_previous_checkpoint(train_example, monkeypatch):
    train, checkpoint = train_example
    torch.save({"state_dict": {}, "sampler": "sa"}, checkpoint)
    previous = checkpoint.read_bytes()

    def fail_after_write(_, destination):
        if hasattr(destination, "write"):
            destination.write(b"partial checkpoint")
        else:
            Path(destination).write_bytes(b"partial checkpoint")
        raise OSError("checkpoint write failed")

    monkeypatch.setattr(train.torch, "save", fail_after_write)
    with pytest.raises(OSError, match="checkpoint write failed"):
        train.main()

    assert checkpoint.read_bytes() == previous
    assert list(checkpoint.parent.iterdir()) == [checkpoint]
