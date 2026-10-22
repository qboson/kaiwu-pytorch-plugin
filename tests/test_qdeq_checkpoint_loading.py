"""Checkpoint loading must fail clearly when the file is missing."""

import sys
import types
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"


@pytest.fixture
def etth1(monkeypatch):
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda path: None
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("etth1_common", None)
    import etth1_common

    yield etth1_common
    sys.modules.pop("etth1_common", None)


def test_a_saved_checkpoint_is_loaded_on_the_cpu(etth1, tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"state_dict": {"w": torch.zeros(1)}}, path)

    checkpoint = etth1.load_checkpoint(path)

    assert set(checkpoint) == {"state_dict"}
    assert checkpoint["state_dict"]["w"].device.type == "cpu"


def test_a_missing_checkpoint_names_the_training_script(etth1, tmp_path):
    with pytest.raises(FileNotFoundError, match="simple_train_example.py"):
        etth1.load_checkpoint(tmp_path / "missing.pt")
