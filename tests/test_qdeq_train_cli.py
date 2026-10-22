"""The training example must reject non-positive epoch and batch counts."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"


@pytest.fixture
def train_example(monkeypatch):
    """Imports simple_train_example with a stubbed pandas module."""
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda path: None
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("etth1_common", None)
    spec = importlib.util.spec_from_file_location(
        "simple_train_example", EXAMPLE_DIR / "simple_train_example.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("etth1_common", None)
    sys.modules.pop("simple_train_example", None)


@pytest.mark.parametrize("flag", ["--epochs", "--max-train-batches"])
@pytest.mark.parametrize("value", ["0", "-3", "1.5"])
def test_non_positive_counts_are_rejected(train_example, monkeypatch, flag, value):
    monkeypatch.setattr(sys, "argv", ["simple_train_example.py", flag, value])

    with pytest.raises(SystemExit):
        train_example.parse_args()


def test_defaults_are_unchanged(train_example, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["simple_train_example.py"])

    args = train_example.parse_args()

    assert args.epochs == 2
    assert args.max_train_batches == 60
