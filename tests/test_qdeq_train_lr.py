"""The training example must reject a non-positive learning rate."""

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


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf"])
def test_a_non_positive_learning_rate_is_rejected(train_example, monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["simple_train_example.py", "--lr", value])

    with pytest.raises(SystemExit):
        train_example.parse_args()


def test_the_default_learning_rate_is_kept(train_example, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["simple_train_example.py"])

    args = train_example.parse_args()

    assert args.lr == 1e-3
