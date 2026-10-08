"""Test dataset loading.

Usage:
    python test_datasets.py              # test all datasets
    python test_datasets.py thyroid      # test thyroid only
    python test_datasets.py creditcard   # test creditcard only
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.datasets import load_thyroid, load_creditcard

LOADERS = {
    "thyroid": load_thyroid,
    "creditcard": load_creditcard,
}


def test_dataset_shapes(name: str):
    """Test dataset shapes."""
    loader = LOADERS[name]
    data = loader()
    assert data["name"] == name
    assert data["dim"] == data["x_train"].shape[1]
    assert data["x_val"].shape[1] == data["dim"]
    assert data["x_test"].shape[1] == data["dim"]
    print(f"test_{name}_shapes: OK")


def test_dataset_labels(name: str):
    """Test dataset labels are 0/1."""
    loader = LOADERS[name]
    data = loader()
    y = data["y_train"]
    assert set(y.unique().numpy()) <= {0, 1}, f"{name} labels should be 0 or 1"
    print(f"test_{name}_labels: OK")


def test_dataset_split(name: str):
    """Test dataset split: val/test have both classes."""
    loader = LOADERS[name]
    data = loader()
    val_pos = data["y_val"].sum().item()
    val_neg = len(data["y_val"]) - val_pos
    assert val_pos > 0 and val_neg > 0, f"{name} val should have both classes"
    print(f"test_{name}_split: OK")


def test_dataset(name: str):
    """Run all tests for one dataset."""
    print(f"\n=== Testing {name} ===")
    test_dataset_shapes(name)
    test_dataset_labels(name)
    test_dataset_split(name)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = list(LOADERS.keys())

    for name in names:
        if name not in LOADERS:
            print(f"Unknown dataset: {name}")
            continue
        test_dataset(name)

    print("\nAll dataset tests passed!")
