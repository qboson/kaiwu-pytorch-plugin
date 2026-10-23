"""Loss-history writer contract for the QVAE-MNIST helpers."""

import importlib.util
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_mnist_helpers_txt", os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "helpers.py")
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def test_values_are_written_with_six_decimals_one_per_line(tmp_path):
    """Each value becomes a fixed-precision line."""
    target = tmp_path / "losses.txt"
    helpers.save_list_to_txt(str(target), [0.1, 2.0 / 3.0, 1.23456789])
    assert target.read_text(encoding="utf-8").splitlines() == [
        "0.100000",
        "0.666667",
        "1.234568",
    ]


def test_existing_file_is_replaced(tmp_path):
    """The writer truncates an existing history file."""
    target = tmp_path / "losses.txt"
    target.write_text("old contents\n", encoding="utf-8")
    helpers.save_list_to_txt(str(target), [0.5])
    assert target.read_text(encoding="utf-8") == "0.500000\n"
