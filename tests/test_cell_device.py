"""Backend selection without requiring GPU hardware or single-cell dependencies."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location(
    "cell_runtime", Path(__file__).resolve().parents[1] / "example/qvae_cell/runtime.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.mark.parametrize("requested,cuda,mps,expected", [
    ("mps", False, True, "mps"),
    ("mps", False, False, "cpu"),
    ("cuda:1", True, False, "cuda:1"),
    ("cuda:1", False, True, "cpu"),
    ("cpu", True, True, "cpu"),
])
def test_requested_backend(requested, cuda, mps, expected):
    with patch.object(runtime.torch.cuda, "is_available", return_value=cuda), patch.object(
        runtime.torch.backends.mps, "is_available", return_value=mps
    ):
        assert str(runtime.resolve_device(requested)) == expected
