"""MNIST test collection must not replace SDK modules used by other tests."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["skip", "load"])
def test_mnist_collection_preserves_shared_import_state(mode):
    """Check both optional-dependency skips and successful collection in isolation."""
    script = r"""
import importlib
import runpy
import sys
from unittest.mock import patch

import pytest
import kaiwu
import kaiwu.torch_plugin

# Third-party libraries may legitimately initialize their own search paths.
# Snapshot after those imports so this contract targets this test module.
if sys.argv[2] == "load":
    try:
        for dependency in ("imageio", "torchvision", "torchmetrics", "pandas", "seaborn", "tqdm"):
            pytest.importorskip(dependency)
    except pytest.skip.Exception:
        sys.exit(5)

names = ("kaiwu.classical", "kaiwu.cim", "kaiwu.common", "gif")
for name in names[:-1]:
    importlib.import_module(name)
missing = object()
modules_before = {name: sys.modules.get(name, missing) for name in names}
path_before = list(sys.path)
kaiwu_path_before = kaiwu.__path__

try:
    if sys.argv[2] == "skip":
        with patch.object(pytest, "importorskip", side_effect=pytest.skip.Exception("forced")):
            runpy.run_path(sys.argv[1])
    else:
        runpy.run_path(sys.argv[1])
except pytest.skip.Exception:
    skipped = True
else:
    skipped = False

assert sys.path == path_before, "collection changed sys.path"
assert kaiwu.__path__ is kaiwu_path_before, "collection replaced the SDK package path"
for name, previous in modules_before.items():
    assert sys.modules.get(name, missing) is previous, "collection replaced " + name
if skipped and sys.argv[2] == "load":
    sys.exit(5)
"""
    test_module = Path(__file__).with_name("test_mnist_device_placement.py")
    result = subprocess.run(
        [sys.executable, "-c", script, str(test_module), mode],
        cwd=test_module.resolve().parents[1],
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    if result.returncode == 5:
        pytest.skip("MNIST example dependencies are unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
