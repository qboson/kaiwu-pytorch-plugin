"""Contract test for ``dplm.utils.io.default_outputs_root``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IO_MODULE = REPO / "example" / "qdiffusion" / "dplm" / "utils" / "io.py"


def _load_io():
    spec = importlib.util.spec_from_file_location("dplm_io_outputs_uut", IO_MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_points_at_the_example_outputs_directory():
    io_module = _load_io()

    root = io_module.default_outputs_root()

    assert isinstance(root, Path)
    assert root.is_absolute()
    assert root == io_module.CASE_ROOT / "outputs"
    # CASE_ROOT is the qdiffusion example directory, not the package directory.
    assert root.parent == IO_MODULE.resolve().parents[2]
    assert root.name == "outputs"


def test_matches_the_case_root_derivation():
    io_module = _load_io()

    assert io_module.CASE_ROOT == IO_MODULE.resolve().parents[2]
    assert io_module.default_outputs_root().parent == io_module.CASE_ROOT
