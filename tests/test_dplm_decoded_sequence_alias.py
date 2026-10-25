"""Contract test for ``dplm.utils.io.normalize_decoded_sequence``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
IO_MODULE = REPO / "example" / "qdiffusion" / "dplm" / "utils" / "io.py"


def _load_io():
    spec = importlib.util.spec_from_file_location("dplm_io_decoded_uut", IO_MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("decoded", "expected"),
    [
        ("A C G T", "ACGT"),
        ("  ACGT  ", "ACGT"),
        ("AC GT  AC", "ACGTAC"),
        ("", ""),
        ("acgt", "acgt"),
        ("M K L", "MKL"),
    ],
)
def test_compact_sequence_text(decoded, expected):
    io_module = _load_io()

    assert io_module.normalize_decoded_sequence(decoded) == expected


@pytest.mark.parametrize("decoded", ["A C G T", "  ACGT  ", "", "ac gt"])
def test_stays_an_alias_of_normalize_sequence(decoded):
    io_module = _load_io()

    assert io_module.normalize_decoded_sequence(
        decoded
    ) == io_module.normalize_sequence(decoded)
