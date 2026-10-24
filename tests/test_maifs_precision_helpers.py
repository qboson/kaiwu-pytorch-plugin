"""Kaiwu precision-helper lookup contract for the MAIFS QUBO module."""

import os
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import kaiwu.torch_plugin.maifs.qubo as qubo  # noqa: E402


def fake_import(mapping):
    """Builds an ``import_module`` replacement backed by a name mapping."""
    imports = []

    def importer(name):
        imports.append(name)
        if name not in mapping:
            raise ImportError(name)
        return mapping[name]

    importer.imports = imports
    return importer


def test_helpers_are_returned_from_the_first_supporting_module(monkeypatch):
    """A module exposing both helpers wins and its functions are returned."""
    adjust = object()
    width = object()
    importer = fake_import(
        {"kaiwu.preprocess": SimpleNamespace(adjust_ising_matrix_precision=adjust,
                                            calculate_ising_matrix_bit_width=width)}
    )
    monkeypatch.setattr(qubo.importlib, "import_module", importer)
    assert qubo._get_kaiwu_precision_helpers() == (adjust, width)


def test_missing_modules_and_missing_symbols_raise_import_error(monkeypatch):
    """Without a supporting module the lookup fails with a named error."""
    importer = fake_import({"kaiwu.preprocess": SimpleNamespace(), "kaiwu.ising": SimpleNamespace()})
    monkeypatch.setattr(qubo.importlib, "import_module", importer)
    with pytest.raises(ImportError, match="precision helpers are unavailable"):
        qubo._get_kaiwu_precision_helpers()


def test_fallback_module_is_used_when_the_first_lacks_symbols(monkeypatch):
    """The second module path is tried when the first lacks the helpers."""
    adjust = object()
    width = object()
    importer = fake_import(
        {
            "kaiwu.preprocess": SimpleNamespace(),
            "kaiwu.ising": SimpleNamespace(
                adjust_ising_matrix_precision=adjust,
                calculate_ising_matrix_bit_width=width,
            ),
        }
    )
    monkeypatch.setattr(qubo.importlib, "import_module", importer)
    assert qubo._get_kaiwu_precision_helpers() == (adjust, width)
    assert importer.imports == ["kaiwu.preprocess", "kaiwu.ising"]
