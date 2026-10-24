"""Split-helper lookup contract for the MAIFS QUBO module."""

import os
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import kaiwu.torch_plugin.maifs.qubo as qubo  # noqa: E402


def test_helper_is_returned_from_the_preprocess_module(monkeypatch):
    """The split helper is read from ``kaiwu.preprocess`` by name."""
    helper = object()
    monkeypatch.setattr(
        qubo.importlib,
        "import_module",
        lambda name: SimpleNamespace(perform_precision_adaption_split=helper),
    )
    assert qubo._get_kaiwu_split_helper() is helper


def test_missing_helper_surfaces_as_attribute_error(monkeypatch):
    """A runtime without the helper fails loudly instead of returning None."""
    monkeypatch.setattr(
        qubo.importlib, "import_module", lambda name: SimpleNamespace()
    )
    with pytest.raises(AttributeError, match="perform_precision_adaption_split"):
        qubo._get_kaiwu_split_helper()
