"""Smoke tests for the public QDiffusion API surface."""

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def test_qdiffusion_names_are_exported():
    module = importlib.import_module("kaiwu.torch_plugin")
    assert "QDiffusion" in module.__all__
    assert "QDiffusionConfig" in module.__all__


def test_qdiffusion_direct_import_when_dependencies_exist():
    pytest.importorskip("omegaconf")
    pytest.importorskip("transformers")

    module = importlib.import_module("kaiwu.torch_plugin")
    assert module.QDiffusion.__name__ == "QDiffusion"
    assert module.QDiffusionConfig.__name__ == "QDiffusionConfig"
