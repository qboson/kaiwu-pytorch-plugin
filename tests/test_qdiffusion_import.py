"""Smoke tests for the public QDiffusion API surface."""

import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def test_qdiffusion_names_are_exported():
    module = importlib.import_module("kaiwu.torch_plugin")
    assert "QDiffusion" in module.__all__
    assert "QDiffusionConfig" in module.__all__
    assert "EnergyModel" in module.__all__


def test_qdiffusion_direct_import():
    module = importlib.import_module("kaiwu.torch_plugin")
    assert module.QDiffusion.__name__ == "QDiffusion"
    assert module.QDiffusionConfig.__name__ == "QDiffusionConfig"
    assert module.EnergyModel.__name__ == "EnergyModel"


def test_qdiffusion_removed_dplm_classmethods():
    module = importlib.import_module("kaiwu.torch_plugin.qdiffusion")
    assert not hasattr(module.QDiffusion, "from_pretrained")
    assert not hasattr(module.QDiffusion, "build")
    assert not hasattr(module.QDiffusion, "load_backbone")
