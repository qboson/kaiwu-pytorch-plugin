"""Entrypoint-export contract for the DPLM training wrapper."""

import importlib
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion"))

module = importlib.import_module("dplm.train_workflow")
workflow = importlib.import_module("dplm.workflows.train")


def test_main_is_reexported_from_the_workflow_module():
    """The compatibility entrypoint runs the real workflow main."""
    assert module.main is workflow.main


def test_public_configuration_names_are_reexported():
    """``import *`` keeps the configuration dataclasses available."""
    for name in ("WorkflowConfig", "DataConfig", "ModelConfig", "TrainConfig"):
        assert hasattr(module, name), name
        assert getattr(module, name) is getattr(workflow, name)
