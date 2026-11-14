"""Regression test for zero/negative training budgets in the DPLM workflow."""

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for extra in (
    os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"),
    os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm", "workflows"),
):
    sys.path.insert(0, extra)
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

pytest.importorskip("omegaconf")
pytest.importorskip("transformers")

from train import TrainConfig, build_default_workflow_config  # noqa: E402


def test_zero_epochs_are_rejected_before_workflow_setup():
    with pytest.raises(ValueError, match="epochs"):
        TrainConfig(epochs=0)


def test_negative_epochs_are_rejected():
    with pytest.raises(ValueError, match="epochs"):
        TrainConfig(epochs=-1)


def test_default_workflow_still_uses_positive_epochs():
    config = build_default_workflow_config()

    assert config.train.epochs >= 1
