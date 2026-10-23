"""Freshness contract for the DPLM workflow config dataclasses."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPLM = os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm")
sys.path[:0] = [DPLM, os.path.join(DPLM, "workflows")]

import train  # noqa: E402


def test_two_default_configs_do_not_share_sub_configs():
    """Each call builds independent dataclass instances."""
    first = train.build_default_workflow_config()
    second = train.build_default_workflow_config()
    assert first is not second
    assert first.data is not second.data
    assert first.model is not second.model
    assert first.sampler is not second.sampler
    assert first.train is not second.train
    assert first.generate is not second.generate


def test_mutating_one_config_leaves_the_next_one_default():
    """Defaults are rebuilt per call instead of aliasing one shared instance."""
    first = train.build_default_workflow_config()
    first.train.epochs = 1
    first.sampler.sampler_kwargs["tmp_dir"] = "shared"
    second = train.build_default_workflow_config()
    assert second.train.epochs == 20
    assert second.sampler.sampler_kwargs == {}
