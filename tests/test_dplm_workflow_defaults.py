"""Default-config contract for the DPLM training workflow."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPLM = os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm")
sys.path[:0] = [DPLM, os.path.join(DPLM, "workflows")]

import train  # noqa: E402
from utils.io import default_fasta_path  # noqa: E402


def test_default_config_uses_the_bundled_fasta_path_and_checkpoints():
    """The example defaults point at the example FASTA and the public ckpts."""
    config = train.build_default_workflow_config()
    assert isinstance(config, train.WorkflowConfig)
    assert config.data.fasta_path == str(default_fasta_path())
    assert config.model.proposal_ckpt == "airkingbd/dplm_150m"
    assert config.model.energy_ckpt == "airkingbd/dplm_150m"
    assert config.model.freeze_proposal is True


def test_default_config_uses_the_sa_sampler_without_extra_kwargs():
    """The documented default sampler is local SA with an empty payload."""
    config = train.build_default_workflow_config()
    assert config.sampler.sampler_type == "sa"
    assert config.sampler.sampler_kwargs == {}


def test_default_config_keeps_the_short_example_budgets():
    """Training and generation defaults stay at the documented budgets."""
    config = train.build_default_workflow_config()
    assert (config.train.epochs, config.train.batch_size) == (20, 4)
    assert config.train.num_candidates == 4
    assert (config.generate.num_candidates, config.generate.steps) == (8, 5)
    assert (config.data.min_length, config.data.max_length) == (50, 256)
    assert config.data.max_records is None
