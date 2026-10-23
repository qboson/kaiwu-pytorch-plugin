"""Builder-payload contract for the DPLM training workflow config."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPLM = os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm")
sys.path[:0] = [DPLM, os.path.join(DPLM, "workflows")]

import train  # noqa: E402


def make_config(sampler_type="cim", sampler_kwargs=None):
    """Builds a workflow config carrying the sampler block."""
    return train.WorkflowConfig(
        data=train.DataConfig(fasta_path="data.fasta"),
        model=train.ModelConfig(proposal_ckpt="p", energy_ckpt="e"),
        sampler=train.SamplerConfig(
            sampler_type=sampler_type,
            sampler_kwargs={} if sampler_kwargs is None else sampler_kwargs,
        ),
    )


def test_payload_forwards_the_sampler_block():
    """The builder receives the configured type and keyword dictionary."""
    config = make_config(sampler_kwargs={"tmp_dir": "checkpoints", "wait": True})
    payload = train.make_bm_build_kwargs(config)
    assert payload == {
        "bm_sampler_type": "cim",
        "bm_sampler_kwargs": {"tmp_dir": "checkpoints", "wait": True},
    }


def test_payload_detaches_the_keyword_dictionary():
    """Editing the payload must not write back into the workflow config."""
    config = make_config(sampler_kwargs={"tmp_dir": "checkpoints"})
    payload = train.make_bm_build_kwargs(config)
    payload["bm_sampler_kwargs"]["wait"] = True
    assert config.sampler.sampler_kwargs == {"tmp_dir": "checkpoints"}
