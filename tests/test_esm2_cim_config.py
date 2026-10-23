"""CIM evaluation-config contract for the ESM2 workflow."""

import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPLM = os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm")
sys.path[:0] = [DPLM, os.path.join(DPLM, "workflows")]

_esm = types.ModuleType("esm")
_added = "esm" not in sys.modules
if _added:
    sys.modules["esm"] = _esm
import esm2_eval  # noqa: E402
if _added:
    sys.modules.pop("esm", None)


def build(**overrides):
    """Builds one CIM config with the required paths and task name."""
    arguments = {
        "reference_fasta": Path("ref.fasta"),
        "proposal_ckpt": "proposal",
        "energy_ckpt": "energy",
        "guided_checkpoint": "guided.pt",
        "output_dir": Path("out"),
        "device": "cpu",
        "task_name": "qdiffusion_eval",
    }
    arguments.update(overrides)
    return esm2_eval.build_cim_eval_config(**arguments)


DEFAULT_KWARGS = {
    "task_name": "qdiffusion_eval",
    "wait": False,
    "interval": 1,
    "project_no": None,
    "task_mode": None,
    "sample_number": None,
    "use_precision_reducer": False,
    "precision": 8,
    "truncated_precision": 10,
    "target_bits": 550,
    "only_feasible_solution": False,
    "tmp_dir": None,
}


def test_cim_sampler_payload_uses_the_documented_defaults():
    """All twelve CIM knobs are recorded with their default values."""
    config = build()
    assert isinstance(config, esm2_eval.EvalConfig)
    assert config.bm_sampler_type == "cim"
    assert config.bm_sampler_kwargs == DEFAULT_KWARGS


def test_explicit_cim_overrides_reach_the_payload():
    """Every explicit knob replaces the default without dropping the others."""
    config = build(
        wait=True,
        interval=3,
        project_no="p-1",
        task_mode="SAMPLING",
        sample_number=5,
        use_precision_reducer=True,
        precision=6,
        truncated_precision=8,
        target_bits=400,
        only_feasible_solution=True,
        tmp_dir="ckpt",
    )
    expected = dict(DEFAULT_KWARGS)
    expected.update(
        {
            "wait": True,
            "interval": 3,
            "project_no": "p-1",
            "task_mode": "SAMPLING",
            "sample_number": 5,
            "use_precision_reducer": True,
            "precision": 6,
            "truncated_precision": 8,
            "target_bits": 400,
            "only_feasible_solution": True,
            "tmp_dir": "ckpt",
        }
    )
    assert config.bm_sampler_kwargs == expected


def test_cim_config_keeps_the_shared_evaluation_defaults():
    """Switching the backend does not change the evaluation settings."""
    config = build()
    assert config.esm2_model == "esm2_t33_650M_UR50D"
    assert (config.pair_mode, config.pooling) == ("order", "mean")
    assert (config.batch_size, config.max_records) == (1, 20)
    assert (config.generation_steps, config.seed) == (500, 42)
    assert config.freeze_proposal is True
    assert config.guided_num_candidates == 4
    assert config.guided_energy_temperature == pytest.approx(1.25)
    assert config.guided_resample_ratio == pytest.approx(0.20)
    assert config.guided_resample_top_p == pytest.approx(0.90)
