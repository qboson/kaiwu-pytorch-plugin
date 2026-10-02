"""Regression test for the CIM branch of ``build_bm_sampler``.

``kp.common.CheckpointManager`` has no default ``save_dir`` and the Kaiwu CIM
optimizer refuses to construct without one:

    >>> kw.common.CheckpointManager.save_dir = None
    >>> kaiwu.cim.CIMOptimizer(task_name="t")
    ValueError: The save directory is required

``build_bm_sampler(sampler_type="cim")`` only propagated ``tmp_dir`` when the
caller passed it, so the documented ``build_cim_eval_config`` defaults (whose
``tmp_dir`` is ``None``) crashed at construction.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import kaiwu as kw  # noqa: E402
import kaiwu.cim  # noqa: E402

from models.sampler import build_bm_sampler  # noqa: E402


class _RecordingOptimizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture()
def _recording_cim(monkeypatch):
    monkeypatch.setattr(kaiwu.cim, "CIMOptimizer", _RecordingOptimizer)
    monkeypatch.setattr(kw.common.CheckpointManager, "save_dir", None)
    yield
    monkeypatch.setattr(kw.common.CheckpointManager, "save_dir", None)


def test_cim_sampler_sets_a_default_checkpoint_dir(_recording_cim, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    sampler = build_bm_sampler(sampler_type="cim", sampler_kwargs={"task_name": "t"})

    assert isinstance(sampler, _RecordingOptimizer)
    assert kw.common.CheckpointManager.save_dir, (
        "build_bm_sampler must configure a checkpoint directory for the CIM runtime"
    )


def test_cim_sampler_honours_an_explicit_tmp_dir(_recording_cim, tmp_path):
    explicit = str(tmp_path / "cim_cache")

    build_bm_sampler(
        sampler_type="cim",
        sampler_kwargs={"task_name": "t", "tmp_dir": explicit},
    )

    assert kw.common.CheckpointManager.save_dir == explicit
