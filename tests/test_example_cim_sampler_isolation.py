# -*- coding: utf-8 -*-
"""CIM sampler construction in examples must not clobber global state."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "example"
for _path in (
    EXAMPLES / "rbm_digits",
    EXAMPLES / "dbn_digits",
    EXAMPLES / "qvae_mnist",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import kaiwu as kw  # noqa: E402  pylint: disable=wrong-import-position
import dbn_trainer  # noqa: E402  pylint: disable=wrong-import-position
import model.model as mnist_model  # noqa: E402  pylint: disable=wrong-import-position
import rbm_digits  # noqa: E402  pylint: disable=wrong-import-position
import utils.helpers as qvae_helpers  # noqa: E402  pylint: disable=wrong-import-position


class _FakeCIM:
    """Record CIM optimizer arguments without contacting the platform."""

    created: list["_FakeCIM"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeCIM.created.append(self)

    def solve(self, ising_matrix):
        """Return no solution; only construction is under test."""
        del ising_matrix
        return None


class _FakeReducer:
    """Minimal stand-in for the precision reducer wrapper."""

    def __init__(self, sampler, **kwargs):
        self.sampler = sampler
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def _restore_checkpoint_dir():
    """Restore the process-global checkpoint directory after every test."""
    original = kw.common.CheckpointManager.save_dir
    _FakeCIM.created.clear()
    yield
    kw.common.CheckpointManager.save_dir = original
    _FakeCIM.created.clear()


def _patch_factories(monkeypatch, module) -> None:
    """Patch the CIM classes imported by one example module."""
    monkeypatch.setattr(module, "CIMOptimizer", _FakeCIM)
    monkeypatch.setattr(module, "PrecisionReducer", _FakeReducer)


def _task_names() -> list[str]:
    return [instance.kwargs.get("task_name") for instance in _FakeCIM.created]


def test_rbm_runner_preserves_checkpoint_dir_and_uses_unique_tasks(
    monkeypatch, tmp_path
) -> None:
    """RBMRunner must keep the user's checkpoint dir and avoid a fixed task name."""
    _patch_factories(monkeypatch, rbm_digits)
    checkpoint_dir = tmp_path / "user-checkpoints"
    kw.common.CheckpointManager.save_dir = str(checkpoint_dir)

    rbm_digits.RBMRunner(use_cim=True)
    rbm_digits.RBMRunner(use_cim=True)

    assert kw.common.CheckpointManager.save_dir == str(checkpoint_dir)
    assert checkpoint_dir.is_dir()
    names = _task_names()
    assert len(names) == 2
    assert names[0] != "test_kpp"
    assert names[0] != names[1]


def test_dbn_trainer_preserves_checkpoint_dir_and_uses_unique_tasks(
    monkeypatch, tmp_path
) -> None:
    """DBNTrainer has the same CIM construction contract as RBMRunner."""
    _patch_factories(monkeypatch, dbn_trainer)
    checkpoint_dir = tmp_path / "user-checkpoints"
    kw.common.CheckpointManager.save_dir = str(checkpoint_dir)

    dbn_trainer.DBNTrainer(use_cim=True)
    dbn_trainer.DBNTrainer(use_cim=True)

    assert kw.common.CheckpointManager.save_dir == str(checkpoint_dir)
    assert checkpoint_dir.is_dir()
    names = _task_names()
    assert names[0] != "test_kpp"
    assert names[0] != names[1]


def test_mnist_qvae_sampler_preserves_checkpoint_dir(monkeypatch, tmp_path) -> None:
    """MnistQVAE must not overwrite a configured checkpoint directory."""
    _patch_factories(monkeypatch, mnist_model)
    checkpoint_dir = tmp_path / "user-checkpoints"
    kw.common.CheckpointManager.save_dir = str(checkpoint_dir)

    mnist_model.MnistQVAE._create_sampler(object(), "cim")
    mnist_model.MnistQVAE._create_sampler(object(), "cim")

    assert kw.common.CheckpointManager.save_dir == str(checkpoint_dir)
    assert checkpoint_dir.is_dir()
    names = _task_names()
    assert names[0] != "qvae_sampling"
    assert names[0] != names[1]


def test_init_qvae_sampler_preserves_checkpoint_dir(monkeypatch, tmp_path) -> None:
    """The generation helper must not reset the global checkpoint directory."""
    monkeypatch.setattr(kw.cim, "CIMOptimizer", _FakeCIM)
    monkeypatch.setattr(kw.cim, "PrecisionReducer", _FakeReducer)
    checkpoint_dir = tmp_path / "user-checkpoints"
    kw.common.CheckpointManager.save_dir = str(checkpoint_dir)
    model = SimpleNamespace(sampler_type="cim")

    qvae_helpers.init_qvae_sampler(model)
    qvae_helpers.init_qvae_sampler(model)

    assert kw.common.CheckpointManager.save_dir == str(checkpoint_dir)
    assert checkpoint_dir.is_dir()
    names = _task_names()
    assert names[0] != "qvae_sampling"
    assert names[0] != names[1]


def test_unset_checkpoint_dir_uses_a_real_temporary_directory(monkeypatch) -> None:
    """With no configured directory, construct a real temp dir instead of ./tmp."""
    _patch_factories(monkeypatch, rbm_digits)
    kw.common.CheckpointManager.save_dir = None

    rbm_digits.RBMRunner(use_cim=True)
    created_dir = kw.common.CheckpointManager.save_dir

    assert created_dir
    assert os.path.isdir(created_dir)
    assert os.path.abspath(created_dir) != os.path.abspath("./tmp")
