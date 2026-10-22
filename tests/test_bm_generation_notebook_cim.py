# -*- coding: utf-8 -*-
"""Notebook CIM snippets must follow the same checkpoint/task isolation."""

from __future__ import annotations

import json
import os
import tempfile
import textwrap
import uuid
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    REPO_ROOT / "example" / "bm_generation" / "train_bm.ipynb",
    REPO_ROOT / "example" / "bm_generation" / "sample_bm.ipynb",
)


class _FakeCheckpointManager:
    save_dir = None


class _FakeCIM:
    created: list["_FakeCIM"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeCIM.created.append(self)


class _FakeReducer:
    def __init__(self, sampler, **kwargs):
        self.sampler = sampler
        self.kwargs = kwargs


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _cim_block(path: Path) -> str:
    source = _notebook_source(path)
    start = source.index('if sampler_type == "cim":')
    end = source.index('elif sampler_type == "sa":', start)
    return textwrap.dedent(source[start:end])


def _run_cim_block(path: Path, checkpoint_dir: str) -> list[str]:
    _FakeCheckpointManager.save_dir = checkpoint_dir
    _FakeCIM.created.clear()
    namespace = {
        "sampler_type": "cim",
        "kw": SimpleNamespace(
            common=SimpleNamespace(CheckpointManager=_FakeCheckpointManager),
            cim=SimpleNamespace(
                CIMOptimizer=_FakeCIM,
                PrecisionReducer=_FakeReducer,
            ),
        ),
        "os": os,
        "tempfile": tempfile,
        "uuid": uuid,
    }
    exec(compile(_cim_block(path), str(path), "exec"), namespace)
    return [instance.kwargs["task_name"] for instance in _FakeCIM.created]


def test_notebooks_do_not_use_the_global_tmp_literal() -> None:
    """The fixed notebook snippets must not reintroduce the old literals."""
    for path in NOTEBOOKS:
        source = _notebook_source(path)
        assert "test_kpp" not in source
        assert "save_dir = './tmp'" not in source
        assert "tempfile.mkdtemp" in source
        assert "uuid.uuid4()" in source


def test_notebook_cim_blocks_preserve_dir_and_generate_tasks(tmp_path) -> None:
    """Executing the notebook CIM block keeps a configured dir and unique names."""
    checkpoint_dir = str(tmp_path / "user-checkpoints")
    for path in NOTEBOOKS:
        first = _run_cim_block(path, checkpoint_dir)
        second = _run_cim_block(path, checkpoint_dir)
        assert _FakeCheckpointManager.save_dir == checkpoint_dir
        assert Path(checkpoint_dir).is_dir()
        assert first and second
        assert first[0] != "test_kpp"
        assert first[0] != second[0]
