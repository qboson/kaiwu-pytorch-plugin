"""Contract test for the ``Trainer.__init__`` output-directory handling."""

from __future__ import annotations

import ast
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "example" / "qvae_mnist" / "trainer" / "trainer.py"


def _load_init():
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Trainer"
    )
    init = next(
        node
        for node in trainer_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    namespace = {"os": os, "datetime": datetime}
    exec(
        compile(ast.Module(body=[init], type_ignores=[]), str(TRAINER), "exec"),
        namespace,
    )
    return namespace["__init__"]


def _config(output_dir=None, model_type="QVAE"):
    return SimpleNamespace(output_dir=output_dir, type=model_type)


def test_uses_and_creates_the_configured_output_directory(tmp_path):
    init = _load_init()
    trainer = SimpleNamespace()
    output_dir = tmp_path / "explicit"

    init(
        trainer,
        config=_config(str(output_dir)),
        custom_train_data=("train",),
        custom_test_data=("test",),
    )

    assert trainer.output_dir == str(output_dir)
    assert output_dir.is_dir()
    assert trainer.config.type == "QVAE"
    assert trainer.custom_train_data == ("train",)
    assert trainer.custom_test_data == ("test",)
    assert trainer.model is None
    assert trainer.tuner is None
    assert trainer.train_loader is None
    assert trainer.test_loader is None
    assert trainer.dataset_mean is None
    assert trainer.train_losses == []
    assert trainer.test_losses == []


def test_derives_a_timestamped_directory_when_none_is_configured(
    tmp_path, monkeypatch
):
    init = _load_init()
    trainer = SimpleNamespace()
    monkeypatch.chdir(tmp_path)

    init(trainer, config=_config(None, model_type="CellQVAE"))

    assert trainer.output_dir.startswith("./output/CellQVAE_")
    suffix = trainer.output_dir.rsplit("_", 1)[-1]
    assert len(suffix) == 6 and suffix.isdigit()
    stamp = trainer.output_dir.rsplit("/", 1)[-1].split("_", 1)[1]
    datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    assert Path(trainer.output_dir).is_dir()
