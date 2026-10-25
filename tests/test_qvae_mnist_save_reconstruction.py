"""Contract test for ``Trainer._save_reconstruction``."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "example" / "qvae_mnist" / "trainer" / "trainer.py"


def _load_method():
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Trainer"
    )
    method = next(
        node
        for node in trainer_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_save_reconstruction"
    )
    namespace = {"os": os, "plot_MNIST_output": None}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(TRAINER), "exec"),
        namespace,
    )
    return namespace


def _recorder(namespace):
    calls: list[tuple[np.ndarray, np.ndarray, str]] = []

    def plot(x_true, x_recon, output=None):
        calls.append((x_true, x_recon, output))

    namespace["plot_MNIST_output"] = plot
    return calls


def test_slices_tensors_to_ten_examples_and_builds_the_epoch_path(tmp_path):
    namespace = _load_method()
    calls = _recorder(namespace)
    trainer = SimpleNamespace(output_dir=str(tmp_path))
    inputs = torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)
    recon = inputs + 100.0

    namespace["_save_reconstruction"](trainer, 7, inputs, recon)

    assert len(calls) == 1
    x_true, x_recon, output = calls[0]
    assert isinstance(x_true, np.ndarray)
    assert x_true.shape == (10, 4)
    assert x_recon.shape == (10, 4)
    assert np.allclose(x_true, inputs[:10].numpy())
    assert output == os.path.join(str(tmp_path), "reconstruction_epoch_7.png")


def test_accepts_list_outputs_and_uses_their_first_entry(tmp_path):
    namespace = _load_method()
    calls = _recorder(namespace)
    trainer = SimpleNamespace(output_dir=str(tmp_path))
    inputs = [torch.arange(3 * 2, dtype=torch.float32).reshape(3, 2)]
    recon = [inputs[0] + 1.0]

    namespace["_save_reconstruction"](trainer, 2, inputs, recon)

    x_true, x_recon, output = calls[0]
    assert x_true.shape == (3, 2)
    assert x_recon.shape == (3, 2)
    assert np.allclose(x_recon, recon[0].numpy())
    assert output.endswith("reconstruction_epoch_2.png")
