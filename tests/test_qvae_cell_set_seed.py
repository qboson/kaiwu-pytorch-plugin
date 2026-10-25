"""Contract test for ``example/qvae_cell/trainer.set_seed``."""

from __future__ import annotations

import ast
import random
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "example" / "qvae_cell" / "trainer.py"


def _load_set_seed():
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "set_seed"
    )
    namespace = {"random": random, "np": np, "torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(TRAINER), "exec"),
        namespace,
    )
    return namespace["set_seed"]


def test_seeds_python_numpy_and_torch_reproducibly():
    set_seed = _load_set_seed()

    set_seed(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    set_seed(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert first == second


def test_different_seeds_produce_different_torch_streams():
    set_seed = _load_set_seed()

    set_seed(11)
    first = float(torch.rand(1))
    set_seed(12)

    assert float(torch.rand(1)) != first


def test_cuda_branch_reseeds_every_device(monkeypatch):
    set_seed = _load_set_seed()
    recorded: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", recorded.append)

    set_seed(7)

    assert recorded and set(recorded) == {7}
