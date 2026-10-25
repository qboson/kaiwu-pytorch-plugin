"""Contract test for the ``simple_train_example.main`` loop."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "example" / "qdiffusion" / "simple" / "simple_train_example.py"


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, sequences, **kwargs):
        self.calls.append((sequences, kwargs))
        return {"input_ids": torch.zeros(len(sequences), 4, dtype=torch.long)}


class _Proposal:
    def eval(self) -> None:
        self.called = True


class _Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.ones(2))
        self.frozen = nn.Parameter(torch.ones(3), requires_grad=False)
        self.device = torch.device("cpu")
        self.tokenizer = _Tokenizer()
        self.proposal_model = _Proposal()
        self.objective_calls: list[dict[str, object]] = []
        self.train_calls = 0

    def to(self, device):
        self.target_device = device
        return self

    def train(self):
        self.train_calls += 1
        return self

    def objective(self, batch):
        self.objective_calls.append(batch)
        return {
            "energy_objective": torch.tensor([1.0], requires_grad=True),
            "logits": torch.zeros(2, 3),
            "loss_mask": torch.ones(2, 3),
        }


class _Optimizer:
    instances: list["_Optimizer"] = []

    def __init__(self, parameters, **kwargs):
        self.parameters = list(parameters)
        self.kwargs = kwargs
        self.events: list[str] = []
        _Optimizer.instances.append(self)

    def zero_grad(self, set_to_none=True):
        self.events.append(f"zero_grad({set_to_none})")

    def step(self):
        self.events.append("step")


def _load_script():
    names = (
        "_example_bootstrap",
        "dplm",
        "dplm.utils",
        "dplm.utils.dplm_builder",
    )
    saved = {name: sys.modules.get(name) for name in names}
    bootstrap = types.ModuleType("_example_bootstrap")
    bootstrap.ensure_repo_src_on_path = lambda: None
    dplm = types.ModuleType("dplm")
    dplm.__path__ = []
    dplm_utils = types.ModuleType("dplm.utils")
    dplm_utils.__path__ = []
    builder = types.ModuleType("dplm.utils.dplm_builder")
    builder.build_qdiffusion = lambda **kwargs: None
    sys.modules.update(
        {
            "_example_bootstrap": bootstrap,
            "dplm": dplm,
            "dplm.utils": dplm_utils,
            "dplm.utils.dplm_builder": builder,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "qdiffusion_simple_train_uut", SCRIPT
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_runs_three_objective_steps_on_the_selected_records(tmp_path, monkeypatch, capsys):
    module = _load_script()
    generator = _Generator()
    build_calls: list[dict[str, object]] = []

    def fake_build(**kwargs):
        build_calls.append(kwargs)
        return generator

    module.build_qdiffusion = fake_build
    module.default_fasta_path = lambda: Path("/tmp/fake.fa")
    module.read_fasta_records = lambda path: [
        ("h1", "A" * 60),
        ("h2", "B" * 30),
        ("h3", "C" * 70),
    ]
    module.AdamW = _Optimizer
    _Optimizer.instances.clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    module.main()

    assert build_calls == [
        {
            "proposal_ckpt": "airkingbd/dplm_150m",
            "energy_ckpt": "airkingbd/dplm_150m",
            "num_candidates": 4,
            "freeze_proposal": True,
        }
    ]
    assert generator.target_device == torch.device("cpu")
    sequences, tokenizer_kwargs = generator.tokenizer.calls[0]
    assert sequences == ["A" * 60, "C" * 70]
    assert tokenizer_kwargs == {
        "return_tensors": "pt",
        "add_special_tokens": True,
        "padding": True,
    }
    optimizer = _Optimizer.instances[0]
    assert optimizer.parameters == [generator.parameter]
    assert optimizer.kwargs == {"lr": 1e-4}
    assert generator.train_calls == 1
    assert generator.proposal_model.called is True
    assert len(generator.objective_calls) == 3
    targets = generator.objective_calls[0]["targets"]
    assert targets.shape == (2, 4)
    assert optimizer.events == [
        "zero_grad(True)",
        "step",
        "zero_grad(True)",
        "step",
        "zero_grad(True)",
        "step",
    ]
    assert "step=3" in capsys.readouterr().out
