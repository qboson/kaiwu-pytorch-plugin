"""Contract test for the ``simple_generate_example.main`` call chain."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "example" / "qdiffusion" / "simple" / "simple_generate_example.py"


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, sequence, **kwargs):
        self.calls.append((sequence, kwargs))
        return {"input_ids": torch.tensor([[7, 8, 9]])}


class _Generator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.tokenizer = _Tokenizer()
        self.eval_called = False
        self.generate_calls: list[dict[str, object]] = []

    def to(self, device):
        self.target_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[1, 2, 3]])


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
            "qdiffusion_simple_generate_uut", SCRIPT
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


def test_generates_from_the_first_usable_sequence(monkeypatch, capsys):
    module = _load_script()
    generator = _Generator()
    build_calls: list[dict[str, object]] = []
    decode_calls: list[tuple[object, object]] = []

    module.build_qdiffusion = lambda **kwargs: (
        build_calls.append(kwargs) or generator
    )
    module.default_fasta_path = lambda: Path("/tmp/fake.fa")
    module.first_usable_sequence = lambda path, **kwargs: ("header-1", "M" * 60)
    module.decode_tokens = lambda gen, tokens: (
        decode_calls.append((gen, tokens)) or "MKV"
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    module.main()

    assert build_calls == [
        {
            "proposal_ckpt": "airkingbd/dplm_150m",
            "energy_ckpt": "airkingbd/dplm_150m",
            "num_candidates": 4,
            "proposal_temperature": 0.3,
            "energy_temperature": 1.25,
            "resample_ratio": 0.20,
            "resample_top_p": 0.90,
            "freeze_proposal": True,
        }
    ]
    assert generator.target_device == torch.device("cpu")
    assert generator.eval_called is True
    sequence, tokenizer_kwargs = generator.tokenizer.calls[0]
    assert sequence == "M" * 60
    assert tokenizer_kwargs == {"return_tensors": "pt", "add_special_tokens": True}
    assert len(generator.generate_calls) == 1
    generate_call = generator.generate_calls[0]
    assert set(generate_call) == {"input_tokens", "max_steps"}
    assert torch.equal(generate_call["input_tokens"], torch.tensor([[7, 8, 9]]))
    assert generate_call["max_steps"] == 5
    assert len(decode_calls) == 1
    assert decode_calls[0][0] is generator
    assert torch.equal(decode_calls[0][1], torch.tensor([[1, 2, 3]]))
    output = capsys.readouterr().out
    assert "header=header-1" in output
    assert "reference_length=60" in output
    assert "generated_length=3" in output
    assert "generated_sequence=MKV" in output
