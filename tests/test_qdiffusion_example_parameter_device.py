"""Q-Diffusion examples must follow parameter devices, not cached metadata.

Upstream ``QDiffusion`` stores ``device`` as plain metadata (see open PR #488).
When a parent ``nn.Module`` moves the generator through ``_apply``, parameters
relocate but ``generator.device`` can stay stale. Example helpers therefore use
``parameter_device`` instead of the cache.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


def _load_runtime():
    runtime_path = (
        Path(__file__).resolve().parents[1]
        / "example/qdiffusion/dplm/utils/runtime.py"
    )
    spec = importlib.util.spec_from_file_location("qdiff_runtime", runtime_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TinyProposal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.size(0)
        return self.weight.unsqueeze(0).expand(batch, -1, -1)[:, :, :8]


class _TinyEnergy(EnergyModel):
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        return torch.zeros(noisy_tokens.size(0), 1, device=noisy_tokens.device)


class TestQDiffusionExampleParameterDevice(unittest.TestCase):
    """Regression coverage for example runtime device resolution."""

    def setUp(self) -> None:
        self.runtime = _load_runtime()
        token_spec = SequenceTokenSpec(
            mask_id=0,
            pad_id=1,
            bos_id=2,
            eos_id=3,
            tokenizer=lambda text, **kwargs: {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)
            },
        )
        generator = QDiffusion(
            _TinyProposal(),
            _TinyEnergy(),
            token_spec,
            QDiffusionConfig(),
        )
        wrapper = nn.Module()
        wrapper.generator = generator
        wrapper._apply(lambda tensor: tensor.to(torch.device("meta")))
        self.generator = generator

    def test_parameter_device_tracks_parent_moves(self) -> None:
        param_device = self.runtime.parameter_device(self.generator)
        self.assertEqual(param_device, torch.device("meta"))
        self.assertNotEqual(
            self.generator.device,
            param_device,
            "cached generator.device should be stale in this scenario",
        )

    def test_encode_sequence_uses_parameter_device(self) -> None:
        tokens = self.runtime.encode_sequence(self.generator, "ACDE")
        self.assertEqual(tokens.device, torch.device("meta"))


if __name__ == "__main__":
    unittest.main()
