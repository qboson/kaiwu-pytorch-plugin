"""Constructor dtype contract tests for the QDiffusion wrapper.

``QDiffusion(dtype=..., device=None)`` recorded the requested dtype but never
converted the submodules, so ``self.dtype`` disagreed with the parameters and
downstream code that trusted ``self.dtype`` (or the proposal logits) silently
mixed precisions. The tests below pin the constructor contract: the requested
dtype is applied whether or not a device is given.
"""

import os
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin import EnergyModel, QDiffusion, QDiffusionConfig
from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec


class DummyProposal(nn.Module):
    def __init__(self, vocab_size=8, hidden_size=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens):
        flat = self.embedding(tokens.reshape(-1))
        logits = self.proj(flat)
        return logits.reshape(*tokens.shape, -1)


class DummyEnergy(EnergyModel):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        scores = candidate_tokens.sum(-1).to(self.scale.dtype)
        return (scores * self.scale).unsqueeze(-1)


def _build_model(**kwargs):
    return QDiffusion(
        proposal_model=DummyProposal(),
        energy_model=DummyEnergy(),
        token_spec=SequenceTokenSpec(mask_id=0, pad_id=1, bos_id=2, eos_id=3),
        **kwargs,
    )


class TestQDiffusionConstructorDtype(unittest.TestCase):
    """The requested dtype must be applied for every device argument form."""

    def _assert_module_dtypes(self, model, expected):
        for name, parameter in model.named_parameters():
            self.assertEqual(
                parameter.dtype,
                expected,
                f"parameter {name} has dtype {parameter.dtype}, expected {expected}",
            )

    def test_dtype_is_applied_when_device_is_omitted(self):
        model = _build_model(dtype=torch.float64)

        self.assertEqual(model.dtype, torch.float64)
        self._assert_module_dtypes(model, torch.float64)

    def test_dtype_is_applied_when_device_is_given(self):
        model = _build_model(dtype=torch.float64, device="cpu")

        self.assertEqual(model.dtype, torch.float64)
        self._assert_module_dtypes(model, torch.float64)

    def test_default_construction_stays_float32(self):
        model = _build_model()

        self.assertEqual(model.dtype, torch.float32)
        self._assert_module_dtypes(model, torch.float32)

    def test_explicit_float32_matches_default(self):
        model = _build_model(dtype=torch.float32)

        self.assertEqual(model.dtype, torch.float32)
        self._assert_module_dtypes(model, torch.float32)

    def test_converted_model_still_generates(self):
        model = _build_model(dtype=torch.float64, config=QDiffusionConfig(
            num_diffusion_timesteps=4,
        ))
        tokens = torch.tensor([[5, 6, 7, 4, 5]])

        generated = model.generate(tokens, max_steps=2)

        self.assertEqual(generated.shape, tokens.shape)
        self.assertTrue((generated >= 0).all() and (generated < 8).all())

    def test_energy_scores_after_conversion(self):
        model = _build_model(dtype=torch.float64)
        tokens = torch.tensor([[5, 6, 7]])

        energy = model.energy(tokens, tokens)

        self.assertEqual(energy.dtype, torch.float64)
        self.assertTrue(torch.isfinite(energy).all())


if __name__ == "__main__":
    unittest.main()
