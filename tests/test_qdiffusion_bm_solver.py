"""Unit tests for the solver-backed BM reranker used by Q-Diffusion examples."""

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from example.qdiffusion.dplm.models import (  # noqa: E402
    BMConditionedEnergyAdapter,
    BMConditionedEnergyModel,
    DPLMFeatureEncoder,
)


class DummySampler:
    """Minimal solver stub that returns two all-one Ising solutions."""

    def solve(self, ising_mat):
        return np.ones((2, ising_mat.shape[0]), dtype=np.float32)


class FakeNet(nn.Module):
    """Tiny masked-LM-like module used to avoid external DPLM dependencies."""

    def __init__(self, vocab_size: int = 8, hidden_size: int = 6) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        del attention_mask
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        return {"last_hidden_state": torch.tanh(self.proj(hidden))}


class FakeBackbone:
    """Backbone wrapper exposing the attributes expected by DPLMFeatureEncoder."""

    def __init__(self, vocab_size: int = 8, hidden_size: int = 6) -> None:
        self.net = FakeNet(vocab_size=vocab_size, hidden_size=hidden_size)
        self.tokenizer = object()


class TestQDiffusionBMSolver(unittest.TestCase):
    """Covers the solver-backed BM reranker path with a fake encoder."""

    def setUp(self):
        backbone = FakeBackbone()
        self.encoder = DPLMFeatureEncoder(backbone)
        self.noisy_tokens = torch.tensor([[1, 2, 3], [1, 3, 2]], dtype=torch.long)
        self.candidate_tokens = torch.tensor(
            [[1, 4, 5], [1, 5, 4]], dtype=torch.long
        )
        self.attention_mask = torch.ones_like(self.noisy_tokens, dtype=torch.bool)

    def test_solver_mode_supports_backward_and_stats(self):
        model = BMConditionedEnergyModel(
            encoder=self.encoder,
            bm_num_visible=4,
            bm_num_hidden=3,
            sampler=DummySampler(),
            use_straight_through=True,
        )
        adapter = BMConditionedEnergyAdapter(model)
        energy = adapter.score_conditioned(
            noisy_tokens=self.noisy_tokens,
            candidate_tokens=self.candidate_tokens,
            attention_mask=self.attention_mask,
        )
        self.assertEqual(energy.shape, (2, 1))

        loss = energy.mean()
        loss.backward()

        self.assertIsNotNone(model.feature_projector.weight.grad)
        self.assertIsNotNone(model.energy_bm.linear_bias.grad)
        stats = adapter.get_last_stats()
        self.assertAlmostEqual(float(stats["solver_mode"].item()), 1.0)
        self.assertIn("visible_on_ratio", stats)
        self.assertIn("hidden_on_ratio", stats)

    def test_solver_mode_uses_binary_visible_path(self):
        model = BMConditionedEnergyModel(
            encoder=self.encoder,
            bm_num_visible=4,
            bm_num_hidden=3,
            sampler=DummySampler(),
        )
        adapter = BMConditionedEnergyAdapter(model)
        energy = adapter.score_conditioned(
            noisy_tokens=self.noisy_tokens,
            candidate_tokens=self.candidate_tokens,
            attention_mask=self.attention_mask,
        )
        self.assertEqual(energy.shape, (2, 1))
        stats = adapter.get_last_stats()
        self.assertAlmostEqual(float(stats["solver_mode"].item()), 1.0)

    def test_cim_sampler_path_prefers_optimizer_interface(self):
        class FakeOptimizer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def solve(self, ising_mat):
                return np.ones((1, ising_mat.shape[0]), dtype=np.float32)

        class FakePrecisionReducer:
            def __init__(self, sampler, **kwargs):
                self.sampler = sampler
                self.kwargs = kwargs

            def solve(self, ising_mat):
                return self.sampler.solve(ising_mat)

        fake_kw = types.SimpleNamespace(
            common=types.SimpleNamespace(
                CheckpointManager=types.SimpleNamespace(save_dir=None)
            )
        )
        fake_task_mode = types.SimpleNamespace(OPTIMIZATION="OPT", SAMPLING="SAMP")
        fake_kaiwu_cim = types.SimpleNamespace(
            CIMOptimizer=FakeOptimizer,
            Optimizer=FakeOptimizer,
            PrecisionReducer=FakePrecisionReducer,
            TaskMode=fake_task_mode,
        )

        with mock.patch.dict(
            sys.modules,
            {
                "kaiwu": fake_kw,
                "kaiwu.cim": fake_kaiwu_cim,
            },
        ):
            model = BMConditionedEnergyModel(
                encoder=self.encoder,
                bm_num_visible=4,
                bm_num_hidden=3,
                sampler=None,
                sampler_type="cim",
                sampler_kwargs={
                    "task_name": "unit-test",
                    "wait": True,
                    "interval": 3,
                    "task_mode": "OPTIMIZATION",
                    "tmp_dir": "tmp",
                },
            )
        self.assertEqual(model.sampler.kwargs["task_name"], "unit-test")
        self.assertEqual(model.sampler.kwargs["wait"], True)
        self.assertEqual(model.sampler.kwargs["interval"], 3)
        self.assertEqual(model.sampler.kwargs["task_mode"], "OPT")


if __name__ == "__main__":
    unittest.main()
