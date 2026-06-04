import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin import EnergyModel


class DummySampler:
    def __init__(self, num_solutions: int = 2) -> None:
        self.num_solutions = num_solutions

    def solve(self, ising_mat):
        return np.ones((self.num_solutions, ising_mat.shape[0]), dtype=np.float32)


class TestEnergyModel(unittest.TestCase):
    def test_score_visible_logits_shape_and_stats(self):
        model = EnergyModel(
            bm_num_visible=2,
            bm_num_hidden=2,
            sampler=DummySampler(num_solutions=2),
        )
        visible_logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])

        energy = model.score_visible_logits(visible_logits)
        stats = model.get_last_stats()

        self.assertEqual(energy.shape, (2, 1))
        self.assertIn("sampling_mode", stats)
        self.assertIn("visible_on_ratio", stats)
        self.assertIn("hidden_on_ratio", stats)
        self.assertEqual(float(stats["sampling_mode"]), 1.0)

    def test_score_visible_logits_aggregates_sampler_solutions(self):
        model = EnergyModel(
            bm_num_visible=2,
            bm_num_hidden=2,
            sampler=DummySampler(num_solutions=3),
        )
        visible_logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])

        energy = model.score_visible_logits(visible_logits)
        visible_state = model.discretize_visible_state(visible_logits)
        full_states, split_sizes = model.sample_hidden_state(visible_state)
        flat_energy = model.energy_bm(full_states).unsqueeze(-1)
        expected = torch.stack(
            [part.mean(dim=0) for part in torch.split(flat_energy, split_sizes.tolist())],
            dim=0,
        )

        self.assertTrue(torch.equal(energy, expected))

    def test_discretize_visible_state_returns_sigmoid_probabilities(self):
        model = EnergyModel(
            bm_num_visible=3,
            bm_num_hidden=1,
            sampler=DummySampler(),
        )

        visible_logits = torch.tensor([[-2.0, 0.0, 2.0]])
        visible_state = model.discretize_visible_state(
            visible_logits
        )

        self.assertTrue(
            torch.allclose(visible_state, torch.sigmoid(visible_logits))
        )

    def test_discretize_visible_state_keeps_sigmoid_gradient(self):
        model = EnergyModel(
            bm_num_visible=2,
            bm_num_hidden=1,
            sampler=DummySampler(),
        )
        visible_logits = torch.tensor([[0.0, 2.0]], requires_grad=True)

        visible_state = model.discretize_visible_state(visible_logits)
        visible_state.sum().backward()

        self.assertTrue(
            torch.allclose(visible_state.detach(), torch.sigmoid(visible_logits.detach()))
        )
        self.assertIsNotNone(visible_logits.grad)
        self.assertTrue(torch.all(visible_logits.grad > 0))


if __name__ == "__main__":
    unittest.main()
