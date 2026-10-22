import os
import sys
import unittest

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

example_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/gbrbm"))
sys.path.insert(0, example_root)

import kaiwu

# Extend namespace package __path__ so that local src is preferred
kaiwu.__path__ = list(kaiwu.__path__) + [os.path.join(src_root, "kaiwu")]

for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith(
        "kaiwu.torch_plugin."
    ):
        del sys.modules[module_name]
if hasattr(kaiwu, "torch_plugin"):
    delattr(kaiwu, "torch_plugin")

import torch

import run_gbrbm


class TestMakeDataset(unittest.TestCase):
    """Dataset helper used by the GBRBM example."""

    def test_shape_and_dtype(self):
        data = run_gbrbm.make_dataset(n_sample=64)
        self.assertEqual(data.shape, (64, 4))
        self.assertEqual(data.dtype, torch.float32)

    def test_deterministic_with_seed(self):
        first = run_gbrbm.make_dataset(n_sample=32, seed=7)
        second = run_gbrbm.make_dataset(n_sample=32, seed=7)
        self.assertTrue(torch.equal(first, second))

    def test_mean_follows_design(self):
        # With enough samples the empirical mean approaches the designed one.
        data = run_gbrbm.make_dataset(n_sample=100000, seed=3)
        designed = torch.tensor([1.0, -0.5, 0.25, -1.25])
        self.assertTrue(torch.allclose(data.mean(dim=0), designed, atol=0.05))


class TestTraining(unittest.TestCase):
    """A tiny end-to-end training run of the example loop."""

    def setUp(self):
        torch.manual_seed(0)
        self.data = run_gbrbm.make_dataset(n_sample=32)
        self.bm = run_gbrbm.GaussianBernoulliRestrictedBoltzmannMachine(
            num_visible=4,
            num_hidden=3,
        )

    def test_training_returns_finite_history(self):
        history = run_gbrbm.train(
            self.bm, self.data, epochs=5, lr=0.05, log_every=0
        )
        self.assertEqual(len(history), 5)
        for value in history:
            self.assertTrue(torch.isfinite(torch.tensor(value)).item())

    def test_training_updates_parameters(self):
        mu_before = self.bm.mu.detach().clone()
        run_gbrbm.train(self.bm, self.data, epochs=5, lr=0.05, log_every=0)
        self.assertFalse(torch.equal(mu_before, self.bm.mu.detach()))


class TestGenerate(unittest.TestCase):
    """Generation helper used by the GBRBM example."""

    def setUp(self):
        torch.manual_seed(0)
        self.bm = run_gbrbm.GaussianBernoulliRestrictedBoltzmannMachine(
            num_visible=4,
            num_hidden=3,
        )

    def test_generated_shape_and_finiteness(self):
        samples = run_gbrbm.generate(
            self.bm, n_sample=8, n_step=6, n_burnin=3, seed=0
        )
        self.assertEqual(samples.ndim, 2)
        self.assertEqual(samples.shape[1], self.bm.num_nodes)
        self.assertTrue(torch.isfinite(samples).all().item())

    def test_deterministic_with_seed(self):
        first = run_gbrbm.generate(self.bm, n_sample=4, n_step=4, n_burnin=2, seed=1)
        second = run_gbrbm.generate(self.bm, n_sample=4, n_step=4, n_burnin=2, seed=1)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
