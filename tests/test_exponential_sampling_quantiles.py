"""Numerical regressions for the complete QVAE distribution utility module."""

import importlib.util
from decimal import Decimal, localcontext
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/kaiwu/torch_plugin/qvae_dist_util.py"
)
SPEC = importlib.util.spec_from_file_location("qvae_quantile_test_module", MODULE_PATH)
DIST_UTIL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIST_UTIL)


def decimal_reference(rate, draw):
    """Return inverse CDF and survival at 80-digit precision."""
    with localcontext() as context:
        context.prec = 80
        beta = Decimal.from_float(float(rate))
        rho = Decimal.from_float(float(draw))
        exp_negative_beta = (-beta).exp()
        survival = (1 - rho) + rho * exp_negative_beta
        quantile = -survival.ln() / beta
        return float(quantile), float(survival)


class ExponentialSamplingQuantileTests(unittest.TestCase):
    """Control only valid uniform draws; execute all sampling arithmetic in Torch."""

    @staticmethod
    def draws():
        one = torch.tensor(1.0, dtype=torch.float32)
        upper = torch.nextafter(one, torch.zeros_like(one)).item()
        return torch.tensor([0.0, 0.5, 0.9, 0.9995, upper], dtype=torch.float32)

    def sample_with_draws(self, distribution, draws):
        # This seam fixes RNG inputs, not the distribution or its arithmetic.
        self.assertTrue(((draws >= 0) & (draws < 1)).all().item())
        with patch.object(DIST_UTIL.torch, "rand", return_value=draws.clone()):
            samples = distribution.sample(tuple(draws.shape))
        self.assertEqual(samples.shape, draws.shape)
        self.assertEqual(samples.dtype, draws.dtype)
        self.assertEqual(samples.device, draws.device)
        self.assertTrue(torch.isfinite(samples).all().item())
        self.assertTrue(((samples >= 0) & (samples <= 1)).all().item())
        return samples

    def assert_quantiles(self, distribution, draws, samples):
        rates = torch.broadcast_to(distribution.beta, draws.shape)
        references = [
            decimal_reference(rate, draw)
            for rate, draw in zip(rates.flatten().tolist(), draws.flatten().tolist())
        ]
        expected = torch.tensor(
            [value[0] for value in references], dtype=torch.float64
        ).reshape(draws.shape)
        expected_survival = torch.tensor(
            [value[1] for value in references], dtype=torch.float64
        ).reshape(draws.shape)
        actual = samples.to(torch.float64)
        quantile_error = (actual - expected).abs().max().item()
        actual_survival = torch.exp(-rates.to(torch.float64) * actual)
        relative_survival_error = (
            (actual_survival - expected_survival).abs() / expected_survival
        ).max().item()
        self.assertLessEqual(quantile_error, 4e-7)
        self.assertLessEqual(relative_survival_error, 5e-6)
        self.assertTrue((samples[1:] >= samples[:-1]).all().item())

    def test_scalar_small_rates_and_upper_tail(self):
        draws = self.draws()
        for rate in (1e-8, 1e-4, 0.1, 1.0, 2.0, 10.0, 18.0, 24.0, 80.0):
            with self.subTest(rate=rate):
                distribution = DIST_UTIL.Exponential(rate)
                samples = self.sample_with_draws(distribution, draws)
                self.assert_quantiles(distribution, draws, samples)

    def test_mixed_rate_broadcast(self):
        distribution = DIST_UTIL.Exponential([[1e-8, 0.1, 18.0, 80.0]])
        draws = self.draws().unsqueeze(1).expand(5, 4).clone()
        samples = self.sample_with_draws(distribution, draws)
        self.assert_quantiles(distribution, draws, samples)

    def test_real_rng_replay(self):
        distribution = DIST_UTIL.Exponential(18.0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261111)
            state = torch.random.get_rng_state()
            draws = torch.rand((1024,))
            expected_state = torch.random.get_rng_state()
            torch.random.set_rng_state(state)
            samples = distribution.sample((1024,))
            self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_state))
        order = torch.argsort(draws)
        self.assert_quantiles(distribution, draws[order], samples[order])


if __name__ == "__main__":
    unittest.main()
