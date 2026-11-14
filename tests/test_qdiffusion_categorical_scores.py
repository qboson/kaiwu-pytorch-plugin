"""Check token/log-probability pairs from the QDiffusion sampling helper."""

import importlib.util
import math
from pathlib import Path
import unittest

import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "kaiwu" / "torch_plugin" / "_qdiffusion_sampling.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_qdiffusion_categorical_scores_under_test", _MODULE_PATH
)
_SAMPLING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SAMPLING)


class TestCategoricalScores(unittest.TestCase):
    def setUp(self):
        # At temperature 2 these weights become (1, 2, 3) and (4, 1, 2).
        self.logits = torch.tensor(
            [[math.log(weight) for weight in row]
             for row in ((1.0, 4.0, 9.0), (16.0, 1.0, 4.0))],
            dtype=torch.float64,
        )

    def test_greedy_returns_argmax_and_log_probability(self):
        tokens, scores = _SAMPLING.sample_from_categorical(
            self.logits, temperature=0
        )

        self.assertEqual(tokens.tolist(), [2, 0])
        self.assertEqual(tuple(scores.shape), (2,))
        for actual, probability in zip(scores.tolist(), (9.0 / 14.0, 16.0 / 21.0)):
            self.assertAlmostEqual(actual, math.log(probability), places=12)

    def test_sampled_score_matches_returned_token_at_nonunit_temperature(self):
        # Keep the real draw's CPU RNG effects local to this test.
        with torch.random.fork_rng(devices=[]):
            tokens, scores = _SAMPLING.sample_from_categorical(
                self.logits, temperature=2.0
            )

        self.assertEqual(tuple(tokens.shape), (2,))
        self.assertEqual(tuple(scores.shape), (2,))
        probabilities = ((1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0),
                         (4.0 / 7.0, 1.0 / 7.0, 2.0 / 7.0))
        for token, actual, row in zip(tokens.tolist(), scores.tolist(), probabilities):
            self.assertGreaterEqual(token, 0)
            self.assertLess(token, len(row))
            self.assertAlmostEqual(actual, math.log(row[token]), places=12)


if __name__ == "__main__":
    unittest.main()
