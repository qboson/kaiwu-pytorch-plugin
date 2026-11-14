"""Check nucleus prefixes, multidimensional logits and vocabulary clamping."""
import importlib.util
from pathlib import Path
import unittest

import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "kaiwu" / "torch_plugin" / "_qdiffusion_sampling.py"
)
_SPEC = importlib.util.spec_from_file_location("_qdiffusion_filtering_under_test", _MODULE_PATH)
_SAMPLING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SAMPLING)


_LOGITS_3D = (
    ((1.0, 4.0, 2.0, 3.0), (6.0, 2.0, 5.0, 1.0), (-3.0, -1.0, -4.0, -2.0)),
    ((0.5, -0.5, 2.5, 1.5), (9.0, 7.0, 8.0, 6.0), (0.0, -2.0, 3.0, 1.0)),
)


class TestFiltering(unittest.TestCase):
    def test_equal_probabilities_keep_strict_boundary(self):
        logits = torch.zeros((1, 4), dtype=torch.float64)
        filtered = _SAMPLING.top_k_top_p_filtering(logits, top_k=0, top_p=0.5)

        finite = torch.isfinite(filtered)
        self.assertEqual(finite.sum().item(), 3)
        self.assertEqual(torch.isneginf(filtered).sum().item(), 1)
        self.assertEqual(filtered[finite].tolist(), [0.0, 0.0, 0.0])

    def test_peaked_prefix_and_disabled_filter(self):
        for top_p, expected in (
            (0.9, [[2.0, 1.0, 0.0, -float("inf")]]),
            (1.0, [[2.0, 1.0, 0.0, -1.0]]),
        ):
            with self.subTest(top_p=top_p):
                logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]], dtype=torch.float64)
                filtered = _SAMPLING.top_k_top_p_filtering(logits, top_k=0, top_p=top_p)
                torch.testing.assert_close(
                    filtered, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0
                )

    def test_multidimensional_top_k_preserves_shape_and_rows(self):
        logits = torch.tensor(_LOGITS_3D, dtype=torch.float64)
        filtered = _SAMPLING.top_k_top_p_filtering(logits, top_k=2, top_p=1.0)

        self.assertEqual(tuple(filtered.shape), (2, 3, 4))
        self.assertEqual(torch.isfinite(filtered).sum(dim=-1).tolist(), [[2, 2, 2], [2, 2, 2]])
        negative_infinity = -float("inf")
        expected = torch.tensor(
            [
                [[negative_infinity, 4.0, negative_infinity, 3.0],
                 [6.0, negative_infinity, 5.0, negative_infinity],
                 [negative_infinity, -1.0, negative_infinity, -2.0]],
                [[negative_infinity, negative_infinity, 2.5, 1.5],
                 [9.0, negative_infinity, 8.0, negative_infinity],
                 [negative_infinity, negative_infinity, 3.0, 1.0]],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(filtered, expected, rtol=0, atol=0)

    def test_multidimensional_disabled_filter_preserves_values(self):
        logits = torch.tensor(_LOGITS_3D, dtype=torch.float64)
        expected = torch.tensor(_LOGITS_3D, dtype=torch.float64)
        filtered = _SAMPLING.top_k_top_p_filtering(logits, top_k=0, top_p=1.0)

        self.assertEqual(tuple(filtered.shape), (2, 3, 4))
        torch.testing.assert_close(filtered, expected, rtol=0, atol=0)

    def test_top_k_above_vocabulary_preserves_values(self):
        logits = torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float64)
        filtered = _SAMPLING.top_k_top_p_filtering(logits, top_k=99, top_p=1.0)
        expected = torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float64)

        torch.testing.assert_close(filtered, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
