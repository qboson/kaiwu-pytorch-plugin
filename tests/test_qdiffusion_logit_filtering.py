"""Contract tests for the QDiffusion logit filtering helper.

``top_k_top_p_filtering`` documents a pure transformation: it returns the
filtered logits and must leave the caller's tensor untouched. The tests below
pin that contract for contiguous, non-contiguous, and autograd inputs.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin._qdiffusion_sampling import top_k_top_p_filtering


class TestTopKTopPFilteringContract(unittest.TestCase):
    """The filter must behave like a function, not an in-place op."""

    def test_contiguous_input_is_not_mutated(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        original = logits.clone()

        filtered = top_k_top_p_filtering(logits, top_k=2, top_p=1.0)

        self.assertTrue(
            torch.equal(logits, original),
            "top_k_top_p_filtering must not modify the caller's logits in place",
        )
        # The returned values keep the documented filtering semantics.
        self.assertEqual(filtered[0, 0].item(), -float("inf"))
        self.assertEqual(filtered[0, 1].item(), -float("inf"))
        self.assertEqual(filtered[0, 2].item(), 3.0)
        self.assertEqual(filtered[0, 3].item(), 4.0)
        self.assertEqual(filtered[1, 2].item(), -float("inf"))
        self.assertEqual(filtered[1, 3].item(), -float("inf"))
        self.assertEqual(filtered[1, 0].item(), 4.0)

    def test_repeated_calls_on_the_same_tensor_are_stable(self):
        """A second pass with different settings must see the original logits."""
        logits = torch.tensor([[0.1, 5.0, 0.2, 4.0, 0.3]])
        reference = logits.clone()

        first = top_k_top_p_filtering(logits, top_k=2, top_p=1.0)
        second = top_k_top_p_filtering(logits, top_k=0, top_p=1.1)

        self.assertTrue(torch.equal(logits, reference))
        # First pass: only the two largest logits survive.
        self.assertTrue(torch.isneginf(first[0, [0, 2, 4]]).all())
        self.assertEqual(first[0, 1].item(), 5.0)
        self.assertEqual(first[0, 3].item(), 4.0)
        # Second pass on the untouched input must still see the small logits.
        self.assertEqual(second[0, 0].item(), reference[0, 0].item())
        self.assertEqual(second[0, 2].item(), reference[0, 2].item())

    def test_non_contiguous_input_is_not_mutated(self):
        base = torch.arange(24, dtype=torch.float).reshape(4, 6)
        strided = base[:, ::2]  # shape (4, 3), non-contiguous view
        original = strided.clone()

        filtered = top_k_top_p_filtering(strided, top_k=1, top_p=1.0)

        self.assertTrue(torch.equal(strided, original))
        kept = (~torch.isneginf(filtered[0])).sum().item()
        self.assertEqual(kept, 1)

    def test_autograd_input_is_supported(self):
        """Filtering must not crash on tensors that require gradient."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)

        filtered = top_k_top_p_filtering(logits, top_k=2, top_p=1.0)
        filtered.sum().backward()

        # Gradient flows to the kept entries and is zero on the filtered ones.
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 2].item(), 1.0)
        self.assertEqual(logits.grad[0, 3].item(), 1.0)
        self.assertEqual(logits.grad[0, 0].item(), 0.0)
        self.assertEqual(logits.grad[0, 1].item(), 0.0)

    def test_output_does_not_alias_the_input(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        filtered = top_k_top_p_filtering(logits, top_k=2, top_p=1.0)
        filtered[0, 3] = 999.0

        self.assertEqual(logits[0, 3].item(), 4.0)

    def test_top_p_only_path_leaves_input_untouched(self):
        logits = torch.randn(3, 5)
        original = logits.clone()

        filtered = top_k_top_p_filtering(logits, top_k=0, top_p=0.9)

        self.assertTrue(torch.equal(logits, original))
        self.assertEqual(filtered.shape, logits.shape)
        # Each row keeps at least its highest-logit entry.
        self.assertTrue((~torch.isneginf(filtered)).any(dim=-1).all())


if __name__ == "__main__":
    unittest.main()
