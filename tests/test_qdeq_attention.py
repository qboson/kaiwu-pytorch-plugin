"""Higher-order QDEQ training with real PyTorch attention."""
import copy
import unittest

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class AttentionLayer(torch.nn.Module):
    """Small contractive attention map with trainable projections."""

    def __init__(self):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(8, 2, batch_first=True)
        self.fail = False
        self.backends = []

    def forward(self, z, x):
        """Uses the same attention route as the ETTh1 QDEQ example."""
        if torch.is_grad_enabled():
            self.backends.append(backend_flags())
            if self.fail:
                raise RuntimeError("injected final layer failure")
        h = 0.1 * z + x
        return 0.1 * self.attention(h, h, h, need_weights=False)[0] + x


class IteratedQDEQ(QDEQ):
    """Uses the documented custom forward solver extension, without a license."""

    def solve_fixed_point(self, func, z0):
        """Finds this small contraction's equilibrium by plain iteration."""
        z = z0
        for _ in range(12):
            z = func(z)
        return z, {"nstep": 12}


def backend_flags():
    """Returns all caller SDPA backend flags."""
    return tuple(getter() for getter in (
        torch.backends.cuda.flash_sdp_enabled,
        torch.backends.cuda.math_sdp_enabled,
        torch.backends.cuda.mem_efficient_sdp_enabled,
        torch.backends.cuda.cudnn_sdp_enabled,
    ))


class TestAttentionRegulariser(unittest.TestCase):
    """Regularisation can backpropagate through the attention Jacobian."""

    def setUp(self):
        torch.manual_seed(17)
        self.layer = AttentionLayer()
        self.x = torch.randn(2, 4, 8)

    def train_step(self, layer, x):
        """Backpropagates output MSE and the documented Jacobian penalty."""
        deq = IteratedQDEQ(layer, None, QDEQConfig(jac_loss_vecs=1, b_thres=25))
        x = x.detach().clone().requires_grad_()
        torch.manual_seed(23)
        out = deq(x)
        loss = out.square().mean() + 0.1 * deq.last_jac_loss
        loss.backward()
        grads = [x.grad] + [p.grad for p in layer.parameters()]
        for grad in grads:
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
        return out.detach(), deq.last_jac_loss.detach(), grads

    def test_regularised_attention_matches_math_reference(self):
        """Fused-backend callers get finite gradients matching explicit math SDPA."""
        reference_layer = copy.deepcopy(self.layer)
        before = backend_flags()
        actual = self.train_step(self.layer, self.x)
        self.assertEqual(backend_flags(), before)
        with sdpa_kernel(SDPBackend.MATH):
            expected = self.train_step(reference_layer, self.x)
        self.assertEqual(backend_flags(), before)
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-7)
        for actual_grad, expected_grad in zip(actual[2], expected[2]):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)

    def test_default_training_and_eval_preserve_backend_selection(self):
        """No higher-order attention restriction applies without regularisation."""
        before = backend_flags()
        for regulariser, training in ((0, True), (1, False)):
            layer = copy.deepcopy(self.layer)
            deq = IteratedQDEQ(layer, None, QDEQConfig(jac_loss_vecs=regulariser))
            deq.train(training)
            deq(self.x)
            self.assertEqual(layer.backends, [before])
            self.assertEqual(backend_flags(), before)

    def test_backend_restored_after_final_layer_exception(self):
        """The scoped mathematical backend restores caller state on failure."""
        before = backend_flags()
        self.layer.fail = True
        deq = IteratedQDEQ(self.layer, None, QDEQConfig(jac_loss_vecs=1))
        with self.assertRaisesRegex(RuntimeError, "injected final layer failure"):
            deq(self.x)
        self.assertEqual(backend_flags(), before)


if __name__ == "__main__":
    unittest.main()
