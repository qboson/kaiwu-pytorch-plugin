"""Regression tests for RBM state propagation on the parameter device.

``RestrictedBoltzmannMachine.get_hidden`` and ``get_visible`` build full-state
buffers and run matmuls against the RBM parameters. Callers often supply CPU
visible/hidden tensors even after a parent module relocates the RBM weights
through ``nn.Module._apply`` (the same path covered for Boltzmann machines in
merged **#164** / ``tests/test_bm_device_sync.py``). Buffers and inputs must
therefore follow the **parameter** device, not a possibly stale ``self.device``
cache left over from construction or manual bookkeeping.
"""

# Parent-initiated moves are exercised through ``_apply`` (see #164).
# pylint: disable=protected-access

import unittest

import torch
from torch import nn

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


class RecordingRestrictedBoltzmannMachine(RestrictedBoltzmannMachine):
    """RBM that records every direct ``.to()`` invocation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.direct_to_calls: list[tuple[tuple, dict]] = []

    def to(self, *args, **kwargs):
        """Record ``.to()`` and delegate to ``nn.Module.to``."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


class TestRbmStatePropagationMechanism(unittest.TestCase):
    """Meta-tensor mechanism tests that run on every machine."""

    def test_stale_device_cache_still_propagates_on_parameter_device(self):
        """``_parameter_device()`` must win over a manually stale ``self.device``."""
        rbm = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2, device="cpu")
        rbm._apply(lambda tensor: tensor.to(torch.device("meta")))
        # ``_apply`` normally refreshes ``self.device``; force a stale cache to
        # prove propagation follows parameters, not the attribute.
        rbm.device = torch.device("cpu")

        hidden_state = rbm.get_hidden(torch.ones(4, 3))

        self.assertEqual(rbm.device, torch.device("cpu"))
        self.assertEqual(hidden_state.device, rbm.quadratic_coef.device)
        self.assertEqual(hidden_state.device.type, "meta")
        self.assertEqual(hidden_state.shape, (4, 5))

    def test_parent_apply_move_does_not_require_child_to(self):
        """Parent ``_apply`` must relocate states without calling child ``.to()``."""
        parent = nn.Module()
        parent.rbm = RecordingRestrictedBoltzmannMachine(num_visible=3, num_hidden=2)

        parent._apply(lambda tensor: tensor.to(torch.device("meta")))

        self.assertEqual(parent.rbm.direct_to_calls, [])
        hidden_state = parent.rbm.get_hidden(torch.ones(2, 3))
        visible_state = parent.rbm.get_visible(torch.ones(2, 2))

        parameter_device = parent.rbm.quadratic_coef.device
        self.assertEqual(hidden_state.device, parameter_device)
        self.assertEqual(visible_state.device, parameter_device)

    def test_get_hidden_aligns_cpu_inputs_after_apply_move(self):
        """CPU caller tensors must be coerced before buffer assignment and matmul."""
        rbm = RestrictedBoltzmannMachine(num_visible=4, num_hidden=3, device="cpu")
        rbm._apply(lambda tensor: tensor.to(torch.device("meta")))

        hidden_state = rbm.get_hidden(torch.ones(5, 4))

        self.assertEqual(hidden_state.device.type, "meta")
        self.assertEqual(hidden_state.shape, (5, 7))

    def test_get_visible_aligns_cpu_inputs_after_apply_move(self):
        """``get_visible`` must mirror the hidden-path device contract."""
        rbm = RestrictedBoltzmannMachine(num_visible=4, num_hidden=3, device="cpu")
        rbm._apply(lambda tensor: tensor.to(torch.device("meta")))

        visible_state = rbm.get_visible(torch.ones(5, 3))

        self.assertEqual(visible_state.device.type, "meta")
        self.assertEqual(visible_state.shape, (5, 7))

    def test_bernoulli_propagation_runs_on_parameter_device(self):
        """Bernoulli draws must be taken on the same device as the parameters."""
        rbm = RestrictedBoltzmannMachine(num_visible=2, num_hidden=2, device="cpu")
        rbm._apply(lambda tensor: tensor.to(torch.device("meta")))

        hidden_state = rbm.get_hidden(
            torch.ones(3, 2),
            bernoulli=True,
        )

        self.assertEqual(hidden_state.device, rbm.quadratic_coef.device)
        self.assertEqual(hidden_state.shape, (3, 4))

    def test_propagation_output_feeds_forward_without_device_mismatch(self):
        """End-to-end: propagated states must score on the parameter device."""
        rbm = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2, device="cpu")
        rbm._apply(lambda tensor: tensor.to(torch.device("meta")))

        state = rbm.get_hidden(torch.ones(2, 3))
        energy = rbm.forward(state)

        self.assertEqual(energy.device, rbm.quadratic_coef.device)
        self.assertEqual(energy.shape, (2,))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestRbmStatePropagationCuda(unittest.TestCase):
    """Accelerator coverage for nested modules moved back to CPU."""

    def test_nested_rbm_get_hidden_after_parent_move_to_cpu(self):
        """CUDA-constructed RBMs must propagate after ``parent.to('cpu')``."""
        parent = nn.Module()
        parent.rbm = RestrictedBoltzmannMachine(num_visible=4, num_hidden=3)
        self.assertEqual(
            parent.rbm.quadratic_coef.device.type,
            "cuda",
            "construction should auto-select CUDA",
        )

        parent.to("cpu")

        hidden_state = parent.rbm.get_hidden(torch.zeros(2, 4))

        self.assertEqual(parent.rbm.quadratic_coef.device.type, "cpu")
        self.assertEqual(hidden_state.device.type, "cpu")
        self.assertEqual(hidden_state.shape, (2, 7))

    def test_get_visible_accepts_cuda_inputs_when_model_on_cuda(self):
        """Visible propagation must accept accelerator-resident hidden states."""
        rbm = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
        self.assertEqual(rbm.quadratic_coef.device.type, "cuda")

        visible_state = rbm.get_visible(
            torch.ones(4, 2, device=rbm.quadratic_coef.device)
        )

        self.assertEqual(visible_state.device.type, "cuda")
        self.assertEqual(visible_state.shape, (4, 5))


if __name__ == "__main__":
    unittest.main()
