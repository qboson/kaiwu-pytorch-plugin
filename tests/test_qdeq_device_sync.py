"""Regression tests for QDEQ input device alignment.

``QDEQ`` wraps a user-supplied ``layer`` whose parameters may sit on a
different device than the caller's input tensor — for example after
``parent.qdeq = QDEQ(...)`` followed by ``parent.to(device)``. Unlike
``nn.Linear``, ``QDEQ.forward`` must move ``x`` (and an optional ``z0``) onto
the layer's parameter device before solving and before the final differentiable
call.

The CUDA case is skipped on CPU-only machines. The ``meta`` device case runs
everywhere and mirrors the parent-move bookkeeping tests in
``test_bm_device_sync.py``.
"""

import unittest

import torch
from torch import nn

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class LinearLayer(nn.Module):
    """Contractive map ``layer(z, x) = W z + x`` for fixed-point tests."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dim) * 0.1)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Returns ``W z + x``."""
        return z @ self.weight.T + x


class IteratedQDEQ(QDEQ):
    """QDEQ that reaches a fixed point by plain iteration (no Ising sampler)."""

    def solve_fixed_point(self, func, z0):
        z = z0
        for _ in range(50):
            z = func(z)
        return z, {"nstep": 50}


class TestQDEQInputDeviceSync(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_parent_move_aligns_cpu_input_with_layer_device(self):
        """Output must live on the layer device after a parent-initiated move."""
        parent = nn.Module()
        parent.deq = IteratedQDEQ(LinearLayer(), None, QDEQConfig())
        parent.to(torch.device("meta"))

        output = parent.deq(torch.randn(2, 4))

        self.assertEqual(
            output.device,
            parent.deq.layer.weight.device,
            "forward left activations on the input device instead of the layer",
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestQDEQInputDeviceSyncCuda(unittest.TestCase):
    """End-to-end coverage when parameters were constructed on CUDA."""

    def test_cpu_input_after_cuda_parent_move(self):
        """A CPU tensor must be moved before ``layer(z, x)`` runs on CUDA."""
        parent = nn.Module()
        parent.deq = IteratedQDEQ(LinearLayer(), None, QDEQConfig())
        parent.to("cuda")

        output = parent.deq(torch.randn(2, 4))

        self.assertEqual(output.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
