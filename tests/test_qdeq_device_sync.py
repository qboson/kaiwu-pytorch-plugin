"""Regression tests for QDEQ input device alignment.

``QDEQ`` wraps a user-supplied ``layer`` whose parameters may sit on a
different device than the caller's input tensor — for example after
``parent.qdeq = QDEQ(...)`` followed by ``parent.to(device)``. PyTorch moves
nested modules through ``_apply`` without calling child ``.to()`` overrides;
callers may therefore still hold CPU tensors while the layer already lives on
``meta`` or CUDA. ``QDEQ.forward`` must move ``x`` and an optional ``z0`` onto
the layer parameter device before the fixed-point solve and the final
differentiable ``layer(z*, x)`` call.

The CUDA-dependent tests run on any CUDA machine. ``meta`` and ``_apply`` cases
mirror the bookkeeping coverage in ``test_bm_device_sync.py`` (#164).
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


class RecordingLinearLayer(LinearLayer):
    """Linear layer that records the devices of tensors passed into ``forward``."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__(dim)
        self.x_devices: list[torch.device] = []
        self.z_devices: list[torch.device] = []

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Records devices, then returns ``W z + x``."""
        self.x_devices.append(x.device)
        self.z_devices.append(z.device)
        return super().forward(z, x)


class IteratedQDEQ(QDEQ):
    """QDEQ that reaches a fixed point by plain iteration (no Ising sampler)."""

    def solve_fixed_point(self, func, z0):
        """Iterates ``z <- func(z)`` until the map stabilises for these tests."""
        z = z0
        for _ in range(50):
            z = func(z)
        return z, {"nstep": 50}


class TestQDEQInputDeviceSync(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_parent_to_move_aligns_stale_cpu_input(self):
        """CPU ``x`` after ``parent.to(meta)`` must not leave outputs on CPU."""
        layer = RecordingLinearLayer()
        parent = nn.Module()
        parent.deq = IteratedQDEQ(layer, None, QDEQConfig())
        parent.to(torch.device("meta"))

        stale_x = torch.randn(2, 4)  # still reports CPU while params are on meta
        self.assertEqual(stale_x.device.type, "cpu")

        output = parent.deq(stale_x)

        self.assertEqual(
            output.device,
            parent.deq.layer.weight.device,
            "forward left activations on the stale input device",
        )
        self.assertTrue(
            all(device == layer.weight.device for device in layer.x_devices),
            "layer saw CPU activations during the solve or reattach call",
        )

    def test_parent_apply_move_aligns_stale_cpu_input(self):
        """``parent._apply`` is the path ``.to()`` uses; inputs must still align."""
        layer = RecordingLinearLayer()
        parent = nn.Module()
        parent.deq = IteratedQDEQ(layer, None, QDEQConfig())

        parent._apply(  # pylint: disable=protected-access
            lambda tensor: tensor.to(torch.device("meta"))
        )

        self.assertEqual(parent.deq.layer.weight.device, torch.device("meta"))
        output = parent.deq(torch.randn(2, 4))

        self.assertEqual(output.device, torch.device("meta"))
        self.assertTrue(all(device.type == "meta" for device in layer.x_devices))

    def test_stale_z0_moved_with_cpu_input(self):
        """An explicit ``z0`` on CPU must follow ``x`` onto the layer device."""
        layer = RecordingLinearLayer()
        deq = IteratedQDEQ(layer, None, QDEQConfig())
        deq._apply(  # pylint: disable=protected-access
            lambda tensor: tensor.to(torch.device("meta"))
        )

        stale_z0 = torch.ones(2, 4)
        self.assertEqual(stale_z0.device.type, "cpu")

        output = deq(torch.randn(2, 4), z0=stale_z0)

        self.assertEqual(output.device, torch.device("meta"))
        self.assertTrue(all(device.type == "meta" for device in layer.z_devices))

    def test_layer_without_parameters_uses_input_device(self):
        """``_layer_device`` falls back to ``x.device`` when the layer has no params."""

        class NoParamLayer(nn.Module):
            """Returns ``x`` unchanged; used only to inspect device selection."""

            def forward(self, _z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                """Returns ``x``."""
                return x

        class ImmediateSolveQDEQ(QDEQ):
            """Skips iteration and returns ``func(z0)`` for the no-param layer."""

            def solve_fixed_point(self, func, z0):
                return func(z0), {"nstep": 0}

        deq = ImmediateSolveQDEQ(NoParamLayer(), None, QDEQConfig()).eval()
        x = torch.randn(2, 3)
        with torch.no_grad():
            output = deq(x)

        self.assertEqual(output.device, x.device)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestQDEQInputDeviceSyncCuda(unittest.TestCase):
    """End-to-end coverage when parameters live on CUDA."""

    def test_cpu_input_with_cuda_parameters(self):
        """CUDA weights plus CPU ``x`` must run without a device-mismatch error."""
        parent = nn.Module()
        parent.deq = IteratedQDEQ(LinearLayer(), None, QDEQConfig())
        parent.to("cuda")

        output = parent.deq(torch.randn(2, 4))

        self.assertEqual(output.device.type, "cuda")

    def test_construct_on_cuda_move_parent_to_cpu(self):
        """Construct-on-CUDA / parent-move-to-CPU path must accept CPU logits."""
        parent = nn.Module()
        parent.deq = IteratedQDEQ(LinearLayer(), None, QDEQConfig())
        self.assertEqual(parent.deq.layer.weight.device.type, "cuda")

        parent.to("cpu")
        self.assertEqual(parent.deq.layer.weight.device.type, "cpu")

        output = parent.deq(torch.zeros(2, 4))
        self.assertEqual(output.device.type, "cpu")
        self.assertEqual(output.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
