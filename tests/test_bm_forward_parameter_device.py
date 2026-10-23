"""Regression tests for full BoltzmannMachine forward device alignment.

``BoltzmannMachine.forward`` evaluates the Hamiltonian. Callers often pass
activations that still live on CPU while parameters were placed on an
accelerator, or were moved by a parent ``_apply``. Merged #164 established
that ``nn.Module.to`` does not call a child's ``.to`` override, so device
behavior has to follow the parameters themselves.

``_model_device()`` is that source of truth. These tests keep the CPU
``meta`` mechanism from ``tests/test_bm_device_sync.py`` and add the same
CUDA ``@skipUnless`` split for the construct-on-CUDA / move-to-CPU path.

Open #1268 already coerces RBM ``forward`` and the sampling helpers. Open
#1345 fixes Ising export buffers. This file only covers full BM ``forward``
and ``objective`` (which calls ``forward``).
"""

# Parent-initiated moves go through ``_apply`` (see #164).
# pylint: disable=protected-access

import unittest

import torch
from torch import nn

from kaiwu.torch_plugin import BoltzmannMachine
from kaiwu.torch_plugin.abstract_boltzmann_machine import AbstractBoltzmannMachine


class ParameterFreeMachine(AbstractBoltzmannMachine):  # pylint: disable=abstract-method
    """Stand-in with no parameters, covering the ``_model_device`` fallback."""

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """Return ``s_all`` unchanged."""
        return s_all


class RecordingBoltzmannMachine(BoltzmannMachine):
    """BoltzmannMachine that records every direct ``.to()`` invocation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.direct_to_calls: list[tuple[tuple, dict]] = []

    def to(self, *args, **kwargs):
        """Record the call, then perform the real move."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def _cpu_hamiltonian(bm: BoltzmannMachine, spins: torch.Tensor) -> torch.Tensor:
    """Hamiltonian of ``spins`` against a CPU copy of ``bm`` weights."""
    reference = BoltzmannMachine(
        bm.num_nodes,
        quadratic_coef=bm.quadratic_coef.detach().to(device="cpu").clone(),
        linear_bias=bm.linear_bias.detach().to(device="cpu").clone(),
        device="cpu",
    )
    return reference(spins.detach().to(device="cpu"))


class TestBoltzmannMachineForwardParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine, including CPU-only CI."""

    def test_model_device_falls_back_without_parameters(self):
        """Parameter-free modules report ``self.device``."""
        machine = ParameterFreeMachine(device=torch.device("cpu"))

        self.assertEqual(machine._model_device(), torch.device("cpu"))

        machine.device = torch.device("meta")
        self.assertEqual(machine._model_device(), torch.device("meta"))

    def test_forward_after_apply_move_accepts_cpu_batch(self):
        """CPU caller tensors must score once parameters were moved via ``_apply``."""
        bm = BoltzmannMachine(4, device="cpu")
        bm._apply(lambda tensor: tensor.to(torch.device("meta")))
        spins = torch.ones(2, 4)

        energy = bm(spins)

        self.assertEqual(spins.device, torch.device("cpu"))
        self.assertEqual(energy.shape, (2,))
        self.assertEqual(
            energy.device,
            torch.device("meta"),
            "forward must place the Hamiltonian on the parameter device",
        )

    def test_forward_after_parent_move_accepts_cpu_batch(self):
        """A parent ``_apply`` must not need the child's ``.to()`` override."""
        parent = nn.Module()
        parent.bm = RecordingBoltzmannMachine(4, device="cpu")
        parent._apply(lambda tensor: tensor.to(torch.device("meta")))

        energy = parent.bm(torch.zeros(3, 4))

        self.assertEqual(parent.bm.direct_to_calls, [])
        self.assertEqual(energy.shape, (3,))
        self.assertEqual(energy.device, torch.device("meta"))
        self.assertEqual(
            parent.bm._model_device(),
            parent.bm.quadratic_coef.device,
        )

    def test_forward_follows_parameters_when_device_cache_is_stale(self):
        """A manually stale ``self.device`` must not choose the work device."""
        bm = BoltzmannMachine(4, device="cpu")
        bm._apply(lambda tensor: tensor.to(torch.device("meta")))
        bm.device = torch.device("cpu")

        energy = bm(torch.ones(2, 4))

        self.assertEqual(bm.device, torch.device("cpu"))
        self.assertEqual(bm._model_device(), torch.device("meta"))
        self.assertEqual(energy.shape, (2,))
        self.assertEqual(energy.device, torch.device("meta"))

    def test_forward_matches_hand_computed_hamiltonian(self):
        """Coercion on an already-correct device must keep the Hamiltonian and grads."""
        bm = BoltzmannMachine(
            2,
            quadratic_coef=torch.tensor([[0.0, 4.0], [4.0, 0.0]]),
            linear_bias=torch.tensor([1.0, -2.0]),
            device="cpu",
        )
        spins = torch.tensor([[1.0, -1.0], [1.0, 1.0]], requires_grad=True)

        energy = bm(spins)
        energy.sum().backward()

        torch.testing.assert_close(energy, torch.tensor([1.0, -3.0]))
        torch.testing.assert_close(
            spins.grad, torch.tensor([[3.0, -2.0], [-5.0, -2.0]])
        )
        torch.testing.assert_close(bm.linear_bias.grad, torch.tensor([-2.0, 0.0]))
        self.assertIsNotNone(bm.quadratic_coef.grad)
        self.assertEqual(bm.quadratic_coef.grad.shape, (2, 2))
        self.assertTrue(torch.isfinite(bm.quadratic_coef.grad).all())

    def test_forward_preserves_float64_after_apply_move(self):
        """``.to(device)`` must not downcast a float64 batch onto meta weights."""
        bm = BoltzmannMachine(3, device="cpu").double()
        bm._apply(lambda tensor: tensor.to(torch.device("meta")))
        spins = torch.ones(2, 3, dtype=torch.float64)

        energy = bm(spins)

        self.assertEqual(bm.linear_bias.dtype, torch.float64)
        self.assertEqual(energy.dtype, torch.float64)
        self.assertEqual(energy.device.type, "meta")

    def test_dtype_only_move_still_scores_on_cpu(self):
        """``.to(dtype=...)`` keeps the device and the Hamiltonian values."""
        bm = BoltzmannMachine(
            2,
            quadratic_coef=torch.tensor([[0.0, 4.0], [4.0, 0.0]]),
            linear_bias=torch.tensor([1.0, -2.0]),
            device="cpu",
        )
        bm.to(dtype=torch.float64)
        spins = torch.tensor([[1.0, -1.0], [1.0, 1.0]], dtype=torch.float64)

        energy = bm(spins)

        self.assertEqual(bm.device, torch.device("cpu"))
        self.assertEqual(energy.dtype, torch.float64)
        self.assertEqual(energy.device, torch.device("cpu"))
        torch.testing.assert_close(
            energy, torch.tensor([1.0, -3.0], dtype=torch.float64)
        )

    def test_objective_after_apply_move_accepts_cpu_batches(self):
        """``objective`` goes through ``forward`` and must follow the same device."""
        bm = BoltzmannMachine(4, device="cpu")
        bm._apply(lambda tensor: tensor.to(torch.device("meta")))

        objective = bm.objective(torch.ones(2, 4), -torch.ones(5, 4))

        self.assertEqual(objective.shape, ())
        self.assertEqual(objective.device, torch.device("meta"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestForwardAcrossAccelerators(unittest.TestCase):
    """Construct-on-CUDA and parent move-to-CPU coverage for ``forward``."""

    def test_construct_on_cuda_accepts_cpu_batch(self):
        """A CPU batch must match a CPU reference and land on CUDA parameters."""
        bm = BoltzmannMachine(4)
        self.assertEqual(
            bm.quadratic_coef.device.type,
            "cuda",
            "construction should auto-select CUDA",
        )
        spins = torch.tensor([[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, 1.0]])

        energy = bm(spins)

        self.assertEqual(spins.device, torch.device("cpu"))
        self.assertEqual(energy.shape, (2,))
        self.assertEqual(energy.device.type, "cuda")
        torch.testing.assert_close(energy.detach().cpu(), _cpu_hamiltonian(bm, spins))

    def test_parent_move_to_cpu_accepts_cuda_batch(self):
        """After ``parent.to('cpu')``, a CUDA caller batch must score on CPU."""
        parent = nn.Module()
        parent.bm = BoltzmannMachine(4)
        parent.to("cpu")
        spins = torch.ones(3, 4, device="cuda")

        energy = parent.bm(spins)

        self.assertEqual(parent.bm.quadratic_coef.device.type, "cpu")
        self.assertEqual(parent.bm.device, torch.device("cpu"))
        self.assertEqual(spins.device.type, "cuda")
        self.assertEqual(energy.device, torch.device("cpu"))
        self.assertEqual(energy.shape, (3,))
        torch.testing.assert_close(energy, parent.bm(spins.cpu()))

    def test_stale_device_cache_still_scores_on_cuda(self):
        """Exporting ``self.device`` back to CPU must not pull weights off CUDA."""
        bm = BoltzmannMachine(4, device="cpu")
        bm.to("cuda")
        bm.device = torch.device("cpu")
        spins = torch.zeros(2, 4)

        energy = bm(spins)

        self.assertEqual(bm.device, torch.device("cpu"))
        self.assertEqual(bm._model_device().type, "cuda")
        self.assertEqual(energy.device.type, "cuda")
        self.assertEqual(energy.shape, (2,))
        torch.testing.assert_close(energy.detach().cpu(), _cpu_hamiltonian(bm, spins))

    def test_objective_cpu_batches_land_on_cuda(self):
        """``objective`` must return a CUDA scalar when parameters were built there."""
        bm = BoltzmannMachine(4)

        objective = bm.objective(torch.ones(2, 4), torch.full((3, 4), -1.0))

        self.assertEqual(objective.shape, ())
        self.assertEqual(objective.device.type, "cuda")
        self.assertTrue(torch.isfinite(objective).all())

    def test_backward_reaches_parameters_from_cpu_batch(self):
        """A device copy inside ``forward`` must stay in the autograd graph."""
        bm = BoltzmannMachine(4)
        spins = torch.ones(2, 4, requires_grad=True)

        energy = bm(spins)
        energy.sum().backward()

        self.assertEqual(energy.device.type, "cuda")
        self.assertIsNotNone(bm.linear_bias.grad)
        self.assertEqual(bm.linear_bias.grad.device.type, "cuda")
        self.assertTrue(torch.isfinite(bm.linear_bias.grad).all())
        self.assertIsNotNone(bm.quadratic_coef.grad)
        self.assertEqual(bm.quadratic_coef.grad.device.type, "cuda")
        self.assertIsNotNone(spins.grad)
        self.assertEqual(spins.grad.device, torch.device("cpu"))
        self.assertEqual(spins.grad.shape, spins.shape)


if __name__ == "__main__":
    unittest.main()
