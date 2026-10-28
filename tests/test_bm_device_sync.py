"""Regression tests for BoltzmannMachine device tracking.

``AbstractBoltzmannMachine`` stores its device as a plain attribute. PyTorch's
``nn.Module.to()`` moves child modules through ``_apply()`` *without* calling a
child's ``.to()`` override, so device bookkeeping must live in ``_apply``:
otherwise a BM nested inside another module (for example
``EnergyModel.energy_bm``) keeps a stale ``self.device`` after its parameters
were moved, and ``_to_ising_matrix()`` crashes mixing two devices. The
``_to_ising_matrix`` implementations therefore also derive their work-buffer
device from model parameters as defense in depth.

The CUDA-dependent tests below run on any CUDA machine; they cover the
construct-on-CUDA / move-to-CPU path that CPU-only development machines
cannot exercise.
"""

import unittest

import numpy as np
import torch
from torch import nn

from kaiwu.torch_plugin import BoltzmannMachine, EnergyModel


class DummySampler:
    """Deterministic stand-in for a Kaiwu optimizer."""

    def __init__(self, num_solutions: int = 2) -> None:
        self.num_solutions = num_solutions

    def solve(self, ising_mat):
        return np.ones(
            (self.num_solutions, ising_mat.shape[0]), dtype=np.float32
        )


class RecordingBoltzmannMachine(BoltzmannMachine):
    """BoltzmannMachine that records every direct ``.to()`` invocation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.direct_to_calls: list[tuple[tuple, dict]] = []

    def to(self, *args, **kwargs):
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


class TestDeviceBookkeeping(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_parent_move_keeps_device_attribute_synced(self):
        """A parent-initiated move must still update the BM device attribute."""
        parent = nn.Module()
        parent.bm = RecordingBoltzmannMachine(4)

        parent.to(parent.bm.quadratic_coef.device)

        # nn.Module never invokes the child's ``.to()`` override; the sync
        # must happen through the ``_apply`` override instead.
        self.assertEqual(parent.bm.direct_to_calls, [])
        self.assertEqual(
            parent.bm.device,
            parent.bm.quadratic_coef.device,
            "device attribute diverged from the parameter device after a "
            "parent-initiated move",
        )

    def test_apply_move_syncs_device_attribute(self):
        """``_apply`` is the exact path parent modules use; it must sync."""
        bm = BoltzmannMachine(4, device="cpu")

        bm._apply(lambda tensor: tensor.to(torch.device("meta")))

        self.assertEqual(bm.device, torch.device("meta"))
        self.assertEqual(bm.quadratic_coef.device, torch.device("meta"))

    def test_dtype_only_move_keeps_device_attribute_intact(self):
        """``.to(dtype=...)`` must behave like on any other nn.Module."""
        bm = BoltzmannMachine(4, device="cpu")

        bm.to(dtype=torch.float64)

        self.assertEqual(bm.quadratic_coef.dtype, torch.float64)
        self.assertEqual(bm.device, torch.device("cpu"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDeviceSyncAcrossAccelerators(unittest.TestCase):
    """End-to-end coverage of the construct-on-CUDA / move-to-CPU path."""

    def test_parent_move_leaves_no_stale_device_attribute(self):
        """Params and the device attribute move together with ``.to()``."""
        energy = EnergyModel(
            bm_num_visible=4,
            bm_num_hidden=4,
            sampler=DummySampler(),
        )
        self.assertEqual(
            energy.energy_bm.quadratic_coef.device.type,
            "cuda",
            "construction should auto-select CUDA",
        )

        energy.to("cpu")

        self.assertEqual(energy.energy_bm.quadratic_coef.device.type, "cpu")
        self.assertEqual(energy.energy_bm.device, torch.device("cpu"))

    def test_score_visible_logits_after_parent_move(self):
        """Scoring must work after moving a parent module between devices."""
        energy = EnergyModel(
            bm_num_visible=4,
            bm_num_hidden=4,
            sampler=DummySampler(),
        )
        energy.to("cpu")

        scores = energy.score_visible_logits(torch.zeros(2, 4))

        self.assertEqual(scores.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
