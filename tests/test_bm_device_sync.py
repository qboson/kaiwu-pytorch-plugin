"""Regression tests for BoltzmannMachine device tracking.

``AbstractBoltzmannMachine`` stores its device as a plain attribute. PyTorch's
``nn.Module.to()`` moves child modules through ``_apply()`` *without* calling a
child's ``.to()`` override, so transfers must synchronize in ``_apply``.
Loading a parent checkpoint with ``assign=True`` also replaces the child's
parameters and needs synchronization before the next sampler call.

The CUDA-dependent tests below run on any CUDA machine; they cover the
construct-on-CUDA / move-to-CPU path that CPU-only development machines
cannot exercise.
"""

import unittest

import numpy as np
import torch
from torch import nn

from kaiwu.torch_plugin import BoltzmannMachine, EnergyModel
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


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

    def test_assign_load_syncs_nested_machines_before_sampling(self):
        """Assigning CPU checkpoint parameters materializes meta machines."""
        for model_type, dimensions in (
            (BoltzmannMachine, (4,)),
            (RestrictedBoltzmannMachine, (2, 2)),
            (GaussianBernoulliRestrictedBoltzmannMachine, (2, 2)),
        ):
            with self.subTest(model=model_type.__name__):
                cpu = nn.Sequential(model_type(*dimensions, device="cpu"))
                target = nn.Sequential(model_type(*dimensions, device="cpu")).to("meta")
                target.load_state_dict(cpu.state_dict(), assign=True)
                self.assertEqual(target[0].device, torch.device("cpu"))
                np.testing.assert_allclose(
                    target[0].get_ising_matrix(), cpu[0].get_ising_matrix()
                )
                torch.testing.assert_close(
                    target[0].sample(DummySampler()), cpu[0].sample(DummySampler())
                )


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
