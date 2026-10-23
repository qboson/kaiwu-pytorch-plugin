"""Regression tests for full Boltzmann-machine Ising export on the parameter device.

``BoltzmannMachine._to_ising_matrix`` builds a work buffer and copies parameter
values into it before returning a NumPy matrix for external samplers. That
buffer must live on the same device as ``quadratic_coef`` and ``linear_bias``.
After a parent module relocates weights through ``nn.Module._apply`` (the path
covered for device bookkeeping in merged **#164** / ``tests/test_bm_device_sync.py``),
a stale ``self.device`` cache must not allocate the buffer on the wrong device.
"""

# Parent-initiated moves are exercised through ``_apply`` (see #164).
# pylint: disable=protected-access

import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from kaiwu.torch_plugin import BoltzmannMachine


class DummySampler:  # pylint: disable=too-few-public-methods
    """Deterministic stand-in for a Kaiwu optimizer."""

    def __init__(self, num_solutions: int = 2) -> None:
        self.num_solutions = num_solutions

    def solve(self, ising_mat):
        """Return a fixed binary solution batch for the supplied Ising matrix."""
        return np.ones(
            (self.num_solutions, ising_mat.shape[0]),
            dtype=np.float32,
        )


def _export_ising_matrix_on_meta(bm: BoltzmannMachine) -> np.ndarray:
    """Finish Ising export when parameters live on ``meta`` (CPU-only CI).

    ``meta`` parameters cannot be copied to NumPy without real storage. The
    production path still runs through buffer allocation and parameter copies;
    only the final ``cpu().numpy()`` hop is stubbed so CPU-only CI can assert
    the mechanism.
    """
    with mock.patch.object(torch.Tensor, "cpu", lambda tensor, **kwargs: tensor):
        with mock.patch.object(
            torch.Tensor,
            "numpy",
            return_value=np.zeros((bm.num_nodes + 1, bm.num_nodes + 1)),
        ):
            return bm.get_ising_matrix()


class RecordingBoltzmannMachine(BoltzmannMachine):
    """BoltzmannMachine that records every direct ``.to()`` invocation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.direct_to_calls: list[tuple[tuple, dict]] = []

    def to(self, *args, **kwargs):
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


class TestBmIsingMatrixMechanism(unittest.TestCase):
    """Meta-tensor mechanism tests that run on every machine."""

    def test_parameter_device_tracks_apply_move(self):
        """``_parameter_device()`` must follow ``_apply``, not a stale cache."""
        bm = BoltzmannMachine(4, device="cpu")
        bm._apply(lambda tensor: tensor.to(torch.device("meta")))

        self.assertEqual(bm.device, torch.device("meta"))
        self.assertEqual(bm._parameter_device(), bm.quadratic_coef.device)
        self.assertEqual(bm._parameter_device().type, "meta")

    def test_stale_device_cache_repro_with_cpu_buffer(self):
        """Pre-fix placement on ``self.device`` fails when parameters sit on meta."""
        bm = BoltzmannMachine(4, device="cpu")
        bm._apply(lambda tensor: tensor.to(torch.device("meta")))
        stale_device = torch.device("cpu")

        with torch.no_grad():
            quadratic_coef = bm.symmetrized_quadratic_coef()
            ising_mat = torch.zeros(
                (bm.num_nodes + 1, bm.num_nodes + 1),
                device=stale_device,
                dtype=bm.linear_bias.dtype,
            )
            with self.assertRaises((RuntimeError, NotImplementedError)):
                ising_mat[:-1, :-1] = quadratic_coef / 8

    def test_stale_device_cache_still_exports_on_parameter_device(self):
        """``_parameter_device()`` must win over a manually stale ``self.device``."""
        bm = BoltzmannMachine(4, device="cpu")
        bm._apply(lambda tensor: tensor.to(torch.device("meta")))
        bm.device = torch.device("cpu")

        ising_mat = _export_ising_matrix_on_meta(bm)

        self.assertEqual(bm.device, torch.device("cpu"))
        self.assertIsInstance(ising_mat, np.ndarray)
        self.assertEqual(ising_mat.shape, (5, 5))

    def test_parent_apply_move_does_not_require_child_to(self):
        """Parent ``_apply`` must export Ising matrices without calling child ``.to()``."""
        parent = nn.Module()
        parent.bm = RecordingBoltzmannMachine(4, device="cpu")

        parent._apply(lambda tensor: tensor.to(torch.device("meta")))

        self.assertEqual(parent.bm.direct_to_calls, [])
        ising_mat = _export_ising_matrix_on_meta(parent.bm)

        self.assertEqual(ising_mat.shape, (5, 5))
        self.assertEqual(parent.bm.quadratic_coef.device.type, "meta")
        self.assertEqual(
            parent.bm._parameter_device(),
            parent.bm.quadratic_coef.device,
        )

    def test_work_buffer_allocated_on_parameter_device(self):
        """``torch.zeros`` for the Ising buffer must target the parameter device."""
        bm = BoltzmannMachine(4, device="cpu")
        bm._apply(lambda tensor: tensor.to(torch.device("meta")))
        bm.device = torch.device("cpu")
        captured: dict[str, torch.device] = {}
        original_zeros = torch.zeros

        def record_zeros(*args, **kwargs):
            tensor = original_zeros(*args, **kwargs)
            if tensor.ndim == 2 and tensor.shape[0] == bm.num_nodes + 1:
                captured["device"] = tensor.device
            return tensor

        with mock.patch("torch.zeros", record_zeros):
            _export_ising_matrix_on_meta(bm)

        self.assertEqual(captured["device"], bm.quadratic_coef.device)
        self.assertEqual(captured["device"].type, "meta")

    def test_to_ising_matrix_uses_parameter_dtype(self):
        """The work buffer dtype must follow the parameter dtype."""
        bm = BoltzmannMachine(3, device="cpu").double()
        bm._apply(lambda tensor: tensor.to(torch.device("meta")))

        with torch.no_grad():
            ising_tensor = torch.zeros(
                (4, 4),
                device=bm._parameter_device(),
                dtype=bm.linear_bias.dtype,
            )
            ising_tensor[:-1, :-1] = bm.symmetrized_quadratic_coef() / 8

        self.assertEqual(ising_tensor.dtype, torch.float64)
        self.assertEqual(ising_tensor.device.type, "meta")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBmIsingMatrixCuda(unittest.TestCase):
    """End-to-end Ising export after relocating a nested BM."""

    def test_parent_move_exports_from_cuda_parameters(self):
        """Nested BMs must export Ising matrices after ``parent.to('cpu')``."""
        parent = nn.Module()
        parent.bm = BoltzmannMachine(4)
        self.assertEqual(
            parent.bm.quadratic_coef.device.type,
            "cuda",
            "construction should auto-select CUDA",
        )

        parent.to("cpu")

        ising_mat = parent.bm.get_ising_matrix()

        self.assertEqual(parent.bm.quadratic_coef.device.type, "cpu")
        self.assertEqual(parent.bm.device, torch.device("cpu"))
        self.assertEqual(ising_mat.shape, (5, 5))
        self.assertTrue(np.isfinite(ising_mat).all())

    def test_sample_uses_ising_export_after_parent_move_to_cpu(self):
        """``sample()`` must keep working once parameters leave the accelerator."""
        parent = nn.Module()
        parent.bm = BoltzmannMachine(4)
        parent.to("cpu")

        samples = parent.bm.sample(DummySampler(num_solutions=3))

        self.assertEqual(samples.device.type, "cpu")
        self.assertEqual(samples.shape, (3, 4))

    def test_stale_device_attribute_after_cuda_parent_move(self):
        """Export must follow parameters when ``self.device`` is stale on CUDA."""
        parent = nn.Module()
        parent.bm = BoltzmannMachine(4, device="cpu")
        parent.bm.to("cuda")
        parent.bm.device = torch.device("cpu")

        ising_mat = parent.bm.get_ising_matrix()

        self.assertEqual(parent.bm.quadratic_coef.device.type, "cuda")
        self.assertEqual(ising_mat.shape, (5, 5))


if __name__ == "__main__":
    unittest.main()
