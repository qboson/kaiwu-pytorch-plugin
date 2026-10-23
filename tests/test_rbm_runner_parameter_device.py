"""Regression tests for ``RBMRunner`` device alignment.

``RBMRunner`` is a scikit-learn transformer, not an ``nn.Module``. It used to
cache ``device`` at construction and send ``transform`` batches there. Nesting
the trained ``RestrictedBoltzmannMachine`` in a parent and moving that parent
through ``_apply`` (merged #164) relocates the RBM parameters and refreshes
``rbm.device`` without updating a plain runner attribute. ``get_hidden`` then
allocates its state on the parameter device and assigns the caller batch into
it:

``RuntimeError: Tensor on device meta is not on the expected device cpu!``

``RBM.get_hidden`` is left unchanged (#1334). These tests pin the runner: the
batch handed to ``get_hidden`` follows the live parameter device, while a new
``fit`` still places its model on the construction preference.

CPU cases use ``meta``. CUDA cases are skipped unless a GPU is present.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MPLBACKEND", "Agg")

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

META = torch.device("meta")
CPU = torch.device("cpu")


class _Optimizer:  # pylint: disable=too-few-public-methods
    """Stand-in for a Kaiwu sampler this example constructs but does not run."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class _Reducer(_Optimizer):  # pylint: disable=too-few-public-methods
    """Stand-in for ``PrecisionReducer``."""


def _install_kaiwu_stubs() -> None:
    """Register the Kaiwu SDK modules imported by the digits example."""
    classical = types.ModuleType("kaiwu.classical")
    classical.SimulatedAnnealingOptimizer = _Optimizer
    cim = types.ModuleType("kaiwu.cim")
    cim.CIMOptimizer = _Optimizer
    cim.PrecisionReducer = _Reducer
    common = types.ModuleType("kaiwu.common")
    common.CheckpointManager = _Optimizer
    for name, module in (
        ("kaiwu.classical", classical),
        ("kaiwu.cim", cim),
        ("kaiwu.common", common),
    ):
        sys.modules[name] = module


def _load_runner_module():
    """Load the example without requiring the Kaiwu SDK."""
    _install_kaiwu_stubs()
    path = (
        Path(__file__).resolve().parents[1] / "example" / "rbm_digits" / "rbm_digits.py"
    )
    spec = importlib.util.spec_from_file_location("rbm_digits_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


def _apply_device(module: nn.Module, device: torch.device) -> None:
    """Move parameters through ``_apply``, the parent-module path."""
    module._apply(lambda tensor: tensor.to(device))  # pylint: disable=protected-access


def _retarget_storage(device: torch.device):
    """Return an ``_apply`` callback that places zero storage on ``device``.

    ``Module.to`` cannot copy out of ``meta``. Replacing storage still goes
    through ``_apply``, so the parameter device changes without ``.to()``.
    Zeros keep a later CPU ``transform`` finite.
    """

    def _move(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == device and device.type != "meta":
            return tensor
        return torch.zeros(tensor.shape, dtype=tensor.dtype, device=device)

    return _move


def _runner(**kwargs):
    """Build a runner without printing the sampler banner."""
    kwargs.setdefault("n_components", 2)
    kwargs.setdefault("n_iter", 1)
    kwargs.setdefault("batch_size", 2)
    with redirect_stdout(io.StringIO()):
        return RUNNER.RBMRunner(**kwargs)


def _batch(rows: int = 5, cols: int = 4) -> np.ndarray:
    """Return a deterministic float32 batch that matches ``num_visible``."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((rows, cols)).astype(np.float32)


def _expected_features(rbm, batch: np.ndarray) -> np.ndarray:
    """Hidden features from the class method, ignoring an instance spy."""
    visible = torch.tensor(batch, dtype=torch.float32, device=CPU)
    hidden = RUNNER.RestrictedBoltzmannMachine.get_hidden(rbm, visible)
    return hidden[:, rbm.num_visible :].detach().cpu().numpy()


def _spy_get_hidden(rbm, seen: dict):
    """Record the device of the batch ``transform`` hands to ``get_hidden``."""

    def _spy(s_visible, *args, **kwargs):
        seen["input"] = s_visible.device
        output = RUNNER.RestrictedBoltzmannMachine.get_hidden(
            rbm, s_visible, *args, **kwargs
        )
        seen["output"] = output.device
        return output

    rbm.get_hidden = _spy


class RecordingRBM(  # pylint: disable=too-few-public-methods
    RUNNER.RestrictedBoltzmannMachine
):
    """RBM that records every direct ``.to()`` invocation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.direct_to_calls: list[tuple[tuple, dict]] = []

    def to(self, *args, **kwargs):
        """Record and forward module moves."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def _attach(runner, rbm, fallback: torch.device):
    """Hang ``rbm`` on ``runner`` and pin the construction preference."""
    runner.rbm = rbm
    runner.device = fallback
    parent = nn.Module()
    parent.rbm = rbm
    return parent


class TestRBMRunnerParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_cpu_transform_matches_hidden_features(self):
        """A CPU runner returns the hidden slice and leaves the batch intact."""
        runner = _runner()
        rbm = RUNNER.RestrictedBoltzmannMachine(4, 2, device=CPU)
        _attach(runner, rbm, CPU)
        batch = _batch()
        original = batch.copy()

        features = runner.transform(batch)

        np.testing.assert_array_equal(batch, original)
        np.testing.assert_allclose(features, _expected_features(rbm, batch))
        self.assertEqual(features.shape, (batch.shape[0], rbm.num_hidden))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(runner.device, CPU)

    def test_parent_to_does_not_call_rbm_to(self):
        """A parent move relocates weights without calling the child ``.to()``."""
        runner = _runner()
        rbm = RecordingRBM(4, 2, device=CPU)
        parent = _attach(runner, rbm, CPU)

        parent.to(META)

        self.assertEqual(rbm.direct_to_calls, [])
        self.assertEqual(next(rbm.parameters()).device, META)
        self.assertEqual(rbm.device, META)
        self.assertEqual(runner.device, META)
        self.assertEqual(runner._device, CPU)  # pylint: disable=protected-access

    def test_parent_apply_places_batch_on_parameter_device(self):
        """``transform`` must feed ``get_hidden`` on the live parameter device."""
        runner = _runner()
        rbm = RecordingRBM(4, 2, device=CPU)
        parent = _attach(runner, rbm, CPU)
        batch = _batch()
        pointer = batch.__array_interface__["data"][0]
        seen: dict = {}
        _spy_get_hidden(rbm, seen)
        _apply_device(parent, META)

        with self.assertRaises(NotImplementedError) as caught:
            runner.transform(batch)

        self.assertIn("meta", str(caught.exception).lower())
        self.assertEqual(seen["input"], META)
        self.assertEqual(seen["output"], META)
        self.assertEqual(batch.__array_interface__["data"][0], pointer)
        self.assertEqual(runner._device, CPU)  # pylint: disable=protected-access
        self.assertEqual(rbm.direct_to_calls, [])

    def test_stale_cache_does_not_win(self):
        """Assigning ``device`` must not redirect a batch off the parameters."""
        runner = _runner()
        rbm = RUNNER.RestrictedBoltzmannMachine(4, 2, device=CPU)
        _attach(runner, rbm, META)
        batch = _batch()

        features = runner.transform(batch)

        self.assertEqual(next(rbm.parameters()).device, CPU)
        self.assertEqual(runner.device, CPU)
        np.testing.assert_allclose(features, _expected_features(rbm, batch))
        self.assertTrue(np.isfinite(features).all())

    def test_dtype_only_move_keeps_cpu_device(self):
        """A parent ``.to(dtype=...)`` must not change the parameter device."""
        runner = _runner()
        rbm = RecordingRBM(4, 2, device=CPU)
        parent = _attach(runner, rbm, CPU)

        parent.to(dtype=torch.float64)

        self.assertEqual(rbm.direct_to_calls, [])
        parameter = next(rbm.parameters())
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(parameter.device, CPU)
        self.assertEqual(rbm.device, CPU)
        self.assertEqual(runner.device, CPU)

    def test_parameter_device_is_reread_after_round_trip(self):
        """The runner must follow parameters off ``meta`` and back."""
        runner = _runner()
        rbm = RecordingRBM(4, 2, device=CPU)
        parent = _attach(runner, rbm, CPU)
        batch = _batch()
        seen: dict = {}
        _spy_get_hidden(rbm, seen)

        _apply_device(parent, META)
        with self.assertRaises(NotImplementedError) as caught:
            runner.transform(batch)
        self.assertIn("meta", str(caught.exception).lower())
        self.assertEqual(seen["input"], META)
        self.assertEqual(runner.device, META)

        parent._apply(_retarget_storage(CPU))  # pylint: disable=protected-access
        del rbm.get_hidden
        features = runner.transform(batch)
        self.assertEqual(runner.device, CPU)
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features, _expected_features(rbm, batch))

        _apply_device(parent, META)
        self.assertEqual(runner.device, META)
        self.assertEqual(rbm.direct_to_calls, [])
        np.testing.assert_array_equal(batch, _batch())

    def test_fallback_when_rbm_missing(self):
        """With no parameters, ``device`` follows the setter; weights win later."""
        runner = _runner()
        runner.rbm = None
        runner.device = META

        self.assertIsNone(runner.rbm)
        self.assertEqual(runner.device, META)

        runner.rbm = RUNNER.RestrictedBoltzmannMachine(4, 2, device=CPU)
        self.assertEqual(runner.device, CPU)
        self.assertEqual(runner._device, META)  # pylint: disable=protected-access

    def test_fit_places_new_rbm_on_fallback(self):
        """Refit must place the new RBM on the construction device."""
        runner = _runner()
        runner.rbm = RUNNER.RestrictedBoltzmannMachine(4, 2, device=META)
        runner.device = CPU
        recorded: list = []
        original = RUNNER.RestrictedBoltzmannMachine.to

        def _stop(module, *args, **kwargs):  # pylint: disable=unused-argument
            recorded.append(args[0] if args else kwargs.get("device"))
            raise RuntimeError("stop before training")

        RUNNER.RestrictedBoltzmannMachine.to = _stop
        try:
            with self.assertRaisesRegex(RuntimeError, "stop before training"):
                runner.fit(np.zeros((4, 4), dtype=np.float32))
        finally:
            RUNNER.RestrictedBoltzmannMachine.to = original

        self.assertEqual(recorded, [CPU])
        self.assertEqual(runner._device, CPU)  # pylint: disable=protected-access


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestRBMRunnerParameterDeviceCUDA(unittest.TestCase):
    """End-to-end coverage of a parent move onto CUDA."""

    def test_parent_apply_accepts_cpu_batch(self):
        """CPU batches stay put; features match a CPU clone of the weights."""
        cuda = torch.device("cuda")
        runner = _runner()
        rbm = RecordingRBM(4, 2, device=CPU)
        weights = {
            name: value.detach().cpu().clone()
            for name, value in rbm.state_dict().items()
        }
        parent = _attach(runner, rbm, CPU)
        batch = _batch()
        pointer = batch.__array_interface__["data"][0]
        seen: dict = {}
        _spy_get_hidden(rbm, seen)
        _apply_device(parent, cuda)

        features = runner.transform(batch)

        reference = RUNNER.RestrictedBoltzmannMachine(
            4,
            2,
            quadratic_coef=weights["quadratic_coef"],
            linear_bias=weights["linear_bias"],
            device=CPU,
        )
        self.assertEqual(batch.__array_interface__["data"][0], pointer)
        self.assertEqual(seen["input"].type, "cuda")
        self.assertEqual(seen["output"].type, "cuda")
        self.assertEqual(runner.device.type, "cuda")
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features, _expected_features(reference, batch))
        self.assertEqual(rbm.direct_to_calls, [])

    def test_device_is_reread_across_cuda_and_cpu(self):
        """CUDA, then CPU, then CUDA must follow parameters."""
        cuda = torch.device("cuda")
        runner = _runner()
        rbm = RecordingRBM(4, 2, device=CPU)
        parent = _attach(runner, rbm, CPU)
        batch = _batch()

        _apply_device(parent, cuda)
        self.assertEqual(runner.device.type, "cuda")
        _apply_device(parent, CPU)
        features = runner.transform(batch)
        self.assertEqual(runner.device, CPU)
        self.assertTrue(np.isfinite(features).all())
        _apply_device(parent, cuda)
        self.assertEqual(runner.device.type, "cuda")
        self.assertEqual(rbm.direct_to_calls, [])

    def test_dtype_only_move_keeps_cuda_device(self):
        """A parent dtype cast must not pull CUDA parameters back to CPU."""
        cuda = torch.device("cuda")
        runner = _runner()
        rbm = RecordingRBM(4, 2, device=CPU)
        parent = _attach(runner, rbm, CPU)
        _apply_device(parent, cuda)

        parent.to(dtype=torch.float64)

        parameter = next(rbm.parameters())
        self.assertEqual(rbm.direct_to_calls, [])
        self.assertEqual(parameter.device.type, "cuda")
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(runner.device.type, "cuda")
        self.assertEqual(rbm.device.type, "cuda")

    def test_stale_fallback_does_not_win(self):
        """A CPU construction preference must not override CUDA parameters."""
        cuda = torch.device("cuda")
        runner = _runner()
        rbm = RUNNER.RestrictedBoltzmannMachine(4, 2, device=CPU)
        weights = {
            name: value.detach().cpu().clone()
            for name, value in rbm.state_dict().items()
        }
        parent = _attach(runner, rbm, CPU)
        _apply_device(parent, cuda)
        runner.device = CPU
        batch = _batch()

        features = runner.transform(batch)

        reference = RUNNER.RestrictedBoltzmannMachine(
            4,
            2,
            quadratic_coef=weights["quadratic_coef"],
            linear_bias=weights["linear_bias"],
            device=CPU,
        )
        self.assertEqual(runner.device.type, "cuda")
        self.assertEqual(runner._device, CPU)  # pylint: disable=protected-access
        np.testing.assert_allclose(features, _expected_features(reference, batch))
