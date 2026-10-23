"""Regression tests for ``FIDCalculator`` device alignment.

``FIDCalculator`` caches ``device`` at construction and moves
``FrechetInceptionDistance`` there once. A parent ``nn.Module._apply``
(merged #164, ``tests/test_bm_device_sync.py``) moves that metric without
calling its ``.to`` and without updating the cache. ``update_batch`` used to
send preprocessed images to ``self.device``, so a meta or CUDA metric
received a CPU batch:

``RuntimeError: Tensor on device meta is not on the expected device cpu!``

A meta batch copied onto that stale CPU cache raises
``NotImplementedError: Cannot copy out of meta tensor``.

``self.device`` stays the construction preference. Each update rereads the
live parameter device. CUDA tests are collected and skipped without a GPU.

``compute_fid_in_batches`` list inputs are #567. ``MLPClassifier`` is #1401.
This module does not change either path.
"""

# Parent moves go through ``_apply`` (see merged #164).
# pylint: disable=protected-access

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

QVAE_ROOT = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
if str(QVAE_ROOT) not in sys.path:
    sys.path.insert(0, str(QVAE_ROOT))

try:
    from utils import helpers
except ImportError:  # CI installs the library, not the example image stack.
    helpers = None

CPU = torch.device("cpu")
META = torch.device("meta")


def _ready():
    """Build a CPU calculator whose metric a parent module can move."""
    if helpers is None:
        raise unittest.SkipTest("requires qvae_mnist FID dependencies")
    try:
        calculator = helpers.FIDCalculator(CPU, feature=64)
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest(str(exc)) from exc
    calls = _record_to(calculator.fid)
    parent = nn.Module()
    parent.metric = calculator.fid
    return calculator, parent, calls


def _record_to(module: nn.Module) -> list:
    """Record direct ``.to`` calls. Parent ``_apply`` must leave this empty."""
    calls: list[tuple[tuple, dict]] = []
    original = module.to

    def _recorded(*args, **kwargs):
        """Append the call, then perform the real move."""
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    module.to = _recorded
    return calls


def _spy(calculator) -> dict:
    """Record the tensor ``fid.update`` actually receives."""
    seen: dict = {}
    original = calculator.fid.update

    def _wrapped(images, real):
        """Store the batch, then run the real update."""
        seen["images"] = images
        seen["real"] = real
        return original(images, real)

    calculator.fid.update = _wrapped
    return seen


def _images(rows: int = 2, device: torch.device = CPU) -> torch.Tensor:
    """Return a deterministic flattened MNIST-shaped batch."""
    if device.type == "meta":
        return torch.empty(rows, 784, device=device)
    generator = torch.Generator()
    generator.manual_seed(0)
    return torch.rand(rows, 784, generator=generator).to(device)


def _retarget_storage(device: torch.device):
    """Return an ``_apply`` callback that places zero storage on ``device``.

    ``Module.to`` cannot copy out of ``meta``. Replacing storage still goes
    through ``_apply``, so the parameter device changes without ``.to()``.
    """

    def _move(tensor: torch.Tensor) -> torch.Tensor:
        """Place ``tensor`` on ``device`` without reading its data."""
        if tensor.device == device:
            return tensor
        return torch.zeros(tensor.shape, dtype=tensor.dtype, device=device)

    return _move


@unittest.skipUnless(helpers is not None, "requires qvae_mnist FID dependencies")
class TestFIDCalculatorParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_cpu_update_is_finite_and_keeps_the_caller(self):
        """An unmoved calculator scores on CPU and does not rewrite the batch."""
        calculator, _parent, calls = _ready()
        seen = _spy(calculator)
        batch = _images()
        pointer = batch.data_ptr()

        calculator.update_batch(batch, True)
        calculator.update_batch(batch, False)
        score = calculator.compute()

        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        self.assertEqual(calculator._metric_device(), CPU)
        self.assertEqual(seen["images"].device, CPU)
        self.assertEqual(seen["images"].dtype, torch.uint8)
        self.assertEqual(tuple(seen["images"].shape), (2, 3, 299, 299))
        self.assertIsInstance(score, float)
        self.assertLess(abs(score), 1e-4)
        self.assertEqual(batch.data_ptr(), pointer)

    def test_parent_apply_meta_does_not_call_child_to(self):
        """A parent move leaves the cache behind and follows the parameters."""
        calculator, parent, calls = _ready()

        parent._apply(lambda tensor: tensor.to(META))

        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        self.assertEqual(next(calculator.fid.parameters()).device, META)
        self.assertEqual(calculator._metric_device(), META)

    def test_cpu_batch_after_parent_apply_meta_follows_parameters(self):
        """A CPU batch is presented to the metric on meta, not on the cache."""
        calculator, parent, calls = _ready()
        parent._apply(lambda tensor: tensor.to(META))
        seen = _spy(calculator)
        batch = _images()
        pointer = batch.data_ptr()

        calculator.update_batch(batch, True)

        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        self.assertEqual(seen["images"].device, META)
        self.assertEqual(seen["images"].dtype, torch.uint8)
        self.assertEqual(seen["real"], True)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(batch.device, CPU)
        with self.assertRaises(RuntimeError) as caught:
            calculator.compute()
        self.assertIn("meta", str(caught.exception).lower())

    def test_meta_batch_after_parent_apply_meta_is_not_copied_out(self):
        """A meta batch stays on meta instead of being copied onto the CPU cache."""
        calculator, parent, _calls = _ready()
        parent._apply(lambda tensor: tensor.to(META))
        seen = _spy(calculator)
        batch = _images(device=META)

        calculator.update_batch(batch, False)

        self.assertEqual(batch.device, META)
        self.assertEqual(seen["images"].device, META)
        self.assertEqual(seen["images"].dtype, torch.uint8)
        self.assertEqual(seen["real"], False)

    def test_cache_placement_mixes_meta_weights_with_cpu_batches(self):
        """The old cache send is the device-mix ``RuntimeError``."""
        calculator, parent, _calls = _ready()
        parent._apply(lambda tensor: tensor.to(META))
        calculator._metric_device = lambda: calculator.device

        with self.assertRaises(RuntimeError) as caught:
            calculator.update_batch(_images(), True)

        message = str(caught.exception)
        self.assertIn("meta", message)
        self.assertIn("cpu", message)

    def test_meta_cache_does_not_override_cpu_weights(self):
        """Pointing the cache at meta leaves CPU weights in charge."""
        calculator, _parent, calls = _ready()
        calculator.device = META
        seen = _spy(calculator)
        batch = _images()

        calculator.update_batch(batch, True)
        calculator.update_batch(batch, False)
        score = calculator.compute()

        self.assertEqual(calls, [])
        self.assertEqual(next(calculator.fid.parameters()).device, CPU)
        self.assertEqual(calculator._metric_device(), CPU)
        self.assertEqual(seen["images"].device, CPU)
        self.assertLess(abs(score), 1e-4)
        self.assertEqual(batch.device, CPU)

    def test_parent_to_meta_does_not_call_child_to(self):
        """``Module.to(device)`` is the same ``_apply`` path as a raw move."""
        calculator, parent, calls = _ready()

        parent.to(device=META)
        seen = _spy(calculator)
        calculator.update_batch(_images(), True)

        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        self.assertEqual(seen["images"].device, META)

    def test_dtype_only_move_keeps_the_cpu_device(self):
        """A dtype-only parent move must not change the parameter device."""
        calculator, parent, calls = _ready()

        parent.to(dtype=torch.float64)

        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        parameter = next(calculator.fid.parameters())
        self.assertEqual(parameter.device, CPU)
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(calculator._metric_device(), CPU)

    def test_module_to_cpu_cannot_copy_out_of_meta(self):
        """Returning from meta replaces storage with ``_apply`` and scores."""
        calculator, parent, calls = _ready()
        parent._apply(lambda tensor: tensor.to(META))

        with self.assertRaises(NotImplementedError) as caught:
            parent.to(CPU)
        self.assertIn("meta", str(caught.exception).lower())
        self.assertEqual(next(calculator.fid.parameters()).device, META)
        self.assertEqual(calls, [])

        parent._apply(_retarget_storage(CPU))
        seen = _spy(calculator)
        batch = _images(rows=4)
        calculator.update_batch(batch, True)
        calculator.update_batch(_images(rows=4), False)

        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        self.assertEqual(seen["images"].device, CPU)
        self.assertEqual(next(calculator.fid.parameters()).device, CPU)
        self.assertTrue(torch.isfinite(torch.tensor(calculator.compute())))

    def test_metric_device_falls_back_without_parameters(self):
        """No parameters follows a buffer, then the construction preference."""
        calculator, _parent, _calls = _ready()

        class _BufferOnly(nn.Module):
            """Module whose only storage is a buffer."""

            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("marker", torch.ones(1))

            def forward(self, *_args, **_kwargs):
                """Unused. The test only reads the buffer device."""
                raise NotImplementedError

        buffered = _BufferOnly()
        calculator.fid = buffered
        parent = nn.Module()
        parent.metric = buffered
        parent._apply(lambda tensor: tensor.to(META))

        self.assertEqual(calculator.device, CPU)
        self.assertEqual(calculator._metric_device(), META)

        calculator.fid = nn.Module()
        calculator.device = META
        self.assertEqual(calculator._metric_device(), META)
        calculator.device = CPU
        self.assertEqual(calculator._metric_device(), CPU)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipUnless(helpers is not None, "requires qvae_mnist FID dependencies")
class TestFIDCalculatorCuda(unittest.TestCase):
    """End-to-end coverage of a parent move onto CUDA."""

    def test_parent_apply_cuda_updates_on_cuda(self):
        """A CPU batch is scored on CUDA while the cache stays on CPU."""
        calculator, parent, calls = _ready()
        parent._apply(lambda tensor: tensor.to("cuda"))
        seen = _spy(calculator)
        batch = _images()
        pointer = batch.data_ptr()

        calculator.update_batch(batch, True)
        calculator.update_batch(_images(rows=2), False)
        score = calculator.compute()

        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        self.assertEqual(calculator._metric_device().type, "cuda")
        self.assertEqual(seen["images"].device.type, "cuda")
        self.assertEqual(seen["images"].dtype, torch.uint8)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertIsInstance(score, float)
        self.assertTrue(math.isfinite(score))

    def test_parent_to_cuda_accepts_a_cuda_batch(self):
        """A batch that already lives on CUDA is not pulled back to the cache."""
        calculator, parent, calls = _ready()
        parent.to(device="cuda")
        seen = _spy(calculator)
        batch = torch.rand(2, 784, device="cuda")
        pointer = batch.data_ptr()

        calculator.update_batch(batch, True)
        calculator.update_batch(batch, False)

        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(batch.device.type, "cuda")
        self.assertEqual(seen["images"].device.type, "cuda")
        self.assertTrue(torch.isfinite(torch.tensor(calculator.compute())))


if __name__ == "__main__":
    unittest.main()
