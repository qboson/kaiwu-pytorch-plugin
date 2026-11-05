"""Regression tests for ``FIDCalculator`` device alignment.

``FIDCalculator`` caches ``device`` at construction and moves
``FrechetInceptionDistance`` there once. A parent ``nn.Module._apply``
(merged #164, ``tests/test_bm_device_sync.py``) moves that metric without
calling its ``.to`` and without updating the cache. ``update_batch`` used to
send preprocessed images to ``self.device``, so a meta or CUDA metric
received a CPU batch:

    ``RuntimeError: Tensor on device meta is not on the expected device cpu!``

A meta batch copied onto that stale CPU cache raises
``NotImplementedError: Cannot copy out of meta tensor; no data!``.

Pointing the cache at ``cuda`` on a CPU-only build does not get as far as the
metric. ``Tensor.to(torch.device("cuda"))`` raises
``AssertionError: Torch not compiled with CUDA enabled``.

``self.device`` stays the construction preference. Each update rereads the
live parameter device (then a buffer, then the preference). Same-device
``Tensor.to`` returns the same object, including a noncontiguous view. Every
meta tensor reports ``data_ptr() == 0``, so meta identity is the Python
object. ``Module.to(cpu)`` cannot copy out of ``meta``; the return trip
retargets storage with ``_apply`` and ``zeros_like``.

``compute_fid_in_batches`` list inputs are #567. ``MLPClassifier`` is #1401.
This module does not change either path. Images are cast to ``uint8`` before
the metric. A float64 inception is a dtype mismatch, not the device bug.
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
    from utils import helpers  # pylint: disable=import-error
except ImportError:  # CI installs the library, not the example image stack.
    helpers = None

CPU = torch.device("cpu")
META = torch.device("meta")
CUDA = torch.device("cuda")


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


def _record_to(module):
    """Record direct ``.to`` calls. Parent ``_apply`` must leave this empty."""
    calls = []
    original = module.to

    def _recorded(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    module.to = _recorded
    return calls


def _watch(calculator, preprocessor=None):
    """Record the tensor ``fid.update`` receives and the tensor ``.to`` saw.

    The two are the same object when the preprocessor output is already on
    the metric device. The real ``update`` still runs.
    """
    seen = {}
    if getattr(calculator, "_watched", False):
        original_update = calculator._origin_update
        original_pre = calculator._origin_pre
    else:
        original_update = calculator.fid.update
        original_pre = calculator.preprocessor
        calculator._origin_update = original_update
        calculator._origin_pre = original_pre
        calculator._watched = True
    if preprocessor is not None:
        original_pre = preprocessor

    def _pre(images):
        processed = original_pre(images)
        seen["processed"] = processed
        return processed

    def _update(images, real):
        seen["images"] = images
        seen["real"] = real
        return original_update(images, real)

    calculator.preprocessor = _pre
    calculator.fid.update = _update
    return seen


def _images(rows=2, device=CPU, dtype=torch.float32, noncontiguous=False):
    """Return a flattened MNIST-shaped batch, optionally a noncontiguous view.

    The view is sliced on ``device``. ``Tensor.to(meta)`` would materialize a
    contiguous tensor and make a same-device identity check vacuous.
    """
    destination = torch.device(device)
    width = 784 * (2 if noncontiguous else 1)
    if destination.type == "meta":
        base = torch.empty(rows, width, dtype=dtype, device=destination)
    else:
        generator = torch.Generator()
        generator.manual_seed(0)
        base = torch.rand(rows, width, generator=generator, dtype=dtype)
    if not noncontiguous:
        return base
    return base.reshape(rows, 784 * 2)[:, ::2]


def _noncontiguous_processed(images):
    """Stand in for the resizer and return a noncontiguous ``uint8`` view.

    ``Resize`` always materializes a contiguous tensor, so the real
    preprocessor cannot show that ``Tensor.to`` keeps a view.
    """
    rows = images.shape[0]
    base = torch.zeros(rows, 3, 299, 598, dtype=torch.uint8, device=images.device)
    return base[:, :, :, ::2]


def _apply_device(module, device):
    """Move parameters the way a parent module does.

    Leaving ``meta`` cannot copy values out, so that direction allocates
    fresh zeros with ``zeros_like`` (the source values are not read). Every
    other hop uses ``Tensor.to``.
    """
    destination = torch.device(device)

    def _move(tensor):
        if tensor.device.type == "meta" and destination.type != "meta":
            return torch.zeros_like(tensor, device=destination)
        return tensor.to(destination)

    module._apply(_move)


def _snapshot(tensor):
    """Capture storage identity. Meta tensors have no values to clone."""
    values = None
    if tensor.device.type != "meta":
        values = tensor.detach().clone()
    return {
        "tensor": tensor,
        "ptr": tensor.data_ptr(),
        "device": tensor.device,
        "dtype": tensor.dtype,
        "contiguous": tensor.is_contiguous(),
        "values": values,
    }


def _assert_unchanged(case, tensor, snapshot):
    """The update must not rewrite the caller's storage."""
    case.assertIs(tensor, snapshot["tensor"])
    case.assertEqual(tensor.data_ptr(), snapshot["ptr"])
    case.assertEqual(tensor.device, snapshot["device"])
    case.assertEqual(tensor.dtype, snapshot["dtype"])
    case.assertEqual(tensor.is_contiguous(), snapshot["contiguous"])
    if snapshot["values"] is not None:
        case.assertTrue(torch.equal(tensor, snapshot["values"]))


def _assert_device(case, actual, expected):
    """Compare CPU and meta exactly. CUDA matches on device type."""
    if expected.type == "cuda":
        case.assertEqual(actual.type, "cuda")
        return
    case.assertEqual(actual, expected)


def _assert_placed(case, seen, device, same_storage):
    """Check the image ``fid.update`` received against the preprocessor output."""
    images = seen["images"]
    processed = seen["processed"]
    _assert_device(case, images.device, device)
    case.assertEqual(images.dtype, torch.uint8)
    case.assertEqual(tuple(images.shape[1:]), (3, 299, 299))
    if same_storage:
        case.assertIs(images, processed)
        case.assertEqual(images.data_ptr(), processed.data_ptr())
        case.assertEqual(images.is_contiguous(), processed.is_contiguous())
    else:
        case.assertIsNot(images, processed)
        both_meta = images.device.type == "meta" and processed.device.type == "meta"
        if not both_meta:
            case.assertNotEqual(images.data_ptr(), processed.data_ptr())
    if device.type == "meta":
        case.assertEqual(images.data_ptr(), 0)


@unittest.skipUnless(helpers is not None, "requires qvae_mnist FID dependencies")
class TestFIDCalculatorParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_cpu_update_is_finite_and_keeps_the_caller(self):
        """An unmoved calculator scores on CPU and does not rewrite the batch."""
        calculator, _parent, calls = _ready()
        seen = _watch(calculator)
        batch = _images(noncontiguous=True)
        snapshot = _snapshot(batch)
        self.assertFalse(batch.is_contiguous())

        calculator.update_batch(batch, True)
        self.assertEqual(seen["real"], True)
        _assert_placed(self, seen, CPU, True)
        self.assertTrue(seen["images"].is_contiguous())
        calculator.update_batch(batch, False)
        score = calculator.compute()

        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        self.assertEqual(calculator._metric_device(), CPU)
        self.assertIsInstance(score, float)
        self.assertTrue(math.isfinite(score))
        self.assertLess(abs(score), 1e-4)
        _assert_unchanged(self, batch, snapshot)

    def test_tensor_to_same_device_returns_the_same_object(self):
        """``Tensor.to`` keeps identity and ``data_ptr``, including a view.

        Meta identity is the Python object. Every meta tensor reports
        ``data_ptr() == 0``.
        """
        cases = ((CPU, False), (CPU, True), (META, False), (META, True))
        for device, noncontiguous in cases:
            with self.subTest(device=str(device), noncontiguous=noncontiguous):
                tensor = _images(device=device, noncontiguous=noncontiguous)
                moved = tensor.to(tensor.device)
                self.assertIs(moved, tensor)
                self.assertEqual(moved.data_ptr(), tensor.data_ptr())
                self.assertEqual(moved.is_contiguous(), not noncontiguous)
                if device.type == "meta":
                    self.assertEqual(tensor.data_ptr(), 0)

    def test_noncontiguous_processed_tensor_keeps_identity(self):
        """A view that already matches the metric stays the object ``.to`` saw."""
        for device in (CPU, META):
            with self.subTest(device=str(device)):
                calculator, parent, calls = _ready()
                if device.type == "meta":
                    _apply_device(parent, META)
                seen = _watch(calculator, preprocessor=_noncontiguous_processed)
                batch = _images(device=device, noncontiguous=True)
                snapshot = _snapshot(batch)

                calculator.update_batch(batch, True)

                self.assertEqual(calls, [])
                self.assertEqual(calculator.device, CPU)
                self.assertFalse(seen["processed"].is_contiguous())
                _assert_unchanged(self, batch, snapshot)
                _assert_placed(self, seen, device, True)

    def test_parent_to_meta_does_not_call_child_to(self):
        """Positional and keyword ``Module.to(meta)`` do not call child ``.to``."""
        moves = (
            ("positional", ("meta",), {}),
            ("keyword", (), {"device": "meta"}),
        )
        for label, args, kwargs in moves:
            with self.subTest(move=label):
                calculator, parent, calls = _ready()
                batch = _images(noncontiguous=True)
                snapshot = _snapshot(batch)

                parent.to(*args, **kwargs)
                seen = _watch(calculator)
                calculator.update_batch(batch, True)

                self.assertEqual(calls, [])
                self.assertEqual(calculator.device, CPU)
                self.assertEqual(next(calculator.fid.parameters()).device, META)
                _assert_unchanged(self, batch, snapshot)
                _assert_placed(self, seen, META, False)
                self.assertEqual(seen["real"], True)

    def test_parent_apply_meta_places_cpu_batch_on_meta(self):
        """``_apply(meta)`` is the parent path. ``compute`` cannot read meta."""
        calculator, parent, calls = _ready()
        batch = _images(dtype=torch.float64, noncontiguous=True)
        snapshot = _snapshot(batch)
        _apply_device(parent, META)
        seen = _watch(calculator)

        calculator.update_batch(batch, True)

        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        self.assertEqual(batch.dtype, torch.float64)
        parameter = next(calculator.fid.parameters())
        self.assertEqual(parameter.device, META)
        self.assertEqual(parameter.dtype, torch.float32)
        _assert_unchanged(self, batch, snapshot)
        _assert_placed(self, seen, META, False)
        with self.assertRaises(RuntimeError) as caught:
            calculator.compute()
        self.assertIn("meta", str(caught.exception).lower())

    def test_meta_batch_already_aligned_keeps_storage(self):
        """A meta batch that already matches the weights stays the same object."""
        for noncontiguous in (False, True):
            with self.subTest(noncontiguous=noncontiguous):
                calculator, parent, calls = _ready()
                _apply_device(parent, META)
                batch = _images(device=META, noncontiguous=noncontiguous)
                self.assertEqual(batch.is_contiguous(), not noncontiguous)
                if noncontiguous:
                    seen = _watch(calculator, preprocessor=_noncontiguous_processed)
                else:
                    seen = _watch(calculator)

                calculator.update_batch(batch, False)

                self.assertEqual(calls, [])
                self.assertEqual(calculator.device, CPU)
                self.assertEqual(batch.device, META)
                self.assertEqual(seen["real"], False)
                _assert_placed(self, seen, META, True)
                if noncontiguous:
                    self.assertFalse(seen["images"].is_contiguous())

    def test_dtype_only_move_keeps_the_cpu_device(self):
        """``parent.to(dtype=...)`` stays on CPU and does not call child ``.to``.

        A poisoned ``meta`` cache must not pull the batch off CPU. Inception
        then rejects float64 weights. That dtype error is not the device mix.
        """
        calculator, parent, calls = _ready()
        calculator.device = META
        batch = _images(dtype=torch.float64, noncontiguous=True)
        snapshot = _snapshot(batch)

        parent.to(dtype=torch.float64)
        seen = _watch(calculator)
        with self.assertRaises(RuntimeError) as caught:
            calculator.update_batch(batch, True)

        parameter = next(calculator.fid.parameters())
        message = str(caught.exception)
        self.assertEqual(calls, [])
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(parameter.device, CPU)
        self.assertEqual(calculator.device, META)
        self.assertEqual(calculator._metric_device(), CPU)
        self.assertIn("DoubleTensor", message)
        self.assertNotIn("meta", message.lower())
        _assert_unchanged(self, batch, snapshot)
        _assert_placed(self, seen, CPU, True)

    def test_direct_dtype_move_is_the_recorded_call(self):
        """A direct ``fid.to(dtype=...)`` is the one recorded call and stays on CPU."""
        calculator, _parent, calls = _ready()
        calculator.device = META

        calculator.fid.to(dtype=torch.float64)

        parameter = next(calculator.fid.parameters())
        self.assertEqual(calls, [((), {"dtype": torch.float64})])
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(parameter.device, CPU)
        self.assertEqual(calculator.device, META)
        self.assertEqual(calculator._metric_device(), CPU)

    def test_device_is_reread_after_meta_round_trip(self):
        """Meta, a rejected ``.to(cpu)``, zeros, then meta again.

        ``parent.to(cpu)`` raises and leaves the parameters on meta. The
        return trip uses ``zeros_like`` through ``_apply``. The cache is
        poisoned on the CPU stop and must still lose. The noncontiguous
        caller is unchanged.
        """
        calculator, parent, calls = _ready()
        batch = _images(noncontiguous=True)
        snapshot = _snapshot(batch)
        seen = _watch(calculator)

        _apply_device(parent, META)
        calculator.update_batch(batch, True)
        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        _assert_placed(self, seen, META, False)

        with self.assertRaises(NotImplementedError) as blocked:
            parent.to(CPU)
        self.assertIn("meta", str(blocked.exception).lower())
        self.assertEqual(next(calculator.fid.parameters()).device, META)
        self.assertEqual(calls, [])

        _apply_device(parent, CPU)
        weight = next(calculator.fid.parameters())
        self.assertEqual(weight.device, CPU)
        self.assertEqual(weight.dtype, torch.float32)
        self.assertTrue(torch.isfinite(weight).all())
        self.assertEqual(weight.abs().sum().item(), 0.0)
        calculator.device = META
        calculator.update_batch(batch, True)
        calculator.update_batch(_images(), False)
        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, META)
        self.assertEqual(calculator._metric_device(), CPU)
        _assert_placed(self, seen, CPU, True)
        score = calculator.compute()
        self.assertTrue(math.isfinite(score))

        _apply_device(parent, META)
        calculator.update_batch(batch, True)
        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, META)
        self.assertEqual(next(calculator.fid.parameters()).device, META)
        _assert_unchanged(self, batch, snapshot)
        _assert_placed(self, seen, META, False)

    def test_each_update_rereads_parameter_device(self):
        """A move between two batches is visible to the next ``update_batch``."""
        calculator, parent, calls = _ready()
        batch = _images(noncontiguous=True)
        snapshot = _snapshot(batch)
        seen = _watch(calculator)

        calculator.update_batch(batch, True)
        self.assertIs(seen["images"], seen["processed"])
        self.assertEqual(seen["images"].device, CPU)
        _apply_device(parent, META)
        calculator.update_batch(batch, False)

        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        _assert_unchanged(self, batch, snapshot)
        _assert_placed(self, seen, META, False)
        self.assertEqual(seen["real"], False)

    def test_poisoned_construction_device_loses_to_parameters(self):
        """A ``meta`` or ``cuda`` cache cannot move CPU weights. No GPU is required."""
        for poisoned in (META, CUDA):
            with self.subTest(poisoned=str(poisoned)):
                calculator, _parent, calls = _ready()
                calculator.device = poisoned
                batch = _images(noncontiguous=True)
                snapshot = _snapshot(batch)
                seen = _watch(calculator)

                calculator.update_batch(batch, True)
                calculator.update_batch(batch, False)
                score = calculator.compute()

                self.assertEqual(calls, [])
                self.assertEqual(calculator.device, poisoned)
                self.assertEqual(next(calculator.fid.parameters()).device, CPU)
                self.assertEqual(calculator._metric_device(), CPU)
                _assert_unchanged(self, batch, snapshot)
                _assert_placed(self, seen, CPU, True)
                self.assertTrue(math.isfinite(score))
                self.assertLess(abs(score), 1e-4)

    def test_cache_placement_mixes_meta_weights_with_cpu_batches(self):
        """Restoring the construction cache reproduces the device-mix error.

        Deleting the override places the same caller on meta and does not
        call child ``.to``.
        """
        calculator, parent, calls = _ready()
        _apply_device(parent, META)
        batch = _images(noncontiguous=True)
        snapshot = _snapshot(batch)

        def _on_cache():
            return calculator.device

        calculator._metric_device = _on_cache
        with self.assertRaises(RuntimeError) as caught:
            calculator.update_batch(batch, True)
        message = str(caught.exception).lower()
        self.assertIn("meta", message)
        self.assertIn("cpu", message)
        _assert_unchanged(self, batch, snapshot)

        del calculator._metric_device
        seen = _watch(calculator)
        calculator.update_batch(batch, True)
        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        self.assertEqual(next(calculator.fid.parameters()).device, META)
        _assert_placed(self, seen, META, False)

    def test_meta_batch_against_cpu_weights_is_not_copied_out(self):
        """A meta batch is not copied onto CPU weights or a CUDA cache.

        The live parameter device is CPU, so placement uses ``Tensor.to(cpu)``
        and PyTorch refuses to read meta storage. The CUDA cache is what the
        pre-fix path would have selected.
        """
        calculator, _parent, calls = _ready()
        calculator.device = CUDA
        seen = _watch(calculator)
        batch = _images(device=META, noncontiguous=True)
        snapshot = _snapshot(batch)

        with self.assertRaises(NotImplementedError) as caught:
            calculator.update_batch(batch, True)

        self.assertIn("meta", str(caught.exception).lower())
        self.assertNotIn("images", seen)
        self.assertEqual(seen["processed"].device, META)
        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CUDA)
        self.assertEqual(next(calculator.fid.parameters()).device, CPU)
        self.assertEqual(calculator._metric_device(), CPU)
        _assert_unchanged(self, batch, snapshot)

    def test_metric_device_falls_back_without_parameters(self):
        """No parameters follows a buffer, then the construction preference.

        A CUDA cache loses to a CPU buffer. With no storage at all, the
        preference is the only remaining device, poisoned or not.
        """
        calculator, _parent, _calls = _ready()

        class _BufferOnly(nn.Module):
            """Module whose only storage is a buffer."""

            def __init__(self):
                super().__init__()
                self.register_buffer("marker", torch.ones(1))

            def forward(self, *_args, **_kwargs):
                """Unused. The test only reads the buffer device."""
                raise NotImplementedError

        buffered = _BufferOnly()
        calculator.fid = buffered
        calculator.device = CUDA
        self.assertEqual(calculator._metric_device(), CPU)

        parent = nn.Module()
        parent.metric = buffered
        _apply_device(parent, META)
        self.assertEqual(calculator.device, CUDA)
        self.assertEqual(calculator._metric_device(), META)

        calculator.fid = nn.Module()
        self.assertEqual(calculator._metric_device(), CUDA)
        calculator.device = META
        self.assertEqual(calculator._metric_device(), META)
        calculator.device = CPU
        self.assertEqual(calculator._metric_device(), CPU)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipUnless(helpers is not None, "requires qvae_mnist FID dependencies")
class TestFIDCalculatorCuda(unittest.TestCase):
    """End-to-end coverage of a parent move onto CUDA."""

    def test_parent_to_cuda_does_not_call_child_to(self):
        """``parent.to('cuda')`` and ``to(device='cuda')`` do not call child ``.to``."""
        moves = (
            ("positional", ("cuda",), {}),
            ("keyword", (), {"device": "cuda"}),
        )
        for label, args, kwargs in moves:
            with self.subTest(move=label):
                calculator, parent, calls = _ready()
                batch = _images(noncontiguous=True)
                snapshot = _snapshot(batch)

                parent.to(*args, **kwargs)
                seen = _watch(calculator)
                calculator.update_batch(batch, True)

                self.assertEqual(calls, [])
                self.assertEqual(calculator.device, CPU)
                self.assertEqual(next(calculator.fid.parameters()).device.type, "cuda")
                _assert_unchanged(self, batch, snapshot)
                _assert_placed(self, seen, CUDA, False)

    def test_parent_apply_cuda_scores_on_cuda(self):
        """A CPU batch is scored on CUDA while the cache stays on CPU."""
        calculator, parent, calls = _ready()
        batch = _images(noncontiguous=True)
        snapshot = _snapshot(batch)
        _apply_device(parent, CUDA)
        seen = _watch(calculator)

        calculator.update_batch(batch, True)
        calculator.update_batch(_images(), False)
        score = calculator.compute()

        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        self.assertEqual(calculator._metric_device().type, "cuda")
        _assert_unchanged(self, batch, snapshot)
        _assert_placed(self, seen, CUDA, False)
        self.assertIsInstance(score, float)
        self.assertTrue(math.isfinite(score))

    def test_cuda_batch_keeps_storage(self):
        """A CUDA batch that already matches the weights keeps its storage."""
        for noncontiguous in (False, True):
            with self.subTest(noncontiguous=noncontiguous):
                calculator, parent, calls = _ready()
                _apply_device(parent, CUDA)
                batch = _images(device=CUDA, noncontiguous=noncontiguous)
                pointer = batch.data_ptr()
                self.assertEqual(batch.is_contiguous(), not noncontiguous)
                if noncontiguous:
                    seen = _watch(calculator, preprocessor=_noncontiguous_processed)
                else:
                    seen = _watch(calculator)

                calculator.update_batch(batch, True)
                calculator.update_batch(batch, False)

                self.assertEqual(calls, [])
                self.assertEqual(calculator.device, CPU)
                self.assertEqual(batch.data_ptr(), pointer)
                self.assertEqual(batch.device.type, "cuda")
                _assert_placed(self, seen, CUDA, True)
                if noncontiguous:
                    self.assertFalse(seen["images"].is_contiguous())
                self.assertTrue(math.isfinite(calculator.compute()))

    def test_device_is_reread_across_cuda_and_cpu(self):
        """CUDA, CPU, CUDA. The CPU stop keeps identity and ignores a CUDA cache."""
        calculator, parent, calls = _ready()
        batch = _images(noncontiguous=True)
        snapshot = _snapshot(batch)
        seen = _watch(calculator)

        _apply_device(parent, CUDA)
        calculator.update_batch(batch, True)
        _assert_placed(self, seen, CUDA, False)

        _apply_device(parent, CPU)
        calculator.device = CUDA
        calculator.update_batch(batch, True)
        self.assertEqual(calculator.device, CUDA)
        self.assertEqual(calculator._metric_device(), CPU)
        _assert_placed(self, seen, CPU, True)

        _apply_device(parent, CUDA)
        calculator.update_batch(batch, False)
        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CUDA)
        _assert_unchanged(self, batch, snapshot)
        _assert_placed(self, seen, CUDA, False)

    def test_dtype_only_move_keeps_cuda_device(self):
        """A dtype-only parent move does not pull CUDA parameters back to CPU."""
        calculator, parent, calls = _ready()
        _apply_device(parent, CUDA)
        parent.to(dtype=torch.float64)
        batch = _images(dtype=torch.float64, device=CUDA, noncontiguous=True)
        pointer = batch.data_ptr()
        seen = _watch(calculator)

        with self.assertRaises(RuntimeError) as caught:
            calculator.update_batch(batch, True)

        parameter = next(calculator.fid.parameters())
        self.assertEqual(calls, [])
        self.assertEqual(parameter.device.type, "cuda")
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(calculator.device, CPU)
        self.assertEqual(calculator._metric_device().type, "cuda")
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertIn("DoubleTensor", str(caught.exception))
        _assert_placed(self, seen, CUDA, True)

    def test_stale_cpu_cache_does_not_pull_cuda_weights(self):
        """Forcing the cache back to CPU must not pull CUDA weights to the host."""
        calculator, parent, calls = _ready()
        _apply_device(parent, CUDA)
        calculator.device = CPU
        batch = _images(noncontiguous=True)
        snapshot = _snapshot(batch)
        seen = _watch(calculator)

        calculator.update_batch(batch, True)
        calculator.update_batch(batch, False)
        score = calculator.compute()

        self.assertEqual(calls, [])
        self.assertEqual(calculator.device, CPU)
        self.assertEqual(next(calculator.fid.parameters()).device.type, "cuda")
        _assert_unchanged(self, batch, snapshot)
        _assert_placed(self, seen, CUDA, False)
        self.assertTrue(math.isfinite(score))

    def test_cuda_batch_on_cpu_weights_keeps_caller_storage(self):
        """A CUDA batch is presented on CPU. Caller storage stays on CUDA."""
        for noncontiguous in (False, True):
            with self.subTest(noncontiguous=noncontiguous):
                calculator, _parent, calls = _ready()
                calculator.device = CUDA
                batch = _images(device=CUDA, noncontiguous=noncontiguous)
                pointer = batch.data_ptr()
                seen = _watch(calculator)

                calculator.update_batch(batch, True)

                self.assertEqual(calls, [])
                self.assertEqual(calculator.device, CUDA)
                self.assertEqual(next(calculator.fid.parameters()).device, CPU)
                self.assertEqual(batch.data_ptr(), pointer)
                self.assertEqual(batch.device.type, "cuda")
                self.assertEqual(batch.is_contiguous(), not noncontiguous)
                _assert_placed(self, seen, CPU, False)


if __name__ == "__main__":
    unittest.main()
