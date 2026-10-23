"""Regression tests for feature-selection classifier device alignment.

``TinyCNN``, ``SimpleRNN``, and ``SimpleLSTM`` feed the caller batch straight
into conv, recurrent, and linear weights. A parent ``nn.Module.to`` moves
those weights through ``_apply`` and does not call the child ``.to()``
(merged #164). A CPU batch then meets parameters on ``meta`` or CUDA:

``RuntimeError: Input and parameter tensors are not at the same device``

or the meta/CPU ``Tensor on device meta is not on the expected device cpu!``
raised by the CNN's linear head.

CPU cases use ``meta``, the same stand-in as ``tests/test_bm_device_sync.py``.
CUDA cases are skipped unless a GPU is present.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn

META = torch.device("meta")
CPU = torch.device("cpu")


def _load_models():
    """Load the example module without requiring a package install."""
    path = (
        Path(__file__).resolve().parents[1]
        / "example"
        / "feature_selection"
        / "feature_selection_models.py"
    )
    spec = importlib.util.spec_from_file_location(
        "feature_selection_models_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODELS = _load_models()


def _apply_device(module: nn.Module, device: torch.device) -> None:
    """Move parameters through ``_apply``, the parent-module path."""
    module._apply(lambda tensor: tensor.to(device))  # pylint: disable=protected-access


def _retarget_storage(device: torch.device):
    """Return an ``_apply`` callback that places empty storage on ``device``.

    ``Module.to`` cannot copy out of ``meta``. Replacing storage still goes
    through ``_apply``, so the parameter device changes without ``.to()``.
    """

    def _move(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == device:
            return tensor
        return torch.empty(tensor.shape, dtype=tensor.dtype, device=device)

    return _move


def _cases():
    """Return ``(model, cpu batch, logit shape)`` for each classifier."""
    return (
        (MODELS.SimpleLSTM(4), torch.randn(2, 5, 4), (2, 2)),
        (MODELS.SimpleRNN(4), torch.randn(2, 5, 4), (2, 2)),
        (MODELS.TinyCNN(4), torch.randn(2, 16), (2, 2)),
    )


class RecordingLSTM(MODELS.SimpleLSTM):  # pylint: disable=too-few-public-methods
    """LSTM that records every direct ``.to()`` invocation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.direct_to_calls: list[tuple[tuple, dict]] = []

    def to(self, *args, **kwargs):
        """Record and forward module moves."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


class TestFeatureSelectionParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_parent_to_does_not_call_child_to(self):
        """A parent move must relocate weights without calling ``.to()``."""
        model = RecordingLSTM(4)
        parent = nn.Module()
        parent.model = model

        parent.to(META)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(next(model.parameters()).device, META)

    def test_parent_apply_accepts_cpu_batch(self):
        """CPU batches stay put; logits follow the parameter device."""
        for model, batch, shape in _cases():
            with self.subTest(model=type(model).__name__):
                parent = nn.Module()
                parent.model = model
                _apply_device(parent, META)
                pointer = batch.data_ptr()

                logits = model(batch)

                self.assertEqual(batch.data_ptr(), pointer)
                self.assertEqual(batch.device, CPU)
                self.assertEqual(logits.device, META)
                self.assertEqual(logits.dtype, batch.dtype)
                self.assertEqual(tuple(logits.shape), shape)

    def test_input_already_on_parameter_device_keeps_storage(self):
        """A batch that already matches the weights is not copied."""
        model = MODELS.SimpleLSTM(4)
        _apply_device(model, META)
        batch = torch.empty(2, 5, 4, device=META)
        pointer = batch.data_ptr()

        logits = model(batch)

        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(logits.device, META)
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_dtype_only_move_keeps_cpu_device(self):
        """``.to(dtype=...)`` must not change the parameter device."""
        model = MODELS.SimpleRNN(4)
        model.to(dtype=torch.float64)
        batch = torch.randn(2, 5, 4, dtype=torch.float64)

        logits = model(batch)

        self.assertEqual(next(model.parameters()).device, CPU)
        self.assertEqual(next(model.parameters()).dtype, torch.float64)
        self.assertEqual(logits.device, CPU)
        self.assertEqual(logits.dtype, torch.float64)
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_float64_parameters_survive_meta_move(self):
        """A later ``_apply(meta)`` must not recast float64 weights."""
        model = MODELS.TinyCNN(4)
        model.to(dtype=torch.float64)
        _apply_device(model, META)
        batch = torch.randn(2, 16, dtype=torch.float64)
        pointer = batch.data_ptr()

        logits = model(batch)

        self.assertEqual(next(model.parameters()).dtype, torch.float64)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(batch.device, CPU)
        self.assertEqual(logits.device, META)
        self.assertEqual(logits.dtype, torch.float64)

    def test_parameter_device_is_reread_after_meta_round_trip(self):
        """The helper must follow parameters off ``meta`` and back."""
        model = RecordingLSTM(4)
        parent = nn.Module()
        parent.model = model
        batch = torch.randn(2, 5, 4)

        _apply_device(parent, META)
        self.assertEqual(model(batch).device, META)
        parent._apply(_retarget_storage(CPU))  # pylint: disable=protected-access
        self.assertEqual(model(batch).device, CPU)
        _apply_device(parent, META)
        self.assertEqual(model(batch).device, META)
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(batch.device, CPU)

    def test_parameter_device_uses_live_parameters_or_fallback(self):
        """An empty module follows the fallback; real weights win after a move."""
        # The helper is intentionally private; this test pins its contract.
        # pylint: disable=protected-access
        empty = nn.Module()
        fallback = torch.zeros(1)
        self.assertEqual(MODELS._parameter_device(empty, fallback), CPU)

        model = MODELS.SimpleRNN(4)
        self.assertEqual(MODELS._parameter_device(model, fallback), CPU)
        _apply_device(model, META)
        self.assertEqual(MODELS._parameter_device(model, fallback), META)

    def test_same_device_alignment_keeps_autograd(self):
        """CPU alignment must still train the batch and the weights."""
        model = MODELS.SimpleLSTM(4)
        batch = torch.randn(2, 5, 4, requires_grad=True)

        logits = model(batch)
        logits.sum().backward()

        weight = next(model.parameters())
        self.assertEqual(logits.device, CPU)
        self.assertIsNotNone(batch.grad)
        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.isfinite(batch.grad).all().item())
        self.assertTrue(torch.isfinite(weight.grad).all().item())


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFeatureSelectionParameterDeviceCUDA(unittest.TestCase):
    """End-to-end coverage of a parent move onto CUDA."""

    def test_parent_apply_accepts_cpu_batch(self):
        """CPU batches stay put; CUDA logits match the pre-move values."""
        cuda = torch.device("cuda")
        for model, batch, shape in _cases():
            with self.subTest(model=type(model).__name__):
                expected = model(batch).detach()
                parent = nn.Module()
                parent.model = model
                pointer = batch.data_ptr()
                _apply_device(parent, cuda)

                logits = model(batch)

                self.assertEqual(batch.data_ptr(), pointer)
                self.assertEqual(batch.device, CPU)
                self.assertEqual(logits.device.type, "cuda")
                self.assertEqual(tuple(logits.shape), shape)
                self.assertTrue(torch.isfinite(logits).all().item())
                torch.testing.assert_close(logits.cpu(), expected)

    def test_device_is_reread_across_cuda_and_cpu(self):
        """CUDA → CPU → CUDA must follow parameters, not a cached device."""
        model = RecordingLSTM(4)
        parent = nn.Module()
        parent.model = model
        batch = torch.randn(2, 5, 4)
        cuda = torch.device("cuda")

        _apply_device(parent, cuda)
        self.assertEqual(model(batch).device.type, "cuda")
        _apply_device(parent, CPU)
        cpu_logits = model(batch)
        self.assertEqual(cpu_logits.device, CPU)
        self.assertTrue(torch.isfinite(cpu_logits).all().item())
        _apply_device(parent, cuda)
        self.assertEqual(model(batch).device.type, "cuda")
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(batch.device, CPU)

    def test_backward_from_cpu_batch_reaches_cuda_weights(self):
        """A CPU batch must still produce a CPU input gradient."""
        model = MODELS.SimpleRNN(4)
        parent = nn.Module()
        parent.model = model
        _apply_device(parent, torch.device("cuda"))
        batch = torch.randn(2, 5, 4, requires_grad=True)
        pointer = batch.data_ptr()

        logits = model(batch)
        logits.sum().backward()

        self.assertEqual(pointer, batch.data_ptr())
        self.assertEqual(batch.grad.device, CPU)
        self.assertEqual(next(model.parameters()).grad.device.type, "cuda")
        self.assertTrue(torch.isfinite(batch.grad).all().item())
        self.assertTrue(torch.isfinite(next(model.parameters()).grad).all().item())


if __name__ == "__main__":
    unittest.main()
