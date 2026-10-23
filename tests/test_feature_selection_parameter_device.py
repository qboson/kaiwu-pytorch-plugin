"""Regression tests for feature-selection classifier device alignment.

``TinyCNN``, ``SimpleRNN``, and ``SimpleLSTM`` feed the caller batch straight
into conv, recurrent, and linear weights. A parent ``nn.Module.to`` moves
those weights through ``_apply`` and does not call the child ``.to()``
(merged #164 / ``tests/test_bm_device_sync.py``). A CPU batch then meets
parameters on ``meta`` or CUDA:

``RuntimeError: Tensor on device meta is not on the expected device cpu!``

or ``Input and parameter tensors are not at the same device``.

The live parameter device is reread on every forward. Dtypes stay put.
Same-device callers keep object identity and ``data_ptr``. ``.to(cpu)``
cannot copy out of a meta tensor, so the return trip retargets storage with
``_apply``. These three classifiers have no embedding index, attention mask,
or dropout input. The linear head is the second parameter group: it consumes
activations that already sit on the parameter device.

CPU cases use ``meta``. CUDA cases are skipped unless a GPU is present, and
they are still collected.
"""

# Parent moves go through ``_apply`` (see merged #164).
# pylint: disable=protected-access

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn

META = torch.device("meta")
CPU = torch.device("cpu")
LOGITS = (2, 2)


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


def _classifiers():
    """Return one fresh instance of each feature-selection classifier."""
    return (
        MODELS.TinyCNN(4),
        MODELS.SimpleRNN(4),
        MODELS.SimpleLSTM(4),
    )


def _batch(
    model: nn.Module,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device = CPU,
    noncontiguous: bool = False,
    requires_grad: bool = False,
    meta: bool = False,
) -> torch.Tensor:
    """Return a batch shaped for ``model``.

    ``meta`` builds empty storage, because a meta tensor has no values.
    ``noncontiguous`` is a column slice, which is the layout a feature
    selection can hand the classifier. ``requires_grad`` is only combined
    with a contiguous leaf.
    """
    if meta:
        device = META

    def _allocate(shape: tuple[int, ...]) -> torch.Tensor:
        """Empty storage on meta; real values on every other device."""
        if meta:
            return torch.empty(*shape, dtype=dtype, device=device)
        return torch.randn(*shape, dtype=dtype, device=device)

    cnn = isinstance(model, MODELS.TinyCNN)
    if noncontiguous and cnn:
        batch = _allocate((2, 32))[:, :16]
    elif noncontiguous:
        batch = _allocate((2, 5, 8))[:, :, :4]
    elif cnn:
        batch = _allocate((2, 16))
    else:
        batch = _allocate((2, 5, 4))
    if requires_grad:
        batch.requires_grad_(True)
    return batch


def _cases(**kwargs):
    """Return ``(model, batch, logit shape)`` for each classifier."""
    built = []
    for model in _classifiers():
        built.append((model, _batch(model, **kwargs), LOGITS))
    return tuple(built)


def _nest(model: nn.Module) -> nn.Module:
    """Return a parent whose ``_apply`` moves ``model``."""
    parent = nn.Module()
    parent.model = model
    return parent


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


def _apply_device(module: nn.Module, device: torch.device) -> None:
    """Move parameters through ``_apply``, the parent-module path."""
    module._apply(lambda tensor: tensor.to(device))


def _retarget_storage(device: torch.device):
    """Return an ``_apply`` callback that places new storage on ``device``.

    ``Module.to`` cannot copy out of ``meta``. Replacing storage still goes
    through ``_apply``, so the parameter device changes without ``.to()``.
    Zeros are readable, which is what a later CPU score needs. The callback
    does not read the source tensor.
    """

    def _move(tensor: torch.Tensor) -> torch.Tensor:
        """Place ``tensor`` on ``device`` without reading its data."""
        if tensor.device == device:
            return tensor
        return torch.zeros(tensor.shape, dtype=tensor.dtype, device=device)

    return _move


def _assert_cannot_copy_out_of_meta(case: unittest.TestCase) -> None:
    """``.to(cpu)`` cannot materialise a meta tensor or a meta module."""
    meta_value = torch.empty(1, device=META)
    with case.assertRaises(NotImplementedError) as caught:
        meta_value.to(CPU)
    case.assertIn("meta", str(caught.exception).lower())

    module = nn.Linear(2, 2)
    _apply_device(module, META)
    with case.assertRaises(NotImplementedError) as caught:
        module.to(CPU)
    case.assertIn("meta", str(caught.exception).lower())
    case.assertEqual(next(module.parameters()).device, META)


def _snapshot(tensor: torch.Tensor):
    """Capture storage identity. Meta tensors are not read."""
    values = None
    if tensor.device.type != "meta":
        values = tensor.detach().clone()
    return (
        tensor.data_ptr(),
        tensor.device,
        tensor.dtype,
        tensor.is_contiguous(),
        values,
    )


def _assert_unchanged(case: unittest.TestCase, tensor: torch.Tensor, snap) -> None:
    """Assert ``tensor`` kept its storage, device, dtype, layout, and values."""
    pointer, device, dtype, contiguous, values = snap
    case.assertEqual(tensor.data_ptr(), pointer)
    case.assertEqual(tensor.device, device)
    case.assertEqual(tensor.dtype, dtype)
    case.assertEqual(tensor.is_contiguous(), contiguous)
    if values is not None:
        case.assertTrue(torch.equal(tensor, values))


def _first_stage(model: nn.Module) -> nn.Module:
    """Return the conv or recurrent module that consumes the caller batch."""
    if isinstance(model, MODELS.TinyCNN):
        return model.net[0]
    if isinstance(model, MODELS.SimpleRNN):
        return model.rnn
    return model.lstm


def _head(model: nn.Module) -> nn.Linear:
    """Return the linear head, the second parameter group."""
    if isinstance(model, MODELS.TinyCNN):
        return model.net[-1]
    return model.head


def _spy(module: nn.Module) -> dict:
    """Record the tensor ``module`` actually receives."""
    seen: dict = {}
    original = module.forward

    def _wrapped(*args, **kwargs):
        """Store the first positional input, then run the real forward."""
        if args:
            seen["inputs"] = args[0]
        return original(*args, **kwargs)

    module.forward = _wrapped
    return seen


def _spy_stages(model: nn.Module):
    """Spy the first stage and the linear head."""
    return _spy(_first_stage(model)), _spy(_head(model))


def _assert_logits(
    case: unittest.TestCase,
    logits: torch.Tensor,
    batch: torch.Tensor,
    device: torch.device,
    shape: tuple[int, int],
    *,
    finite: bool,
) -> None:
    """Assert logit device, dtype, and shape. Read values only off meta."""
    case.assertEqual(logits.device, device)
    case.assertEqual(logits.dtype, batch.dtype)
    case.assertEqual(tuple(logits.shape), shape)
    if finite:
        case.assertTrue(torch.isfinite(logits).all().item())


def _assert_stage_devices(
    case: unittest.TestCase,
    model: nn.Module,
    batch: torch.Tensor,
    first_seen: dict,
    head_seen: dict,
    parameter_device: torch.device,
) -> None:
    """Both stages must observe ``parameter_device``.

    A recurrent input is the aligned caller tensor. The CNN reshapes that
    tensor, so the conv input shares storage when the devices already match
    and is a different object either way. The head always sees activations,
    not the caller batch.
    """
    observed = first_seen["inputs"]
    case.assertEqual(observed.device, parameter_device)
    case.assertEqual(observed.dtype, batch.dtype)
    if batch.device == parameter_device:
        case.assertEqual(observed.data_ptr(), batch.data_ptr())
        if isinstance(model, MODELS.TinyCNN):
            case.assertIsNot(observed, batch)
            case.assertEqual(
                tuple(observed.shape),
                (batch.shape[0], 1, model.image_size, model.image_size),
            )
        else:
            case.assertIs(observed, batch)
    else:
        case.assertIsNot(observed, batch)

    head = _head(model)
    case.assertEqual(head_seen["inputs"].device, head.weight.device)
    case.assertEqual(head_seen["inputs"].device, parameter_device)
    case.assertEqual(head_seen["inputs"].dtype, head.weight.dtype)
    case.assertIsNot(head_seen["inputs"], batch)


def _assert_no_index_or_dropout_inputs(
    case: unittest.TestCase, model: nn.Module
) -> None:
    """No embedding, mask, or dropout module consumes a second caller tensor.

    Recurrent ``dropout`` is zero and there is a single layer, so cuDNN-style
    dropout never builds a mask. The second tensor in the forward is the
    activation consumed by the linear head.
    """
    banned = (nn.Embedding, nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)
    for module in model.modules():
        case.assertNotIsInstance(module, banned)
    case.assertIsInstance(_head(model), nn.Linear)
    case.assertIsInstance(_first_stage(model), (nn.Conv2d, nn.RNN, nn.LSTM))
    if isinstance(model, MODELS.SimpleRNN):
        case.assertEqual(model.rnn.dropout, 0.0)
        case.assertEqual(model.rnn.num_layers, 1)
    if isinstance(model, MODELS.SimpleLSTM):
        case.assertEqual(model.lstm.dropout, 0.0)
        case.assertEqual(model.lstm.num_layers, 1)


def _assert_parameters(
    case: unittest.TestCase, model: nn.Module, device, dtype
) -> None:
    """Every parameter, including the linear head, lives on ``device``."""
    parameters = tuple(model.parameters())
    case.assertGreaterEqual(len(parameters), 4)
    for parameter in parameters:
        case.assertEqual(parameter.device, device)
        if dtype is not None:
            case.assertEqual(parameter.dtype, dtype)
    case.assertEqual(list(model.named_buffers()), [])


class TestFeatureSelectionParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine, including CPU-only CI."""

    def test_cpu_forward_keeps_caller_storage_and_identity(self):
        """An unmoved CPU model reuses the caller tensor and stays finite."""
        for model, batch, shape in _cases():
            with self.subTest(model=type(model).__name__):
                snap = _snapshot(batch)
                first_seen, head_seen = _spy_stages(model)

                logits = model(batch)

                self.assertIs(MODELS._on_parameter_device(model, batch), batch)
                self.assertEqual(batch.data_ptr(), snap[0])
                _assert_unchanged(self, batch, snap)
                _assert_logits(self, logits, batch, CPU, shape, finite=True)
                _assert_stage_devices(self, model, batch, first_seen, head_seen, CPU)

    def test_parent_to_meta_skips_child_to(self):
        """Positional and keyword ``Module.to(meta)`` do not call child ``.to``."""
        for keyword in (False, True):
            for model in _classifiers():
                with self.subTest(model=type(model).__name__, keyword=keyword):
                    parent = _nest(model)
                    calls = _record_to(model)
                    if keyword:
                        parent.to(device=META)
                    else:
                        parent.to(META)

                    self.assertEqual(calls, [])
                    _assert_parameters(self, model, META, None)

    def test_parent_apply_accepts_cpu_batch(self):
        """``parent._apply(meta)`` accepts a CPU batch and does not call ``.to``.

        The conv or recurrent module sees a meta tensor. The linear head sees
        meta activations. The caller keeps storage, values, and CPU device.
        """
        for model, batch, shape in _cases():
            with self.subTest(model=type(model).__name__):
                parent = _nest(model)
                calls = _record_to(model)
                snap = _snapshot(batch)
                first_seen, head_seen = _spy_stages(model)
                _assert_no_index_or_dropout_inputs(self, model)

                _apply_device(parent, META)
                logits = model(batch)

                self.assertEqual(calls, [])
                _assert_parameters(self, model, META, torch.float32)
                _assert_unchanged(self, batch, snap)
                aligned = MODELS._on_parameter_device(model, batch)
                self.assertIsNot(aligned, batch)
                self.assertEqual(aligned.device, META)
                self.assertEqual(aligned.dtype, batch.dtype)
                _assert_logits(self, logits, batch, META, shape, finite=False)
                _assert_stage_devices(self, model, batch, first_seen, head_seen, META)

    def test_noncontiguous_caller_keeps_storage(self):
        """A noncontiguous view keeps storage on CPU and again after meta."""
        for model, batch, shape in _cases(noncontiguous=True):
            with self.subTest(model=type(model).__name__, where="cpu"):
                self.assertFalse(batch.is_contiguous())
                snap = _snapshot(batch)
                first_seen, head_seen = _spy_stages(model)

                logits = model(batch)

                self.assertIs(MODELS._on_parameter_device(model, batch), batch)
                _assert_unchanged(self, batch, snap)
                _assert_logits(self, logits, batch, CPU, shape, finite=True)
                _assert_stage_devices(self, model, batch, first_seen, head_seen, CPU)

        for model, batch, shape in _cases(noncontiguous=True):
            with self.subTest(model=type(model).__name__, where="meta"):
                parent = _nest(model)
                calls = _record_to(model)
                snap = _snapshot(batch)
                first_seen, head_seen = _spy_stages(model)
                _apply_device(parent, META)

                logits = model(batch)

                self.assertEqual(calls, [])
                self.assertFalse(batch.is_contiguous())
                _assert_unchanged(self, batch, snap)
                _assert_logits(self, logits, batch, META, shape, finite=False)
                _assert_stage_devices(self, model, batch, first_seen, head_seen, META)

    def test_meta_batch_already_aligned_keeps_storage(self):
        """A meta batch that already matches the weights keeps its storage."""
        for model in _classifiers():
            with self.subTest(model=type(model).__name__):
                _apply_device(model, META)
                batch = _batch(model, meta=True)
                pointer = batch.data_ptr()
                first_seen, head_seen = _spy_stages(model)

                logits = model(batch)

                self.assertIs(MODELS._on_parameter_device(model, batch), batch)
                self.assertEqual(batch.data_ptr(), pointer)
                self.assertEqual(batch.device, META)
                _assert_logits(self, logits, batch, META, LOGITS, finite=False)
                _assert_stage_devices(self, model, batch, first_seen, head_seen, META)

    def test_parent_dtype_only_move_keeps_cpu(self):
        """``parent.to(dtype=...)`` keeps CPU and does not call the child ``.to``.

        The caller dtype is not rewritten. A matching float64 batch keeps
        identity and scores finite float64 logits.
        """
        for model in _classifiers():
            with self.subTest(model=type(model).__name__):
                parent = _nest(model)
                calls = _record_to(model)
                parent.to(dtype=torch.float64)
                batch = _batch(model, dtype=torch.float64)
                snap = _snapshot(batch)

                logits = model(batch)

                self.assertEqual(calls, [])
                _assert_parameters(self, model, CPU, torch.float64)
                self.assertIs(MODELS._on_parameter_device(model, batch), batch)
                _assert_unchanged(self, batch, snap)
                _assert_logits(self, logits, batch, CPU, LOGITS, finite=True)

    def test_direct_dtype_only_stays_on_cpu(self):
        """A direct dtype conversion stays on CPU and scores finite logits."""
        for model in _classifiers():
            with self.subTest(model=type(model).__name__):
                calls = _record_to(model)
                returned = model.to(dtype=torch.float64)
                batch = _batch(model, dtype=torch.float64)
                snap = _snapshot(batch)

                logits = model(batch)

                self.assertIs(returned, model)
                self.assertEqual(calls, [((), {"dtype": torch.float64})])
                _assert_parameters(self, model, CPU, torch.float64)
                self.assertIs(MODELS._on_parameter_device(model, batch), batch)
                _assert_unchanged(self, batch, snap)
                _assert_logits(self, logits, batch, CPU, LOGITS, finite=True)

    def test_float64_parameters_survive_meta_apply(self):
        """Moving float64 weights onto meta must not recast or rewrite callers."""
        for model in _classifiers():
            with self.subTest(model=type(model).__name__):
                parent = _nest(model)
                parent.to(dtype=torch.float64)
                calls = _record_to(model)
                _apply_device(parent, META)
                batch = _batch(model, dtype=torch.float64)
                snap = _snapshot(batch)
                first_seen, head_seen = _spy_stages(model)

                logits = model(batch)

                self.assertEqual(calls, [])
                _assert_parameters(self, model, META, torch.float64)
                _assert_unchanged(self, batch, snap)
                self.assertIsNot(MODELS._on_parameter_device(model, batch), batch)
                _assert_logits(self, logits, batch, META, LOGITS, finite=False)
                _assert_stage_devices(self, model, batch, first_seen, head_seen, META)

    def test_parameter_device_is_reread_after_leaving_meta(self):
        """Each move is reread. ``.to(cpu)`` cannot copy out of meta.

        ``_apply`` still retargets storage with zeros, and the next logits
        follow that device. The child ``.to`` is not called.
        """
        _assert_cannot_copy_out_of_meta(self)
        for model, batch, shape in _cases():
            with self.subTest(model=type(model).__name__):
                parent = _nest(model)
                calls = _record_to(model)
                snap = _snapshot(batch)
                first_seen, head_seen = _spy_stages(model)

                logits = model(batch)
                self.assertEqual(MODELS._parameter_device(model, batch), CPU)
                self.assertIs(MODELS._on_parameter_device(model, batch), batch)
                _assert_logits(self, logits, batch, CPU, shape, finite=True)

                _apply_device(parent, META)
                logits = model(batch)
                self.assertEqual(calls, [])
                self.assertEqual(MODELS._parameter_device(model, batch), META)
                self.assertIsNot(MODELS._on_parameter_device(model, batch), batch)
                _assert_logits(self, logits, batch, META, shape, finite=False)
                _assert_stage_devices(self, model, batch, first_seen, head_seen, META)

                with self.assertRaises(NotImplementedError) as caught:
                    parent.to(CPU)
                self.assertIn("meta", str(caught.exception).lower())
                self.assertEqual(calls, [])
                _assert_parameters(self, model, META, torch.float32)
                self.assertEqual(model(batch).device, META)

                parent._apply(_retarget_storage(CPU))
                logits = model(batch)
                self.assertEqual(calls, [])
                self.assertEqual(MODELS._parameter_device(model, batch), CPU)
                self.assertIs(MODELS._on_parameter_device(model, batch), batch)
                _assert_logits(self, logits, batch, CPU, shape, finite=True)
                _assert_stage_devices(self, model, batch, first_seen, head_seen, CPU)

                _apply_device(parent, META)
                logits = model(batch)
                self.assertEqual(calls, [])
                self.assertEqual(MODELS._parameter_device(model, batch), META)
                self.assertIsNot(first_seen["inputs"], batch)
                _assert_logits(self, logits, batch, META, shape, finite=False)
                _assert_unchanged(self, batch, snap)

    def test_parameter_device_falls_back_without_parameters(self):
        """An empty module follows the fallback; real weights win after a move."""
        empty = nn.Module()
        cpu_fallback = torch.zeros(1)
        meta_fallback = torch.empty(1, device=META)
        self.assertIsNone(next(empty.parameters(), None))
        self.assertEqual(MODELS._parameter_device(empty, cpu_fallback), CPU)
        self.assertEqual(MODELS._parameter_device(empty, meta_fallback), META)
        self.assertIs(MODELS._on_parameter_device(empty, cpu_fallback), cpu_fallback)
        self.assertIs(MODELS._on_parameter_device(empty, meta_fallback), meta_fallback)

        for model in _classifiers():
            with self.subTest(model=type(model).__name__):
                self.assertEqual(MODELS._parameter_device(model, meta_fallback), CPU)
                _apply_device(model, META)
                self.assertEqual(MODELS._parameter_device(model, cpu_fallback), META)

    def test_same_device_alignment_keeps_autograd(self):
        """CPU alignment must train the batch and every weight, including the head."""
        torch.manual_seed(0)
        for model, batch, _shape in _cases(requires_grad=True):
            with self.subTest(model=type(model).__name__):
                pointer = batch.data_ptr()

                logits = model(batch)
                self.assertTrue(logits.requires_grad)
                logits.sum().backward()

                self.assertEqual(batch.data_ptr(), pointer)
                self.assertEqual(batch.device, CPU)
                self.assertIsNotNone(batch.grad)
                self.assertEqual(batch.grad.device, CPU)
                self.assertTrue(torch.isfinite(batch.grad).all().item())
                self.assertGreater(float(batch.grad.abs().sum()), 0.0)
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertEqual(parameter.grad.device, CPU)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_unaligned_cpu_batch_raises_device_mix(self):
        """Restoring the caller-device input reproduces the meta/CPU error."""
        original = MODELS._on_parameter_device

        def _leave_on_caller_device(module, inputs):
            """Return the pre-fix input, still on the caller device."""
            del module
            return inputs

        MODELS._on_parameter_device = _leave_on_caller_device
        try:
            for model, batch, _shape in _cases():
                with self.subTest(model=type(model).__name__):
                    snap = _snapshot(batch)
                    _apply_device(model, META)
                    with self.assertRaises(RuntimeError) as caught:
                        model(batch)
                    message = str(caught.exception).lower()
                    self.assertIn("meta", message)
                    self.assertIn("cpu", message)
                    _assert_unchanged(self, batch, snap)
                    self.assertEqual(next(model.parameters()).device, META)
        finally:
            MODELS._on_parameter_device = original


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFeatureSelectionParameterDeviceCUDA(unittest.TestCase):
    """Parent moves onto CUDA. Collected and skipped on machines without a GPU."""

    def test_parent_move_accepts_cpu_batch(self):
        """``parent._apply`` and ``parent.to('cuda')`` score a CPU batch on CUDA.

        Values match the pre-move CPU forward. The child ``.to`` is not called.
        """
        cuda = torch.device("cuda")
        for how in ("apply", "to"):
            for model, batch, shape in _cases():
                with self.subTest(how=how, model=type(model).__name__):
                    expected = model(batch).detach()
                    parent = _nest(model)
                    calls = _record_to(model)
                    snap = _snapshot(batch)
                    first_seen, head_seen = _spy_stages(model)
                    if how == "apply":
                        _apply_device(parent, cuda)
                    else:
                        parent.to("cuda")

                    logits = model(batch)

                    parameter_device = next(model.parameters()).device
                    self.assertEqual(calls, [])
                    self.assertEqual(parameter_device.type, "cuda")
                    _assert_parameters(self, model, parameter_device, torch.float32)
                    _assert_unchanged(self, batch, snap)
                    _assert_logits(
                        self, logits, batch, parameter_device, shape, finite=True
                    )
                    _assert_stage_devices(
                        self, model, batch, first_seen, head_seen, parameter_device
                    )
                    torch.testing.assert_close(
                        logits.detach().cpu(), expected, rtol=1e-4, atol=1e-5
                    )

    def test_batch_already_on_cuda_keeps_storage(self):
        """A CUDA batch that already matches the weights keeps its storage."""
        cuda = torch.device("cuda")
        for noncontiguous in (False, True):
            for model in _classifiers():
                with self.subTest(
                    model=type(model).__name__, noncontiguous=noncontiguous
                ):
                    _apply_device(model, cuda)
                    batch = _batch(model, device=cuda, noncontiguous=noncontiguous)
                    if noncontiguous:
                        self.assertFalse(batch.is_contiguous())
                    pointer = batch.data_ptr()
                    calls = _record_to(model)

                    logits = model(batch)

                    parameter_device = next(model.parameters()).device
                    self.assertEqual(calls, [])
                    self.assertEqual(parameter_device.type, "cuda")
                    self.assertIs(MODELS._on_parameter_device(model, batch), batch)
                    self.assertEqual(batch.data_ptr(), pointer)
                    self.assertEqual(batch.device, parameter_device)
                    _assert_logits(
                        self, logits, batch, parameter_device, LOGITS, finite=True
                    )

    def test_device_is_reread_across_cuda_and_cpu(self):
        """CUDA, then CPU, then CUDA follows parameters and does not call ``.to``."""
        cuda = torch.device("cuda")
        for model, batch, shape in _cases():
            with self.subTest(model=type(model).__name__):
                expected = model(batch).detach()
                parent = _nest(model)
                calls = _record_to(model)
                snap = _snapshot(batch)
                first_seen, head_seen = _spy_stages(model)

                _apply_device(parent, cuda)
                logits = model(batch)
                parameter_device = next(model.parameters()).device
                self.assertEqual(parameter_device.type, "cuda")
                _assert_logits(
                    self, logits, batch, parameter_device, shape, finite=True
                )
                _assert_stage_devices(
                    self, model, batch, first_seen, head_seen, parameter_device
                )
                torch.testing.assert_close(
                    logits.detach().cpu(), expected, rtol=1e-4, atol=1e-5
                )

                _apply_device(parent, CPU)
                logits = model(batch)
                self.assertEqual(MODELS._parameter_device(model, batch), CPU)
                self.assertIs(MODELS._on_parameter_device(model, batch), batch)
                _assert_logits(self, logits, batch, CPU, shape, finite=True)
                _assert_stage_devices(self, model, batch, first_seen, head_seen, CPU)
                torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-5)

                _apply_device(parent, cuda)
                logits = model(batch)
                parameter_device = next(model.parameters()).device
                self.assertEqual(calls, [])
                self.assertEqual(parameter_device.type, "cuda")
                self.assertIsNot(MODELS._on_parameter_device(model, batch), batch)
                _assert_unchanged(self, batch, snap)
                _assert_logits(
                    self, logits, batch, parameter_device, shape, finite=True
                )
                torch.testing.assert_close(
                    logits.detach().cpu(), expected, rtol=1e-4, atol=1e-5
                )

    def test_float64_survives_cuda_move(self):
        """float64 parameters stay float64 after a parent CUDA ``_apply``."""
        for model in _classifiers():
            with self.subTest(model=type(model).__name__):
                parent = _nest(model)
                parent.to(dtype=torch.float64)
                batch = _batch(model, dtype=torch.float64)
                expected = model(batch).detach()
                calls = _record_to(model)
                snap = _snapshot(batch)
                _apply_device(parent, torch.device("cuda"))

                logits = model(batch)

                parameter_device = next(model.parameters()).device
                self.assertEqual(calls, [])
                self.assertEqual(parameter_device.type, "cuda")
                _assert_parameters(self, model, parameter_device, torch.float64)
                _assert_unchanged(self, batch, snap)
                _assert_logits(
                    self, logits, batch, parameter_device, LOGITS, finite=True
                )
                torch.testing.assert_close(
                    logits.detach().cpu(), expected, rtol=1e-4, atol=1e-5
                )

    def test_dtype_only_move_keeps_cuda(self):
        """``parent.to(dtype=...)`` after a CUDA move stays on CUDA.

        The child ``.to`` is not called. A matching float64 caller is copied
        onto that device; its dtype is not rewritten on the caller itself.
        """
        for model in _classifiers():
            with self.subTest(model=type(model).__name__):
                parent = _nest(model)
                calls = _record_to(model)
                _apply_device(parent, torch.device("cuda"))
                parent.to(dtype=torch.float64)
                batch = _batch(model, dtype=torch.float64)
                snap = _snapshot(batch)

                logits = model(batch)

                parameter_device = next(model.parameters()).device
                self.assertEqual(calls, [])
                self.assertEqual(parameter_device.type, "cuda")
                _assert_parameters(self, model, parameter_device, torch.float64)
                _assert_unchanged(self, batch, snap)
                self.assertIsNot(MODELS._on_parameter_device(model, batch), batch)
                _assert_logits(
                    self, logits, batch, parameter_device, LOGITS, finite=True
                )

    def test_backward_from_cpu_batch_reaches_cuda_weights(self):
        """A CPU batch backpropagates into CUDA weights and a CPU input grad."""
        torch.manual_seed(0)
        cuda = torch.device("cuda")
        for model, batch, _shape in _cases(requires_grad=True):
            with self.subTest(model=type(model).__name__):
                parent = _nest(model)
                calls = _record_to(model)
                _apply_device(parent, cuda)
                pointer = batch.data_ptr()

                logits = model(batch)
                logits.sum().backward()

                self.assertEqual(calls, [])
                self.assertEqual(batch.data_ptr(), pointer)
                self.assertEqual(batch.device, CPU)
                self.assertEqual(batch.grad.device, CPU)
                self.assertTrue(torch.isfinite(batch.grad).all().item())
                self.assertGreater(float(batch.grad.abs().sum()), 0.0)
                for parameter in model.parameters():
                    self.assertEqual(parameter.device.type, "cuda")
                    self.assertIsNotNone(parameter.grad)
                    self.assertEqual(parameter.grad.device, parameter.device)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_cuda_batch_follows_cpu_parameters(self):
        """A CUDA batch is copied onto CPU parameters. Caller storage stays put."""
        cuda = torch.device("cuda")
        for model, _cpu_batch, shape in _cases():
            with self.subTest(model=type(model).__name__):
                batch = _batch(model, device=cuda)
                expected = model(batch.detach().cpu())
                pointer = batch.data_ptr()
                snap = _snapshot(batch)

                logits = model(batch)

                self.assertEqual(next(model.parameters()).device, CPU)
                self.assertIsNot(MODELS._on_parameter_device(model, batch), batch)
                self.assertEqual(MODELS._on_parameter_device(model, batch).device, CPU)
                _assert_unchanged(self, batch, snap)
                self.assertEqual(batch.data_ptr(), pointer)
                self.assertEqual(batch.device.type, "cuda")
                _assert_logits(self, logits, batch, CPU, shape, finite=True)
                torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
