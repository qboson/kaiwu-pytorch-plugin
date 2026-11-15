"""Regression tests for ``DEQForecaster`` parameter-device alignment.

``DEQForecaster`` embeds the caller history with ``nn.Linear`` weights. A
parent ``nn.Module._apply`` (the path ``nn.Module.to`` uses for children;
merged #164 / ``tests/test_bm_device_sync.py``) moves those weights without
calling ``DEQForecaster.to`` and without moving the batch. The CPU history
then mixes with parameters on another device inside ``embed``.

``forward`` rereads the live ``embed`` parameter device on every call and
copies the history onto it. The caller's storage and values stay put, and the
copy keeps the caller's dtype.

Anderson's solver calls ``Tensor.item()``, which meta tensors reject, so the
meta tests replace the equilibrium block with an identity. That stand-in has
no parameters; the failure and the fix are both in the embed / head path.
``.to(cpu)`` cannot copy out of a meta tensor, so the return trip retargets
storage with ``_apply``. The CPU test keeps the real Anderson solver.

Out of scope: library ``QDEQ.forward`` input alignment (#1344) and the qvae
cell energy precision fix (#500). This file does not import or patch
``kaiwu.torch_plugin.qdeq``.
"""

# Parent moves go through ``_apply`` (see merged #164).
# pylint: disable=protected-access

import importlib.machinery
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLE = ROOT / "example" / "qdeq" / "simple" / "etth1_common.py"

_CPU = torch.device("cpu")
_META = torch.device("meta")
_FORECAST_SHAPE = (2, 2, 3)


def _stub_module(name, **attributes):
    """Register a placeholder module when an optional import is missing.

    Meta ``mean`` / ``std`` import ``torch._dynamo``, which calls
    ``find_spec`` on already-imported modules. A stub whose ``__spec__`` is
    ``None`` raises ``ValueError`` there, so the placeholder needs a spec
    even when pandas itself is never used.
    """
    try:
        return __import__(name)
    except ImportError:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        for key, value in attributes.items():
            setattr(module, key, value)
        sys.modules[name] = module
        parent_name, _, child_name = name.rpartition(".")
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is not None and not hasattr(parent, child_name):
                setattr(parent, child_name, module)
        return module


def _load_etth1():
    """Load the ETTh1 example without the Kaiwu SDK or pandas."""
    sys.path.insert(0, str(SRC))
    _stub_module("pandas")
    _stub_module("kaiwu.cim", CIMOptimizer=object, PrecisionReducer=object)
    _stub_module("kaiwu.classical", SimulatedAnnealingOptimizer=object)
    spec = importlib.util.spec_from_file_location("etth1_common_device", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["etth1_common_device"] = module
    spec.loader.exec_module(module)
    return module


_ETTH1 = _load_etth1()
DEQForecaster = _ETTH1.DEQForecaster
QDEQConfig = _ETTH1.QDEQConfig
AndersonDEQ = _ETTH1.AndersonDEQ


class _IdentityEquilibrium(nn.Module):  # pylint: disable=too-few-public-methods
    """Equilibrium stand-in so meta tests do not call ``Tensor.item``."""

    def forward(self, tokens):
        """Return ``tokens`` unchanged."""
        return tokens


class _IdentityForecaster(DEQForecaster):  # pylint: disable=too-few-public-methods
    """Forecaster whose equilibrium block does not call ``Tensor.item``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deq = _IdentityEquilibrium()


class _RecordingForecaster(DEQForecaster):  # pylint: disable=too-few-public-methods
    """Forecaster that records direct ``.to()`` calls.

    ``identity=True`` replaces Anderson so meta tensors never see ``Tensor.item``.
    """

    def __init__(self, *args, identity=False, **kwargs):
        super().__init__(*args, **kwargs)
        if identity:
            self.deq = _IdentityEquilibrium()
        self.direct_to_calls = []

    def to(self, *args, **kwargs):
        """Record the call, then perform the real move."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


class _ParameterFreeEmbed(nn.Module):  # pylint: disable=too-few-public-methods
    """Repeat the series so the token width is ``d_model`` without parameters."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, series):
        """Return ``(batch, variables, d_model)`` on ``series.device``."""
        width = series.shape[-1]
        repeats = (self.d_model + width - 1) // width
        return series.repeat(*([1] * (series.dim() - 1)), repeats)[..., : self.d_model]


class _ParameterFreeHead(nn.Module):  # pylint: disable=too-few-public-methods
    """Project tokens with a slice so the head adds no parameters."""

    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, tokens):
        """Return the first ``pred_len`` features on ``tokens.device``."""
        return tokens[..., : self.pred_len]


def _config():
    """Return a tiny solver config for CPU Anderson and meta tests."""
    return QDEQConfig(
        f_thres=3,
        b_thres=3,
        num_basis=2,
        coeff_bits=2,
        coeff_range=1.0,
        fd_eps=1e-2,
        f_eps=1e-2,
    )


def _build(cls, **extra):
    """Construct one small forecaster of type ``cls``."""
    return cls(
        seq_len=4,
        pred_len=2,
        sampler=None,
        config=_config(),
        d_model=8,
        n_heads=2,
        d_ff=8,
        **extra,
    )


def _forecaster(identity=False, dtype=torch.float32):
    """Return a small forecaster, optionally with an identity equilibrium."""
    model = _build(_IdentityForecaster if identity else DEQForecaster)
    if dtype != torch.float32:
        model.to(dtype=dtype)
    return model


def _recording_forecaster(identity=True):
    """Return a forecaster that records ``.to()``.

    The default skips Anderson. Pass ``identity=False`` for the real solver.
    """
    return _build(_RecordingForecaster, identity=identity)


def _history(dtype=torch.float32):
    """Return a noncontiguous ``(2, 4, 3)`` window with distinct values."""
    base = torch.arange(48, dtype=dtype).reshape(2, 4, 6)
    return base[:, :, ::2]


def _move_parameters(module, device):
    """Move parameters the way a parent does: ``_apply``, not ``.to``."""
    module._apply(lambda tensor: tensor.to(device))


def _retarget_storage(device):
    """Return an ``_apply`` callback that places empty storage on ``device``.

    ``Module.to`` cannot copy out of ``meta``. Replacing storage still goes
    through ``_apply``, so the parameter device changes without ``.to()``.
    """

    def _apply_fn(tensor):
        if tensor.device == device:
            return tensor
        return torch.empty(tensor.shape, dtype=tensor.dtype, device=device)

    return _apply_fn


def _assert_unchanged(case, tensor, before, pointer):
    """Assert ``tensor`` still has the same storage, device, dtype, and values."""
    case.assertEqual(tensor.data_ptr(), pointer)
    case.assertEqual(tensor.device, before.device)
    case.assertEqual(tensor.dtype, before.dtype)
    case.assertFalse(tensor.is_contiguous())
    case.assertTrue(torch.equal(tensor, before))


def _assert_cannot_copy_out_of_meta(case):
    """``.to(cpu)`` cannot materialise a meta tensor or a meta module."""
    meta_value = torch.empty(1, device=_META)
    with case.assertRaises(NotImplementedError) as caught:
        meta_value.to(_CPU)
    case.assertIn("meta", str(caught.exception).lower())

    module = nn.Linear(2, 2)
    _move_parameters(module, _META)
    with case.assertRaises(NotImplementedError) as caught:
        module.to(_CPU)
    case.assertIn("meta", str(caught.exception).lower())
    case.assertEqual(next(module.parameters()).device, _META)


def _reference_identity(model, history):
    """Replay embed and head on CPU, skipping the device move inside ``forward``.

    ``meta`` weights have no data, so callers must use a real device. The
    equilibrium block is not replayed; this matches the identity stand-in.
    """
    dtype = history.dtype
    embed = nn.Linear(model.embed.in_features, model.embed.out_features)
    head = nn.Linear(model.head.in_features, model.head.out_features)
    embed.to(dtype=dtype)
    head.to(dtype=dtype)
    embed.load_state_dict(
        {
            key: value.detach().to(device=_CPU, dtype=dtype)
            for key, value in model.embed.state_dict().items()
        }
    )
    head.load_state_dict(
        {
            key: value.detach().to(device=_CPU, dtype=dtype)
            for key, value in model.head.state_dict().items()
        }
    )
    series = history.detach().to(device=_CPU, dtype=dtype)
    mean = series.mean(dim=1, keepdim=True)
    std = series.std(dim=1, keepdim=True) + 1e-5
    tokens = embed(((series - mean) / std).transpose(1, 2))
    return head(tokens).transpose(1, 2) * std + mean


class TestDEQForecasterParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine, including CPU-only CI."""

    def test_cpu_anderson_forward_leaves_history_unchanged(self):
        """An unmoved CPU model still forecasts with the real Anderson solver."""
        torch.manual_seed(0)
        model = _forecaster()
        history = _history()
        before = history.clone()
        pointer = history.data_ptr()

        forecast = model(history)

        self.assertIsInstance(model.deq, AndersonDEQ)
        _assert_unchanged(self, history, before, pointer)
        self.assertEqual(forecast.shape, _FORECAST_SHAPE)
        self.assertEqual(forecast.device, _CPU)
        self.assertEqual(forecast.dtype, torch.float32)
        self.assertTrue(torch.isfinite(forecast).all().item())

    def test_parent_to_meta_skips_forecaster_to(self):
        """``nn.Module.to`` must move the forecaster without calling its ``.to``."""
        model = _recording_forecaster()
        parent = nn.Module()
        parent.model = model

        parent.to(_META)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(next(model.embed.parameters()).device, _META)
        self.assertEqual(next(model.head.parameters()).device, _META)

    def test_parent_apply_accepts_cpu_history(self):
        """CPU history must forecast on meta, and its storage must stay put."""
        model = _recording_forecaster()
        parent = nn.Module()
        parent.model = model
        _move_parameters(parent, _META)
        history = _history()
        before = history.clone()
        pointer = history.data_ptr()

        forecast = model(history)

        self.assertEqual(model.direct_to_calls, [])
        _assert_unchanged(self, history, before, pointer)
        self.assertEqual(_ETTH1._parameter_device(model.embed, history.device), _META)
        self.assertEqual(next(model.embed.parameters()).device, _META)
        self.assertEqual(next(model.head.parameters()).device, _META)
        self.assertEqual(forecast.device, _META)
        self.assertEqual(forecast.shape, _FORECAST_SHAPE)
        self.assertEqual(forecast.dtype, torch.float32)

    def test_float64_parameters_keep_history_dtype_and_storage(self):
        """Moving float64 weights must not recast or rewrite the caller history."""
        model = _forecaster(identity=True, dtype=torch.float64)
        parent = nn.Module()
        parent.model = model
        _move_parameters(parent, _META)
        history = _history(dtype=torch.float64)
        before = history.clone()
        pointer = history.data_ptr()

        forecast = model(history)

        _assert_unchanged(self, history, before, pointer)
        self.assertEqual(next(model.embed.parameters()).dtype, torch.float64)
        self.assertEqual(next(model.embed.parameters()).device, _META)
        self.assertEqual(forecast.dtype, torch.float64)
        self.assertEqual(forecast.device, _META)
        self.assertEqual(forecast.shape, _FORECAST_SHAPE)

    def test_dtype_only_move_keeps_cpu_device(self):
        """``.to(dtype=...)`` must keep CPU and must not call the child ``.to``."""
        model = _recording_forecaster()
        parent = nn.Module()
        parent.model = model
        parent.to(dtype=torch.float64)
        history = _history(dtype=torch.float64)
        before = history.clone()
        pointer = history.data_ptr()

        forecast = model(history)

        self.assertEqual(model.direct_to_calls, [])
        _assert_unchanged(self, history, before, pointer)
        weight = next(model.embed.parameters())
        self.assertEqual(weight.dtype, torch.float64)
        self.assertEqual(weight.device, _CPU)
        self.assertEqual(next(model.head.parameters()).dtype, torch.float64)
        self.assertEqual(forecast.dtype, torch.float64)
        self.assertEqual(forecast.device, _CPU)
        self.assertEqual(forecast.shape, _FORECAST_SHAPE)
        torch.testing.assert_close(forecast, _reference_identity(model, history))

    def test_dtype_only_anderson_forward_stays_finite(self):
        """A float64 Anderson model must stay on CPU and leave the history put."""
        torch.manual_seed(0)
        model = _forecaster()
        model.to(dtype=torch.float64)
        history = _history(dtype=torch.float64)
        before = history.clone()
        pointer = history.data_ptr()

        forecast = model(history)

        self.assertIsInstance(model.deq, AndersonDEQ)
        _assert_unchanged(self, history, before, pointer)
        self.assertEqual(next(model.parameters()).dtype, torch.float64)
        self.assertEqual(next(model.parameters()).device, _CPU)
        self.assertEqual(forecast.dtype, torch.float64)
        self.assertEqual(forecast.device, _CPU)
        self.assertEqual(forecast.shape, _FORECAST_SHAPE)
        self.assertTrue(torch.isfinite(forecast).all().item())

    def test_parameter_device_is_reread_after_meta_round_trip(self):
        """The helper must be reread after each move, not cached.

        ``.to(cpu)`` cannot copy out of meta. ``_apply`` still retargets
        storage, and the next forecast follows that new device.
        """
        _assert_cannot_copy_out_of_meta(self)
        model = _recording_forecaster()
        parent = nn.Module()
        parent.model = model
        history = _history()
        before = history.clone()
        pointer = history.data_ptr()

        forecast = model(history)
        self.assertEqual(_ETTH1._parameter_device(model.embed, history.device), _CPU)
        self.assertEqual(forecast.device, _CPU)

        _move_parameters(parent, _META)
        self.assertEqual(_ETTH1._parameter_device(model.embed, history.device), _META)
        forecast = model(history)
        self.assertEqual(forecast.device, _META)
        self.assertEqual(forecast.dtype, torch.float32)

        parent._apply(_retarget_storage(_CPU))
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(_ETTH1._parameter_device(model.embed, history.device), _CPU)
        forecast = model(history)
        self.assertEqual(next(model.embed.parameters()).device, _CPU)
        self.assertEqual(next(model.head.parameters()).device, _CPU)
        self.assertEqual(forecast.device, _CPU)
        self.assertEqual(forecast.shape, _FORECAST_SHAPE)

        _move_parameters(parent, _META)
        self.assertEqual(_ETTH1._parameter_device(model.embed, history.device), _META)
        forecast = model(history)
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(forecast.device, _META)
        _assert_unchanged(self, history, before, pointer)

    def test_parameter_device_uses_live_parameters_or_fallback(self):
        """Empty modules fall back; a later embed move wins over that fallback."""
        fallback = _CPU
        self.assertEqual(_ETTH1._parameter_device(nn.Identity(), fallback), fallback)

        model = _forecaster(identity=True)
        self.assertEqual(_ETTH1._parameter_device(model.embed, fallback), _CPU)
        _move_parameters(model.embed, _META)
        self.assertEqual(_ETTH1._parameter_device(model.embed, _CPU), _META)
        self.assertEqual(next(model.head.parameters()).device, _CPU)

    def test_parameter_free_embed_follows_the_history_device(self):
        """With no embed parameters, the forecast follows the history device."""
        model = _forecaster(identity=True)
        model.embed = _ParameterFreeEmbed(d_model=8)
        model.head = _ParameterFreeHead(pred_len=2)
        history = _history()
        before = history.clone()
        pointer = history.data_ptr()

        forecast = model(history)

        _assert_unchanged(self, history, before, pointer)
        self.assertEqual(forecast.device, _CPU)
        self.assertEqual(forecast.dtype, torch.float32)
        self.assertEqual(forecast.shape, _FORECAST_SHAPE)
        self.assertTrue(torch.isfinite(forecast).all().item())

        meta_history = torch.ones(2, 4, 3, device=_META)
        meta_pointer = meta_history.data_ptr()
        forecast = model(meta_history)
        self.assertEqual(meta_history.data_ptr(), meta_pointer)
        self.assertEqual(meta_history.device, _META)
        self.assertEqual(forecast.device, _META)
        self.assertEqual(forecast.dtype, torch.float32)
        self.assertEqual(forecast.shape, _FORECAST_SHAPE)

    def test_same_device_identity_matches_weight_replay(self):
        """On one device, the ``.to`` copy must not change the forecast."""
        torch.manual_seed(0)
        model = _forecaster(identity=True)
        history = _history()
        before = history.clone()
        pointer = history.data_ptr()

        forecast = model(history)

        _assert_unchanged(self, history, before, pointer)
        self.assertEqual(forecast.device, _CPU)
        torch.testing.assert_close(forecast, _reference_identity(model, history))

    def test_same_device_alignment_keeps_autograd(self):
        """A same-device copy must still backpropagate into the history and weights."""
        torch.manual_seed(0)
        model = _forecaster(identity=True)
        history = _history().contiguous().requires_grad_(True)

        forecast = model(history)
        forecast.sum().backward()

        self.assertEqual(forecast.device, _CPU)
        self.assertIsNotNone(history.grad)
        self.assertEqual(history.grad.device, _CPU)
        self.assertEqual(history.grad.shape, history.shape)
        self.assertTrue(torch.isfinite(history.grad).all().item())
        embed_weight = model.embed.weight
        head_weight = model.head.weight
        self.assertIsNotNone(embed_weight.grad)
        self.assertEqual(embed_weight.grad.device, _CPU)
        self.assertTrue(torch.isfinite(embed_weight.grad).all().item())
        self.assertGreater(float(embed_weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(head_weight.grad)
        self.assertEqual(head_weight.grad.device, _CPU)
        self.assertGreater(float(head_weight.grad.abs().sum()), 0.0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDEQForecasterCudaParameterDevice(unittest.TestCase):
    """End-to-end coverage of a parent move onto CUDA. Skipped without a GPU."""

    def test_parent_apply_accepts_cpu_history_with_anderson(self):
        """CPU windows must forecast on CUDA after ``_apply``, without ``.to``."""
        torch.manual_seed(0)
        model = _recording_forecaster(identity=False)
        parent = nn.Module()
        parent.model = model
        cuda = torch.device("cuda")
        _move_parameters(parent, cuda)
        history = _history()
        before = history.clone()
        pointer = history.data_ptr()

        forecast = model(history)

        self.assertIsInstance(model.deq, AndersonDEQ)
        self.assertEqual(model.direct_to_calls, [])
        _assert_unchanged(self, history, before, pointer)
        self.assertEqual(next(model.embed.parameters()).device.type, "cuda")
        self.assertEqual(forecast.device.type, "cuda")
        self.assertEqual(forecast.dtype, torch.float32)
        self.assertEqual(forecast.shape, _FORECAST_SHAPE)
        self.assertTrue(torch.isfinite(forecast).all().item())

    def test_identity_forecast_matches_cpu_weights(self):
        """A CUDA ``_apply`` must match the same embed and head on CPU."""
        torch.manual_seed(0)
        model = _recording_forecaster()
        parent = nn.Module()
        parent.model = model
        _move_parameters(parent, torch.device("cuda"))
        history = _history()
        before = history.clone()
        pointer = history.data_ptr()

        forecast = model(history)

        self.assertEqual(model.direct_to_calls, [])
        _assert_unchanged(self, history, before, pointer)
        self.assertEqual(forecast.device.type, "cuda")
        torch.testing.assert_close(
            forecast.detach().cpu(),
            _reference_identity(model, history),
        )

    def test_device_is_reread_across_cuda_and_cpu(self):
        """CUDA and CPU parent moves must be reread, and must not call ``.to``."""
        torch.manual_seed(0)
        model = _recording_forecaster()
        parent = nn.Module()
        parent.model = model
        history = _history()
        before = history.clone()
        pointer = history.data_ptr()
        cuda = torch.device("cuda")

        _move_parameters(parent, cuda)
        forecast = model(history)
        self.assertEqual(
            _ETTH1._parameter_device(model.embed, history.device).type, "cuda"
        )
        self.assertEqual(forecast.device.type, "cuda")
        torch.testing.assert_close(
            forecast.detach().cpu(),
            _reference_identity(model, history),
        )

        _move_parameters(parent, _CPU)
        forecast = model(history)
        self.assertEqual(_ETTH1._parameter_device(model.embed, history.device), _CPU)
        self.assertEqual(forecast.device, _CPU)
        torch.testing.assert_close(forecast, _reference_identity(model, history))

        _move_parameters(parent, cuda)
        forecast = model(history)
        self.assertEqual(model.direct_to_calls, [])
        _assert_unchanged(self, history, before, pointer)
        self.assertEqual(forecast.device.type, "cuda")
        self.assertEqual(forecast.shape, _FORECAST_SHAPE)

    def test_float64_cuda_move_keeps_dtype(self):
        """A float64 CUDA move must not downcast the history or the forecast."""
        torch.manual_seed(0)
        model = _forecaster(identity=True, dtype=torch.float64)
        parent = nn.Module()
        parent.model = model
        _move_parameters(parent, torch.device("cuda"))
        history = _history(dtype=torch.float64)
        before = history.clone()
        pointer = history.data_ptr()

        forecast = model(history)

        _assert_unchanged(self, history, before, pointer)
        self.assertEqual(next(model.embed.parameters()).dtype, torch.float64)
        self.assertEqual(forecast.dtype, torch.float64)
        self.assertEqual(forecast.device.type, "cuda")
        torch.testing.assert_close(
            forecast.detach().cpu(),
            _reference_identity(model, history),
        )

    def test_dtype_only_move_keeps_cuda_device(self):
        """``.to(dtype=...)`` on CUDA must keep CUDA and must not call child ``.to``."""
        model = _recording_forecaster()
        parent = nn.Module()
        parent.model = model
        _move_parameters(parent, torch.device("cuda"))
        parent.to(dtype=torch.float64)
        history = _history(dtype=torch.float64)
        before = history.clone()
        pointer = history.data_ptr()

        forecast = model(history)

        self.assertEqual(model.direct_to_calls, [])
        _assert_unchanged(self, history, before, pointer)
        weight = next(model.embed.parameters())
        self.assertEqual(weight.dtype, torch.float64)
        self.assertEqual(weight.device.type, "cuda")
        self.assertEqual(forecast.dtype, torch.float64)
        self.assertEqual(forecast.device.type, "cuda")
        torch.testing.assert_close(
            forecast.detach().cpu(),
            _reference_identity(model, history),
        )

    def test_backward_from_cpu_history_reaches_cuda_weights(self):
        """The history copy inside ``forward`` must stay in the autograd graph."""
        torch.manual_seed(0)
        model = _forecaster(identity=True)
        parent = nn.Module()
        parent.model = model
        _move_parameters(parent, torch.device("cuda"))
        history = _history().contiguous().requires_grad_(True)
        pointer = history.data_ptr()

        forecast = model(history)
        forecast.sum().backward()

        self.assertEqual(history.data_ptr(), pointer)
        self.assertEqual(history.device, _CPU)
        self.assertEqual(forecast.device.type, "cuda")
        self.assertIsNotNone(history.grad)
        self.assertEqual(history.grad.device, _CPU)
        self.assertEqual(history.grad.shape, history.shape)
        self.assertTrue(torch.isfinite(history.grad).all().item())
        embed_weight = model.embed.weight
        self.assertIsNotNone(embed_weight.grad)
        self.assertEqual(embed_weight.grad.device.type, "cuda")
        self.assertTrue(torch.isfinite(embed_weight.grad).all().item())
        self.assertGreater(float(embed_weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.head.weight.grad)
        self.assertEqual(model.head.weight.grad.device.type, "cuda")

    def test_cuda_history_follows_cpu_parameters(self):
        """A CUDA history must still forecast on a CPU forecaster."""
        torch.manual_seed(0)
        model = _forecaster(identity=True)
        history = _history().to(device="cuda")
        before = history.detach().cpu().clone()
        pointer = history.data_ptr()

        forecast = model(history)

        self.assertEqual(history.data_ptr(), pointer)
        self.assertEqual(history.device.type, "cuda")
        self.assertEqual(history.dtype, torch.float32)
        torch.testing.assert_close(history.detach().cpu(), before)
        self.assertEqual(next(model.embed.parameters()).device, _CPU)
        self.assertEqual(forecast.device, _CPU)
        self.assertEqual(forecast.shape, _FORECAST_SHAPE)
        torch.testing.assert_close(forecast, _reference_identity(model, history))


if __name__ == "__main__":
    unittest.main()
