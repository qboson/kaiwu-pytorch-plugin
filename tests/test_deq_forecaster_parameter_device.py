"""Regression tests for ``DEQForecaster`` parameter-device alignment.

``DEQForecaster`` embeds the caller history with ``nn.Linear`` weights. A
parent ``nn.Module._apply`` (the path ``nn.Module.to`` uses for children;
merged #164) moves those weights without calling ``DEQForecaster.to`` and
without moving the batch. The CPU history then mixes with parameters on
another device inside ``embed``.

Anderson's solver calls ``Tensor.item()``, which meta tensors reject, so the
meta tests replace the equilibrium block with an identity. That stand-in has
no parameters; the failure and the fix are both in the embed / head path.
The CPU test keeps the real Anderson solver.

Out of scope: library ``QDEQ.forward`` input alignment (#1344) and the qvae
cell energy precision fix (#500).
"""

# Parent moves go through ``_apply`` (see merged #164).
# pylint: disable=protected-access

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


def _stub_module(name, **attributes):
    """Register a placeholder module when an optional import is missing."""
    try:
        return __import__(name)
    except ImportError:
        module = types.ModuleType(name)
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


class _RecordingForecaster(  # pylint: disable=too-few-public-methods
    _IdentityForecaster
):
    """Forecaster that records direct ``.to()`` calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct_to_calls = []

    def to(self, *args, **kwargs):
        """Record the call, then perform the real move."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


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


def _build(cls):
    """Construct one small forecaster of type ``cls``."""
    return cls(
        seq_len=4,
        pred_len=2,
        sampler=None,
        config=_config(),
        d_model=8,
        n_heads=2,
        d_ff=8,
    )


def _forecaster(identity=False, dtype=torch.float32):
    """Return a small forecaster, optionally with an identity equilibrium."""
    model = _build(_IdentityForecaster if identity else DEQForecaster)
    if dtype != torch.float32:
        model.to(dtype=dtype)
    return model


def _recording_forecaster():
    """Return a forecaster that records ``.to()`` and skips Anderson."""
    return _build(_RecordingForecaster)


def _move_parameters_to_meta(module):
    """Move parameters the way a parent does: ``_apply``, not ``.to``."""
    module._apply(lambda tensor: tensor.to(torch.device("meta")))


class TestDEQForecasterParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine, including CPU-only CI."""

    def test_cpu_anderson_forward_stays_finite(self):
        """An unmoved CPU model still forecasts with the real Anderson solver."""
        model = _forecaster()
        history = torch.ones(2, 4, 3)

        forecast = model(history)

        self.assertEqual(forecast.shape, (2, 2, 3))
        self.assertEqual(forecast.device, torch.device("cpu"))
        self.assertEqual(forecast.dtype, torch.float32)
        self.assertTrue(torch.isfinite(forecast).all().item())

    def test_parent_apply_accepts_cpu_history(self):
        """CPU history must forecast once embed and head live on meta."""
        model = _forecaster(identity=True)
        parent = nn.Module()
        parent.model = model
        _move_parameters_to_meta(parent)
        history = torch.ones(2, 4, 3)

        forecast = model(history)

        self.assertEqual(history.device, torch.device("cpu"))
        self.assertEqual(forecast.device, torch.device("meta"))
        self.assertEqual(forecast.shape, (2, 2, 3))
        self.assertEqual(forecast.dtype, torch.float32)
        self.assertEqual(next(model.embed.parameters()).device, torch.device("meta"))

    def test_parent_apply_skips_forecaster_to(self):
        """Parent ``_apply`` must not be required to call ``DEQForecaster.to``."""
        model = _recording_forecaster()
        parent = nn.Module()
        parent.model = model
        _move_parameters_to_meta(parent)

        forecast = model(torch.ones(2, 4, 3))

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(forecast.device, torch.device("meta"))

    def test_double_parameters_keep_history_dtype(self):
        """Moving float64 weights must not recast the caller history."""
        model = _forecaster(identity=True, dtype=torch.float64)
        _move_parameters_to_meta(model)
        history = torch.ones(2, 4, 3, dtype=torch.float64)

        forecast = model(history)

        self.assertEqual(history.dtype, torch.float64)
        self.assertEqual(forecast.dtype, torch.float64)
        self.assertEqual(forecast.device, torch.device("meta"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDEQForecasterCudaParameterDevice(unittest.TestCase):
    """End-to-end coverage of a parent move onto CUDA with the real solver."""

    def test_parent_move_accepts_cpu_history(self):
        """CPU windows must forecast on CUDA after a parent ``_apply``."""
        model = _forecaster()
        parent = nn.Module()
        parent.model = model
        parent._apply(lambda tensor: tensor.to(torch.device("cuda")))
        history = torch.ones(2, 4, 3)

        forecast = model(history)

        self.assertEqual(history.device, torch.device("cpu"))
        self.assertEqual(forecast.device.type, "cuda")
        self.assertEqual(forecast.shape, (2, 2, 3))
        self.assertTrue(torch.isfinite(forecast).all().item())


if __name__ == "__main__":
    unittest.main()
