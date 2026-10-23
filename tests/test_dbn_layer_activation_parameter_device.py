"""Regression tests for supervised DBN layer-activation device placement.

``AbstractSupervisedDBNClassifier.get_layer_activations`` used to build each
visible batch with ``torch.FloatTensor(...).to(unsupervised_dbn.device)``.
That attribute is captured when the DBN is constructed. A parent
``nn.Module._apply`` moves the stacked RBMs (merged #164 refreshes each RBM's
own ``device``) and does not call the child's ``.to()``, so the DBN cache
stays behind. ``get_hidden`` then multiplies a CPU batch by weights that
already live on ``meta``:

``RuntimeError: Tensor on device meta is not on the expected device cpu!``

The batch must follow the live RBM parameter device and dtype. The
construction cache must not choose the device. CPU cases use ``meta``.
CUDA cases are skipped unless a GPU is present.
"""

# Parent moves go through ``_apply`` (see merged #164).
# pylint: disable=protected-access

from __future__ import annotations

import importlib
import os
import sys
import types
import unittest

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MPLBACKEND", "Agg")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "example", "dbn_digits"))


def _ensure_sampler_imports() -> None:
    """Allow the example to import when the Kaiwu SDK is not installed."""
    try:
        importlib.import_module("kaiwu.classical")
        importlib.import_module("kaiwu.cim")
    except ImportError:
        kaiwu = importlib.import_module("kaiwu")
        classical = types.ModuleType("kaiwu.classical")
        cim = types.ModuleType("kaiwu.cim")

        def _init(self, *args, **kwargs) -> None:
            del self, args, kwargs

        optimizer = type("Optimizer", (), {"__init__": _init})
        classical.SimulatedAnnealingOptimizer = optimizer
        cim.CIMOptimizer = optimizer
        cim.PrecisionReducer = optimizer
        sys.modules["kaiwu.classical"] = classical
        sys.modules["kaiwu.cim"] = cim
        kaiwu.classical = classical
        kaiwu.cim = cim


_ensure_sampler_imports()

from supervised_dbn_digits import (  # pylint: disable=wrong-import-position
    SupervisedDBNClassification,
)

_CPU = torch.device("cpu")
_META = torch.device("meta")


def _classifier():
    """One untrained RBM stack whose weights are still on the CPU."""
    classifier = SupervisedDBNClassification(
        hidden_layers_structure=[3],
        n_epochs_rbm=1,
        batch_size=2,
        verbose=False,
        fine_tuning=False,
        random_state=0,
    )
    dbn = classifier.unsupervised_dbn._dbn
    dbn.create_rbm_layer(4)
    dbn.mark_as_trained()
    return classifier


def _features():
    """Small visible batch. ``astype`` inside the method copies this buffer."""
    return np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float32)


def _record_to(module):
    """Record direct ``.to()`` calls. Parent ``_apply`` must not make any."""
    calls = []
    original = module.to

    def wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    module.to = wrapped
    return calls


def _spy_hidden(rbm):
    """Record the visible batch ``get_hidden`` actually receives."""
    seen = []
    real = rbm.get_hidden

    def spy(visible, *args, **kwargs):
        seen.append(visible)
        return real(visible, *args, **kwargs)

    rbm.get_hidden = spy
    return seen


def _retarget(device):
    """``_apply`` callback that places new storage without reading old data."""

    def _apply_fn(tensor):
        return torch.zeros(tensor.shape, dtype=tensor.dtype, device=device)

    return _apply_fn


def _expected(rbm, features, dtype):
    """Hidden slice from a weight copy, computed on CPU."""
    visible = torch.tensor(np.asarray(features), dtype=dtype, device=_CPU)
    with torch.no_grad():
        hidden = rbm.get_hidden(visible)
    return hidden[:, rbm.num_visible :].detach().cpu().numpy()


class TestLayerActivationParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_unmoved_cpu_matches_hidden_slice(self):
        """An unmoved stack matches ``get_hidden`` and does not use the cache."""
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        features = _features()
        dbn.device = _META

        activations = classifier.get_layer_activations(features)

        np.testing.assert_allclose(activations, _expected(rbm, features, torch.float32))
        self.assertEqual(activations.shape, (2, 3))

    def test_parent_apply_meta_reads_parameter_device(self):
        """``_apply(meta)`` must not call ``.to`` or keep the CPU cache."""
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        parent = nn.Module()
        parent.dbn = dbn
        calls = _record_to(rbm)
        seen = _spy_hidden(rbm)

        parent._apply(lambda tensor: tensor.to(_META))

        self.assertEqual(calls, [])
        self.assertEqual(dbn.device, _CPU)
        self.assertEqual(rbm.quadratic_coef.device, _META)
        with self.assertRaises(NotImplementedError):
            classifier.get_layer_activations(_features())
        self.assertEqual(seen[0].device, _META)
        self.assertNotEqual(seen[0].device, dbn.device)

    def test_parent_to_meta_skips_child_to(self):
        """``Module.to`` moves children through ``_apply``, not ``.to()``."""
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        parent = nn.Module()
        parent.dbn = dbn
        calls = _record_to(rbm)

        parent.to(_META)

        self.assertEqual(calls, [])
        self.assertEqual(next(rbm.parameters()).device, _META)
        self.assertEqual(dbn.device, _CPU)

    def test_dtype_only_move_stays_on_cpu(self):
        """``.to(dtype=...)`` keeps the parameter device and the batch follows."""
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        parent = nn.Module()
        parent.dbn = dbn
        calls = _record_to(rbm)
        seen = _spy_hidden(rbm)
        features = _features()

        parent.to(dtype=torch.float64)

        self.assertEqual(calls, [])
        self.assertEqual(rbm.quadratic_coef.device, _CPU)
        self.assertEqual(rbm.quadratic_coef.dtype, torch.float64)
        self.assertEqual(dbn.device, _CPU)
        activations = classifier.get_layer_activations(features)
        self.assertEqual(seen[0].device, _CPU)
        self.assertEqual(seen[0].dtype, torch.float64)
        np.testing.assert_allclose(
            activations, _expected(rbm, features, torch.float64), atol=1e-6
        )

    def test_parameter_device_is_reread_after_meta_roundtrip(self):
        """The device is read again after storage is retargeted off ``meta``."""
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        parent = nn.Module()
        parent.dbn = dbn
        calls = _record_to(rbm)
        seen = _spy_hidden(rbm)
        features = _features()

        parent._apply(lambda tensor: tensor.to(_META))
        with self.assertRaises(NotImplementedError):
            classifier.get_layer_activations(features)
        self.assertEqual(seen[-1].device, _META)

        parent._apply(_retarget(_CPU))
        activations = classifier.get_layer_activations(features)
        self.assertEqual(calls, [])
        self.assertEqual(seen[-1].device, _CPU)
        self.assertEqual(dbn.device, _CPU)
        self.assertTrue(np.isfinite(activations).all())

        parent._apply(lambda tensor: tensor.to(_META))
        with self.assertRaises(NotImplementedError):
            classifier.get_layer_activations(features)
        self.assertEqual(seen[-1].device, _META)
        self.assertEqual(calls, [])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestLayerActivationCuda(unittest.TestCase):
    """Construct-on-CPU / move-to-CUDA coverage."""

    def test_parent_apply_cuda_matches_cpu_weight_copy(self):
        """A CPU batch follows CUDA parameters and matches a CPU weight copy."""
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        weight = rbm.quadratic_coef.detach().cpu().clone()
        bias = rbm.linear_bias.detach().cpu().clone()
        parent = nn.Module()
        parent.dbn = dbn
        calls = _record_to(rbm)
        seen = _spy_hidden(rbm)
        features = _features()

        parent._apply(lambda tensor: tensor.to(torch.device("cuda")))

        self.assertEqual(calls, [])
        self.assertEqual(dbn.device, _CPU)
        dbn.device = _CPU
        activations = classifier.get_layer_activations(features)
        self.assertEqual(seen[0].device.type, "cuda")
        visible = torch.tensor(features, dtype=weight.dtype)
        hidden = torch.sigmoid(visible @ weight + bias[rbm.num_visible :])
        np.testing.assert_allclose(activations, hidden.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
