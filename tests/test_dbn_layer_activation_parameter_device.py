"""Regression tests for supervised DBN layer-activation device placement.

``AbstractSupervisedDBNClassifier.get_layer_activations`` used to build each
visible batch with ``torch.FloatTensor(...).to(unsupervised_dbn.device)``.
That attribute is captured when the DBN is constructed. A parent
``nn.Module._apply`` moves the stacked RBMs (merged #164 refreshes each RBM's
own ``device``) and does not call the child's ``.to()``, so the DBN cache
stays behind. ``get_hidden`` then allocates its state on the live parameter
device and assigns a CPU batch into it:

``RuntimeError: Tensor on device meta is not on the expected device cpu!``

The batch must follow the live RBM parameter device and dtype. The
construction cache must not choose the device. CPU cases use ``meta``.
CUDA cases are skipped unless a GPU is present.

A tensor that already matches keeps its Python identity and ``data_ptr``,
including a noncontiguous view. A dtype-only parent move keeps the
parameter device. Leaving ``meta`` cannot copy values out, so the return
trip retargets storage through ``_apply`` and ``zeros_like``. NumPy input is
copied with ``torch.tensor`` and never aliases the caller buffer. The
multi-layer path rereads each RBM independently, and ``return_all_layers``
is covered.
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
_CUDA = torch.device("cuda")


def _classifier(structure=(3,), input_dim=4):
    """One untrained RBM stack whose weights are still on the CPU."""
    classifier = SupervisedDBNClassification(
        hidden_layers_structure=list(structure),
        n_epochs_rbm=1,
        batch_size=2,
        verbose=False,
        fine_tuning=False,
        random_state=0,
    )
    dbn = classifier.unsupervised_dbn._dbn
    dbn.create_rbm_layer(input_dim)
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


def _apply_device(module, device):
    """Move parameters through ``_apply``, the parent-module path."""
    module._apply(lambda tensor: tensor.to(device))


def _retarget(device):
    """``_apply`` callback that places new storage without reading old data."""

    def _apply_fn(tensor):
        return torch.zeros(tensor.shape, dtype=tensor.dtype, device=device)

    return _apply_fn


def _expected(rbm, features, dtype):
    """Hidden slice from the live weights, computed on CPU."""
    visible = torch.tensor(np.asarray(features), dtype=dtype, device=_CPU)
    with torch.no_grad():
        hidden = rbm.get_hidden(visible)
    return hidden[:, rbm.num_visible :].detach().cpu().numpy()


def _expected_from_weights(weight, bias, num_visible, features, dtype):
    """Hidden slice from a CPU weight copy, independent of the live module."""
    visible = torch.tensor(np.asarray(features), dtype=dtype, device=_CPU)
    with torch.no_grad():
        hidden = torch.sigmoid(visible @ weight + bias[num_visible:])
    return hidden.numpy()


def _numpy_snapshot(batch):
    """Copy values and record the caller buffer address."""
    return {
        "values": batch.copy(),
        "ptr": batch.__array_interface__["data"][0],
    }


def _assert_numpy_unchanged(case, batch, snapshot):
    """The activation call must not rewrite the caller's NumPy buffer."""
    np.testing.assert_array_equal(batch, snapshot["values"])
    case.assertEqual(batch.__array_interface__["data"][0], snapshot["ptr"])


class TestLayerActivationParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_unmoved_cpu_matches_hidden_slice(self):
        """An unmoved stack matches ``get_hidden`` and does not use the cache."""
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        features = _features()
        snapshot = _numpy_snapshot(features)
        dbn.device = _META

        activations = classifier.get_layer_activations(features)

        np.testing.assert_allclose(activations, _expected(rbm, features, torch.float32))
        self.assertEqual(activations.shape, (2, 3))
        self.assertTrue(np.isfinite(activations).all())
        self.assertEqual(dbn.device, _META)
        self.assertEqual(next(rbm.parameters()).device, _CPU)
        _assert_numpy_unchanged(self, features, snapshot)

    def test_poisoned_construction_device_loses_to_parameters(self):
        """A ``meta`` or ``cuda`` cache cannot redirect CPU weights."""
        for poisoned in (_META, _CUDA):
            with self.subTest(poisoned=str(poisoned)):
                classifier = _classifier()
                dbn = classifier.unsupervised_dbn._dbn
                rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
                features = _features()
                seen = _spy_hidden(rbm)
                dbn.device = poisoned

                activations = classifier.get_layer_activations(features)

                self.assertEqual(dbn.device, poisoned)
                self.assertEqual(next(rbm.parameters()).device, _CPU)
                self.assertEqual(seen[-1].device, _CPU)
                self.assertEqual(seen[-1].dtype, torch.float32)
                np.testing.assert_allclose(
                    activations, _expected(rbm, features, torch.float32)
                )
                self.assertTrue(np.isfinite(activations).all())

    def test_parent_apply_meta_reads_parameter_device(self):
        """``_apply(meta)`` must not call ``.to`` or keep the CPU cache."""
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        parent = nn.Module()
        parent.dbn = dbn
        calls = _record_to(rbm)
        seen = _spy_hidden(rbm)
        features = _features()
        snapshot = _numpy_snapshot(features)

        _apply_device(parent, _META)

        self.assertEqual(calls, [])
        self.assertEqual(dbn.device, _CPU)
        self.assertEqual(rbm.device, _META)
        self.assertEqual(rbm.quadratic_coef.device, _META)
        with self.assertRaises(NotImplementedError) as caught:
            classifier.get_layer_activations(features)
        self.assertIn("meta", str(caught.exception).lower())
        self.assertEqual(seen[0].device, _META)
        self.assertNotEqual(seen[0].device, dbn.device)
        self.assertEqual(calls, [])
        _assert_numpy_unchanged(self, features, snapshot)

    def test_parent_to_meta_skips_child_to(self):
        """Positional and keyword ``Module.to(meta)`` do not call child ``.to``."""
        moves = (
            ("positional", (_META,), {}),
            ("keyword", (), {"device": _META}),
            ("string", ("meta",), {}),
        )
        for label, args, kwargs in moves:
            with self.subTest(move=label):
                classifier = _classifier()
                dbn = classifier.unsupervised_dbn._dbn
                rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
                parent = nn.Module()
                parent.dbn = dbn
                calls = _record_to(rbm)

                parent.to(*args, **kwargs)

                self.assertEqual(calls, [])
                self.assertEqual(next(rbm.parameters()).device, _META)
                self.assertEqual(rbm.device, _META)
                self.assertEqual(dbn.device, _CPU)

    def test_parent_to_cpu_after_meta_cannot_copy_out(self):
        """``Module.to(cpu)`` leaves meta parameters in place and records no call."""
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        parent = nn.Module()
        parent.dbn = dbn
        calls = _record_to(rbm)

        _apply_device(parent, _META)
        with self.assertRaises(NotImplementedError) as blocked:
            parent.to(_CPU)

        self.assertIn("meta", str(blocked.exception).lower())
        self.assertEqual(next(rbm.parameters()).device, _META)
        self.assertEqual(rbm.device, _META)
        self.assertEqual(dbn.device, _CPU)
        self.assertEqual(calls, [])

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
        dbn.device = _META

        parent.to(dtype=torch.float64)

        self.assertEqual(calls, [])
        self.assertEqual(rbm.quadratic_coef.device, _CPU)
        self.assertEqual(rbm.quadratic_coef.dtype, torch.float64)
        self.assertEqual(rbm.device, _CPU)
        self.assertEqual(dbn.device, _META)
        activations = classifier.get_layer_activations(features)
        self.assertEqual(seen[0].device, _CPU)
        self.assertEqual(seen[0].dtype, torch.float64)
        np.testing.assert_allclose(
            activations, _expected(rbm, features, torch.float64), atol=1e-6
        )

    def test_parameter_device_is_reread_after_meta_roundtrip(self):
        """Meta, a rejected ``.to(cpu)``, ``zeros_like`` retarget, then meta."""
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        parent = nn.Module()
        parent.dbn = dbn
        calls = _record_to(rbm)
        seen = _spy_hidden(rbm)
        features = _features()
        snapshot = _numpy_snapshot(features)

        _apply_device(parent, _META)
        with self.assertRaises(NotImplementedError):
            classifier.get_layer_activations(features)
        self.assertEqual(seen[-1].device, _META)
        with self.assertRaises(NotImplementedError):
            parent.to(_CPU)
        self.assertEqual(next(rbm.parameters()).device, _META)

        parent._apply(_retarget(_CPU))
        weight = next(rbm.parameters())
        self.assertEqual(weight.device, _CPU)
        self.assertEqual(weight.dtype, torch.float32)
        self.assertTrue(torch.isfinite(weight).all())
        self.assertEqual(weight.abs().sum().item(), 0.0)
        dbn.device = _META
        activations = classifier.get_layer_activations(features)
        self.assertEqual(calls, [])
        self.assertEqual(seen[-1].device, _CPU)
        self.assertEqual(rbm.device, _CPU)
        self.assertEqual(dbn.device, _META)
        self.assertTrue(np.isfinite(activations).all())
        np.testing.assert_allclose(activations, _expected(rbm, features, torch.float32))

        _apply_device(parent, _META)
        with self.assertRaises(NotImplementedError):
            classifier.get_layer_activations(features)
        self.assertEqual(seen[-1].device, _META)
        self.assertEqual(calls, [])
        _assert_numpy_unchanged(self, features, snapshot)

    def test_stale_cache_placement_reproduces_device_mix(self):
        """Restoring the construction-cache placement reproduces the bug.

        Deleting the override places the same caller on meta through the new
        path and does not call child ``.to``.
        """
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        parent = nn.Module()
        parent.dbn = dbn
        calls = _record_to(rbm)
        seen = _spy_hidden(rbm)
        features = _features()
        _apply_device(parent, _META)

        def _on_cache(self, rbm, data_in):
            """The pre-fix placement: always the construction cache."""
            del rbm
            return torch.tensor(
                np.asarray(data_in),
                dtype=torch.float32,
                device=self.unsupervised_dbn.device,
            )

        classifier._batch_on_rbm_parameters = types.MethodType(_on_cache, classifier)
        with self.assertRaises(RuntimeError) as caught:
            classifier.get_layer_activations(features)
        message = str(caught.exception).lower()
        self.assertIn("meta", message)
        self.assertIn("cpu", message)
        self.assertEqual(seen[-1].device, dbn.device)

        del classifier._batch_on_rbm_parameters
        with self.assertRaises(NotImplementedError):
            classifier.get_layer_activations(features)

        self.assertEqual(calls, [])
        self.assertEqual(seen[-1].device, _META)
        self.assertEqual(dbn.device, _CPU)

    def test_numpy_float32_is_not_aliased(self):
        """``torch.tensor`` must not share storage with a float32 NumPy caller.

        ``torch.FloatTensor`` on a float32 array aliases that buffer. A later
        in-place write would change the caller's data.
        """
        classifier = _classifier()
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        batch = np.arange(8, dtype=np.float32).reshape(2, 4).copy()
        snapshot = _numpy_snapshot(batch)
        seen = _spy_hidden(rbm)

        activations = classifier.get_layer_activations(batch)
        seen[0].view(-1)[0] = -1.0

        _assert_numpy_unchanged(self, batch, snapshot)
        self.assertNotEqual(seen[0].data_ptr(), snapshot["ptr"])
        self.assertTrue(np.isfinite(activations).all())
        np.testing.assert_allclose(
            activations,
            _expected(rbm, np.arange(8, dtype=np.float32).reshape(2, 4), torch.float32),
        )

    def test_numpy_list_input_matches_array_input(self):
        """A nested list takes the same NumPy-to-``torch.tensor`` path."""
        classifier = _classifier()
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        as_list = [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
        seen = _spy_hidden(rbm)

        from_list = classifier.get_layer_activations(as_list)
        from_array = classifier.get_layer_activations(_features())

        np.testing.assert_allclose(from_list, from_array)
        self.assertEqual(seen[0].device, _CPU)
        self.assertEqual(seen[0].dtype, torch.float32)

    def test_same_device_tensor_keeps_identity(self):
        """A matching CPU tensor keeps identity, ``data_ptr``, and contiguity."""
        for noncontiguous in (False, True):
            with self.subTest(noncontiguous=noncontiguous):
                classifier = _classifier()
                rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
                if noncontiguous:
                    base = torch.arange(16, dtype=torch.float32).reshape(2, 8)
                    batch = base[:, ::2]
                else:
                    batch = torch.arange(8, dtype=torch.float32).reshape(2, 4)
                values = batch.clone()
                pointer = batch.data_ptr()
                seen = _spy_hidden(rbm)
                self.assertEqual(batch.is_contiguous(), not noncontiguous)

                activations = classifier.get_layer_activations(batch)

                self.assertIs(seen[0], batch)
                self.assertEqual(seen[0].data_ptr(), pointer)
                self.assertEqual(batch.data_ptr(), pointer)
                self.assertEqual(seen[0].is_contiguous(), not noncontiguous)
                self.assertTrue(torch.equal(batch, values))
                np.testing.assert_allclose(
                    activations,
                    _expected(rbm, batch.contiguous().numpy(), torch.float32),
                )

    def test_meta_tensor_keeps_identity_when_parameters_are_meta(self):
        """A meta batch on meta weights is the same Python object.

        Every meta tensor reports ``data_ptr() == 0``, so identity is the
        Python object.
        """
        classifier = _classifier()
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        parent = nn.Module()
        parent.dbn = classifier.unsupervised_dbn._dbn
        _apply_device(parent, _META)
        batch = torch.empty(2, 4, device=_META)
        seen = _spy_hidden(rbm)

        with self.assertRaises(NotImplementedError):
            classifier.get_layer_activations(batch)

        self.assertIs(seen[0], batch)
        self.assertEqual(batch.data_ptr(), 0)
        self.assertEqual(seen[0].device, _META)

    def test_tensor_dtype_is_cast_to_the_parameter_dtype(self):
        """A float64 caller is presented as float32 and is not rewritten."""
        classifier = _classifier()
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        batch = torch.tensor(_features(), dtype=torch.float64)
        values = batch.clone()
        seen = _spy_hidden(rbm)

        activations = classifier.get_layer_activations(batch)

        self.assertIsNot(seen[0], batch)
        self.assertEqual(seen[0].dtype, torch.float32)
        self.assertEqual(batch.dtype, torch.float64)
        self.assertTrue(torch.equal(batch, values))
        np.testing.assert_allclose(
            activations, _expected(rbm, _features(), torch.float32)
        )

    def test_meta_tensor_against_cpu_weights_is_not_copied_out(self):
        """A meta batch is not copied onto CPU weights or a CUDA cache."""
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        dbn.device = _CUDA
        batch = torch.empty(2, 4, device=_META)
        seen = _spy_hidden(rbm)

        with self.assertRaises(NotImplementedError) as caught:
            classifier.get_layer_activations(batch)

        self.assertIn("meta", str(caught.exception).lower())
        self.assertEqual(seen, [])
        self.assertEqual(batch.device, _META)
        self.assertEqual(dbn.device, _CUDA)
        self.assertEqual(next(rbm.parameters()).device, _CPU)

    def test_multi_layer_return_all_and_layer_index(self):
        """Both layers run, each on CPU, and every return mode agrees."""
        classifier = _classifier(structure=(3, 2))
        dbn = classifier.unsupervised_dbn._dbn
        first = classifier.unsupervised_dbn.get_rbm_layer(0)
        second = classifier.unsupervised_dbn.get_rbm_layer(1)
        first_seen = _spy_hidden(first)
        second_seen = _spy_hidden(second)
        features = _features()

        activations = classifier.get_layer_activations(features, return_all_layers=True)

        expected_first = _expected(first, features, torch.float32)
        expected_second = _expected(second, expected_first, torch.float32)
        self.assertEqual(set(activations), {0, 1})
        self.assertEqual(activations[0].shape, (2, 3))
        self.assertEqual(activations[1].shape, (2, 2))
        np.testing.assert_allclose(activations[0], expected_first)
        np.testing.assert_allclose(activations[1], expected_second)
        self.assertEqual(first_seen[-1].shape, (2, 4))
        self.assertEqual(second_seen[-1].shape, (2, 3))
        self.assertEqual(second_seen[-1].device, _CPU)
        self.assertEqual(second_seen[-1].dtype, torch.float32)
        self.assertEqual(dbn.num_layers, 2)

        for index, expected in ((0, expected_first), (1, expected_second)):
            with self.subTest(layer=index):
                np.testing.assert_allclose(
                    classifier.get_layer_activations(features, layer_index=index),
                    expected,
                )

        second_seen.clear()
        classifier.get_layer_activations(features, layer_index=0)
        self.assertEqual(second_seen, [])

        with self.assertRaises(ValueError):
            classifier.get_layer_activations(features, layer_index=2)

    def test_each_layer_reads_its_own_parameter_device(self):
        """Moving only the second RBM must not change the first batch device."""
        classifier = _classifier(structure=(3, 2))
        first = classifier.unsupervised_dbn.get_rbm_layer(0)
        second = classifier.unsupervised_dbn.get_rbm_layer(1)
        first_seen = _spy_hidden(first)
        second_seen = _spy_hidden(second)
        features = _features()

        _apply_device(second, _META)

        activations = classifier.get_layer_activations(features, layer_index=0)
        self.assertEqual(first_seen[-1].device, _CPU)
        self.assertEqual(second_seen, [])
        np.testing.assert_allclose(
            activations, _expected(first, features, torch.float32)
        )

        with self.assertRaises(NotImplementedError):
            classifier.get_layer_activations(features, layer_index=1)
        self.assertEqual(second_seen[-1].device, _META)
        self.assertEqual(first.quadratic_coef.device, _CPU)

    def test_fallback_without_parameters_uses_construction_device(self):
        """No parameters follows a ``torch.device`` cache, then CPU.

        A real RBM always has parameters, so this fallback is unreachable in
        practice. A non-device value is ignored.
        """
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        bare = nn.Module()

        dbn.device = _META
        placed = classifier._batch_on_rbm_parameters(bare, _features())
        self.assertEqual(placed.device, _META)
        self.assertEqual(placed.dtype, torch.float32)
        self.assertEqual(tuple(placed.shape), (2, 4))

        for cached in (None, "meta", "cuda"):
            with self.subTest(cached=str(cached)):
                dbn.device = cached
                placed = classifier._batch_on_rbm_parameters(bare, _features())
                self.assertEqual(placed.device, _CPU)
                self.assertEqual(placed.dtype, torch.float32)


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

        _apply_device(parent, _CUDA)

        self.assertEqual(calls, [])
        self.assertEqual(dbn.device, _CPU)
        self.assertEqual(rbm.device.type, "cuda")
        activations = classifier.get_layer_activations(features)
        self.assertEqual(seen[0].device.type, "cuda")
        self.assertEqual(seen[0].dtype, torch.float32)
        np.testing.assert_allclose(
            activations,
            _expected_from_weights(
                weight, bias, rbm.num_visible, features, torch.float32
            ),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_parent_to_cuda_does_not_call_child_to(self):
        """Positional and keyword ``Module.to(cuda)`` do not call child ``.to``."""
        moves = (
            ("positional", (_CUDA,), {}),
            ("keyword", (), {"device": _CUDA}),
        )
        for label, args, kwargs in moves:
            with self.subTest(move=label):
                classifier = _classifier()
                dbn = classifier.unsupervised_dbn._dbn
                rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
                weight = rbm.quadratic_coef.detach().cpu().clone()
                bias = rbm.linear_bias.detach().cpu().clone()
                parent = nn.Module()
                parent.dbn = dbn
                calls = _record_to(rbm)
                seen = _spy_hidden(rbm)

                parent.to(*args, **kwargs)

                self.assertEqual(calls, [])
                self.assertEqual(dbn.device, _CPU)
                self.assertEqual(next(rbm.parameters()).device.type, "cuda")
                activations = classifier.get_layer_activations(_features())
                self.assertEqual(seen[0].device.type, "cuda")
                np.testing.assert_allclose(
                    activations,
                    _expected_from_weights(
                        weight, bias, rbm.num_visible, _features(), torch.float32
                    ),
                    atol=1e-5,
                    rtol=1e-5,
                )

    def test_cuda_batch_keeps_storage(self):
        """A CUDA batch that already matches the weights keeps its storage."""
        for noncontiguous in (False, True):
            with self.subTest(noncontiguous=noncontiguous):
                classifier = _classifier()
                rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
                parent = nn.Module()
                parent.dbn = classifier.unsupervised_dbn._dbn
                _apply_device(parent, _CUDA)
                if noncontiguous:
                    base = torch.arange(16, dtype=torch.float32, device=_CUDA)
                    batch = base.reshape(2, 8)[:, ::2]
                else:
                    batch = torch.arange(8, dtype=torch.float32, device=_CUDA)
                pointer = batch.data_ptr()
                values = batch.clone()
                seen = _spy_hidden(rbm)
                self.assertEqual(batch.is_contiguous(), not noncontiguous)

                classifier.get_layer_activations(batch)

                self.assertIs(seen[0], batch)
                self.assertEqual(batch.data_ptr(), pointer)
                self.assertEqual(seen[0].is_contiguous(), not noncontiguous)
                self.assertTrue(torch.equal(batch, values))

    def test_stale_cpu_cache_does_not_pull_cuda_weights(self):
        """A CPU cache must not pull CUDA weights to the host."""
        classifier = _classifier()
        dbn = classifier.unsupervised_dbn._dbn
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        weight = rbm.quadratic_coef.detach().cpu().clone()
        bias = rbm.linear_bias.detach().cpu().clone()
        parent = nn.Module()
        parent.dbn = dbn
        calls = _record_to(rbm)
        seen = _spy_hidden(rbm)

        _apply_device(parent, _CUDA)
        dbn.device = _CPU
        activations = classifier.get_layer_activations(_features())

        self.assertEqual(calls, [])
        self.assertEqual(dbn.device, _CPU)
        self.assertEqual(next(rbm.parameters()).device.type, "cuda")
        self.assertEqual(seen[-1].device.type, "cuda")
        np.testing.assert_allclose(
            activations,
            _expected_from_weights(
                weight, bias, rbm.num_visible, _features(), torch.float32
            ),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_dtype_only_move_keeps_cuda_device(self):
        """A dtype-only parent move does not pull CUDA parameters back to CPU."""
        classifier = _classifier()
        rbm = classifier.unsupervised_dbn.get_rbm_layer(0)
        parent = nn.Module()
        parent.dbn = classifier.unsupervised_dbn._dbn
        calls = _record_to(rbm)
        seen = _spy_hidden(rbm)
        _apply_device(parent, _CUDA)

        parent.to(dtype=torch.float64)

        self.assertEqual(calls, [])
        self.assertEqual(rbm.quadratic_coef.device.type, "cuda")
        self.assertEqual(rbm.quadratic_coef.dtype, torch.float64)
        weight = rbm.quadratic_coef.detach().cpu().clone()
        bias = rbm.linear_bias.detach().cpu().clone()
        activations = classifier.get_layer_activations(_features())
        self.assertEqual(seen[-1].device.type, "cuda")
        self.assertEqual(seen[-1].dtype, torch.float64)
        np.testing.assert_allclose(
            activations,
            _expected_from_weights(
                weight, bias, rbm.num_visible, _features(), torch.float64
            ),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_device_is_reread_across_cuda_and_cpu(self):
        """CUDA, then CPU. The CPU stop ignores a CUDA construction cache."""
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

        _apply_device(parent, _CUDA)
        classifier.get_layer_activations(features)
        self.assertEqual(seen[-1].device.type, "cuda")

        _apply_device(parent, _CPU)
        dbn.device = _CUDA
        activations = classifier.get_layer_activations(features)
        self.assertEqual(seen[-1].device, _CPU)
        self.assertEqual(dbn.device, _CUDA)
        self.assertEqual(rbm.device, _CPU)
        self.assertEqual(calls, [])
        np.testing.assert_allclose(
            activations,
            _expected_from_weights(
                weight, bias, rbm.num_visible, features, torch.float32
            ),
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
