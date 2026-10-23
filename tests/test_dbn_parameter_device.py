"""Regression tests for DBN activations versus RBM parameter devices.

``UnsupervisedDBN.device`` is a plain attribute captured at construction.
Parent ``to`` / ``_apply`` moves the stacked RBMs (merged #164 updates each
RBM's own cache) without calling ``UnsupervisedDBN.to`` and without refreshing
the DBN attribute. ``reconstruct`` and ``forward`` used to build the visible
batch from that stale attribute and subtract it from a reconstruction that
lives on the RBM parameters.

Open #242 overrides ``UnsupervisedDBN.to`` and adds a constructor ``device``
argument. A parent move never calls that override. These tests force
``self.device`` stale after the move, so refreshing the attribute inside
``.to`` would still fail them.

RBM ``get_hidden`` input coercion (open #1334) does not repair the
reconstruction error term: it keeps the caller's tensor. ``get_hidden`` also
still allocates a float32 buffer, so a float64 weight matmul stays with
#1334 / #1268. These tests follow ``tests/test_bm_device_sync.py``: meta
mechanism on every machine, CUDA behind ``skipUnless``.
"""

# Parent moves go through ``_apply`` and never call a child ``.to()``.
# pylint: disable=protected-access

import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from kaiwu.torch_plugin import UnsupervisedDBN


def _move_to_meta(tensor):
    """Move one parameter tensor onto the meta device."""
    return tensor.to(torch.device("meta"))


def _call_exporting_meta(fn):
    """Run ``fn`` when results live on meta and cannot be copied to NumPy.

    The production path still places activations and evaluates the
    reconstruction. Only the final ``cpu().numpy()`` hop is stubbed, matching
    the meta export helper used for Ising matrices.
    """

    def _cpu(tensor, memory_format=torch.preserve_format):
        del memory_format
        return tensor

    def _numpy(tensor):
        return np.zeros(tuple(tensor.shape), dtype=np.float32)

    with mock.patch.object(torch.Tensor, "cpu", _cpu):
        with mock.patch.object(torch.Tensor, "numpy", _numpy):
            return fn()


def _spy_get_hidden(original, seen):
    """Record the device of each visible batch passed into ``get_hidden``."""

    def _spy(s_visible, requires_grad=False, bernoulli=False):
        seen.append(s_visible.device)
        return original(
            s_visible,
            requires_grad=requires_grad,
            bernoulli=bernoulli,
        )

    return _spy


def _stub_float64_hidden(rbm, seen):
    """Record the visible batch and return a float64 hidden state.

    Real ``RestrictedBoltzmannMachine.get_hidden`` allocates ``float32``
    zeros, so a float64 weight matmul fails inside the RBM (#1334 / #1268).
    The stub lets ``reconstruct_with_rbm`` run its own matmul on float64
    weights while the DBN's placement of ``s_visible`` stays observable.
    """
    weight = rbm.quadratic_coef

    def _hidden(s_visible, requires_grad=False, bernoulli=False):
        del requires_grad, bernoulli
        seen.append((s_visible.device, s_visible.dtype))
        return torch.zeros(
            s_visible.shape[0],
            rbm.num_visible + rbm.num_hidden,
            dtype=weight.dtype,
            device=s_visible.device,
        )

    rbm.get_hidden = _hidden


class RecordingDBN(UnsupervisedDBN):
    """DBN that records every direct ``.to()`` invocation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct_to_calls = []

    def to(self, *args, **kwargs):
        """Record the call, then perform the real move."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def _moved_dbn(visible=4, hidden=None):
    """Build a trained DBN whose parameters were moved to meta.

    ``device`` is forced back to CPU afterwards so a later ``_apply`` sync of
    the cache cannot hide a stale-attribute bug. The returned object records
    direct ``.to()`` calls so a parent ``_apply`` can be shown to skip them.
    """
    if hidden is None:
        hidden = [3]
    parent = nn.Module()
    parent.dbn = RecordingDBN(hidden)
    parent.dbn.create_rbm_layer(visible)
    parent.dbn.mark_as_trained()
    parent._apply(_move_to_meta)
    parent.dbn.device = torch.device("cpu")
    return parent.dbn


class TestDbnParameterDevice(unittest.TestCase):
    """Reconstruction and forward placement must follow live RBM parameters."""

    def test_parameter_device_falls_back_without_parameters(self):
        """A parameter-free module reports the DBN attribute."""
        dbn = UnsupervisedDBN([3])
        empty = nn.Module()

        self.assertEqual(dbn._parameter_device(empty), dbn.device)

        dbn.device = torch.device("meta")
        self.assertEqual(dbn._parameter_device(empty), torch.device("meta"))

    def test_parent_to_does_not_call_dbn_to(self):
        """Parent ``.to`` moves parameters through ``_apply``, not ``DBN.to``.

        Open #242 overrides ``UnsupervisedDBN.to``. A nested DBN never
        receives that call, whether the device is positional or ``device=``.
        """
        parent = nn.Module()
        parent.dbn = RecordingDBN([3])
        parent.dbn.create_rbm_layer(4)
        parent.dbn.mark_as_trained()
        target = next(parent.dbn.parameters()).device

        parent.to(target)
        parent.to(device=target)

        self.assertEqual(parent.dbn.direct_to_calls, [])
        self.assertEqual(parent.dbn.device, target)
        self.assertEqual(next(parent.dbn.parameters()).device, target)

    def test_reconstruct_and_forward_on_construction_device(self):
        """Unmoved CPU models still return finite NumPy arrays."""
        dbn = UnsupervisedDBN([3])
        dbn.create_rbm_layer(4)
        dbn.mark_as_trained()
        data = np.random.rand(2, 4).astype(np.float32)

        visible, errors = dbn.reconstruct(data)
        features = dbn.forward(data)
        transformed = dbn.transform(data)

        self.assertEqual(visible.shape, (2, 4))
        self.assertEqual(errors.shape, (2,))
        self.assertEqual(features.shape, (2, 3))
        np.testing.assert_allclose(features, transformed)
        self.assertTrue(np.isfinite(visible).all())
        self.assertTrue(np.isfinite(features).all())

    def test_reconstruct_after_parent_apply_accepts_cpu_batch(self):
        """Visible error must not mix a stale CPU batch with meta parameters."""
        dbn = _moved_dbn()
        data = np.random.rand(2, 4).astype(np.float32)
        seen = []
        dbn.rbm_layers[0].get_hidden = _spy_get_hidden(
            dbn.rbm_layers[0].get_hidden, seen
        )

        visible, errors = _call_exporting_meta(lambda: dbn.reconstruct(data))

        self.assertEqual(dbn.direct_to_calls, [])
        self.assertEqual(dbn.device, torch.device("cpu"))
        self.assertEqual(next(dbn.parameters()).device.type, "meta")
        self.assertEqual([device.type for device in seen], ["meta"])
        self.assertEqual(visible.shape, (2, 4))
        self.assertEqual(errors.shape, (2,))

    def test_forward_hands_each_layer_its_parameter_device(self):
        """Each RBM must see activations already on that layer's device."""
        dbn = UnsupervisedDBN([3, 2])
        dbn.create_rbm_layer(4)
        dbn.mark_as_trained()
        dbn.rbm_layers[1]._apply(_move_to_meta)
        dbn.device = torch.device("cpu")
        seen = []
        for rbm in dbn.rbm_layers:
            rbm.get_hidden = _spy_get_hidden(rbm.get_hidden, seen)

        data = np.random.rand(2, 4).astype(np.float32)
        features = _call_exporting_meta(lambda: dbn.forward(data))

        self.assertEqual(dbn.device, torch.device("cpu"))
        self.assertEqual([device.type for device in seen], ["cpu", "meta"])
        self.assertEqual(features.shape, (2, 2))

    def test_reconstruct_selected_layer_follows_that_layer(self):
        """``layer_index`` must use that RBM, not the DBN device cache."""
        dbn = UnsupervisedDBN([3, 2])
        dbn.create_rbm_layer(4)
        dbn.rbm_layers[0]._apply(_move_to_meta)
        dbn.device = torch.device("cpu")
        seen = []
        dbn.rbm_layers[0].get_hidden = _spy_get_hidden(
            dbn.rbm_layers[0].get_hidden, seen
        )
        data = np.random.rand(2, 4).astype(np.float32)

        visible, errors = _call_exporting_meta(
            lambda: dbn.reconstruct(data, layer_index=0)
        )

        self.assertEqual(dbn.rbm_layers[0].quadratic_coef.device.type, "meta")
        self.assertEqual(dbn.rbm_layers[1].quadratic_coef.device.type, "cpu")
        self.assertEqual([device.type for device in seen], ["meta"])
        self.assertEqual(visible.shape, (2, 4))
        self.assertEqual(errors.shape, (2,))

    def test_upper_layer_ignores_stale_device_and_sibling_meta(self):
        """Layer 1 stays on its own parameters when the DBN cache is meta."""
        dbn = UnsupervisedDBN([3, 2])
        dbn.create_rbm_layer(4)
        dbn.rbm_layers[0]._apply(_move_to_meta)
        dbn.device = torch.device("meta")
        layer = dbn.rbm_layers[1]
        seen = []
        layer.get_hidden = _spy_get_hidden(layer.get_hidden, seen)
        data = np.random.rand(2, layer.num_visible).astype(np.float32)

        visible, errors = dbn.reconstruct(data, layer_index=1)

        self.assertEqual(dbn.device.type, "meta")
        self.assertEqual(dbn.rbm_layers[0].quadratic_coef.device.type, "meta")
        self.assertNotEqual(layer.quadratic_coef.device.type, "meta")
        self.assertEqual(seen, [layer.quadratic_coef.device])
        self.assertEqual(visible.shape, data.shape)
        self.assertEqual(errors.shape, (data.shape[0],))
        self.assertTrue(np.isfinite(visible).all())
        self.assertTrue(np.isfinite(errors).all())

    def test_dtype_only_move_places_batch_on_parameter_device(self):
        """``.to(dtype=...)`` must not retarget the device or the visible batch.

        Positional ``.to(torch.float64)`` is ``nn.Module``'s dtype move. The
        visible batch stays float32 (the ``FloatTensor`` contract) and is
        placed on the parameter device after ``self.device`` is forced to meta.
        """
        dbn = UnsupervisedDBN([3])
        dbn.create_rbm_layer(4)
        dbn.mark_as_trained()
        constructed = dbn.device
        dbn.to(torch.float64)
        weight = dbn.rbm_layers[0].quadratic_coef

        self.assertEqual(dbn.device, constructed)
        self.assertEqual(weight.dtype, torch.float64)
        self.assertEqual(weight.device, constructed)
        self.assertEqual(dbn._parameter_device(dbn.rbm_layers[0]), weight.device)

        dbn.to(dtype=torch.float64)
        dbn.device = torch.device("meta")
        seen = []
        _stub_float64_hidden(dbn.rbm_layers[0], seen)
        data = np.random.rand(2, 4).astype(np.float32)

        visible, errors = dbn.reconstruct(data)
        features = dbn.transform(data)

        self.assertEqual(dbn.device.type, "meta")
        self.assertEqual(weight.device, constructed)
        self.assertEqual(
            seen,
            [(constructed, torch.float32), (constructed, torch.float32)],
        )
        self.assertEqual(visible.shape, (2, 4))
        self.assertEqual(visible.dtype, np.float64)
        self.assertEqual(errors.shape, (2,))
        self.assertEqual(features.shape, (2, 3))
        self.assertTrue(np.isfinite(visible).all())
        self.assertTrue(np.isfinite(features).all())

    def test_non_float32_inputs_ignore_stale_device(self):
        """Integer, float16/64, empty, and noncontiguous batches stay finite."""
        dbn = UnsupervisedDBN([3])
        dbn.create_rbm_layer(4)
        dbn.mark_as_trained()
        parameter_device = next(dbn.parameters()).device
        dbn.device = torch.device("meta")
        rows = np.random.rand(4, 2).astype(np.float32).T
        cases = {
            "int": np.random.randint(0, 2, size=(2, 4)),
            "float64": np.random.rand(2, 4),
            "float16": np.random.rand(2, 4).astype(np.float16),
            "empty": np.zeros((0, 4), dtype=np.float32),
            "noncontiguous": rows,
        }

        for name, data in cases.items():
            with self.subTest(kind=name):
                visible, errors = dbn.reconstruct(data)
                features = dbn.transform(data)

                self.assertEqual(dbn.device.type, "meta")
                self.assertEqual(next(dbn.parameters()).device, parameter_device)
                self.assertEqual(visible.shape, (data.shape[0], 4))
                self.assertEqual(errors.shape, (data.shape[0],))
                self.assertEqual(features.shape, (data.shape[0], 3))
                self.assertTrue(np.isfinite(visible).all())
                self.assertTrue(np.isfinite(features).all())


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDbnParameterDeviceCuda(unittest.TestCase):
    """End-to-end coverage of construct-on-CUDA / parent-move-to-CPU."""

    def test_parent_move_to_cpu_reconstructs_and_transforms(self):
        """Stale ``self.device`` must not survive a parent move off CUDA."""
        parent = nn.Module()
        parent.dbn = RecordingDBN([3])
        parent.dbn.create_rbm_layer(4)
        parent.dbn.mark_as_trained()
        self.assertEqual(next(parent.dbn.parameters()).device.type, "cuda")

        parent.to("cpu")
        parent.to(device="cpu")
        self.assertEqual(parent.dbn.direct_to_calls, [])
        parent.dbn.device = torch.device("cuda")
        data = np.random.rand(2, 4).astype(np.float32)

        visible, errors = parent.dbn.reconstruct(data)
        features = parent.dbn.forward(data)
        transformed = parent.dbn.transform(data)

        self.assertEqual(next(parent.dbn.parameters()).device.type, "cpu")
        self.assertEqual(parent.dbn.device.type, "cuda")
        self.assertEqual(visible.shape, (2, 4))
        self.assertEqual(errors.shape, (2,))
        self.assertTrue(np.isfinite(visible).all())
        self.assertTrue(np.isfinite(errors).all())
        self.assertEqual(features.shape, (2, 3))
        np.testing.assert_allclose(features, transformed)

    def test_dtype_only_move_stays_on_cuda(self):
        """A dtype-only parent move must not call ``DBN.to`` or leave CUDA."""
        parent = nn.Module()
        parent.dbn = RecordingDBN([3])
        parent.dbn.create_rbm_layer(4)
        parent.dbn.mark_as_trained()
        parent.to(dtype=torch.float64)
        weight = parent.dbn.rbm_layers[0].quadratic_coef

        self.assertEqual(parent.dbn.direct_to_calls, [])
        self.assertEqual(weight.dtype, torch.float64)
        self.assertEqual(weight.device.type, "cuda")
        self.assertEqual(parent.dbn.device.type, "cuda")

        parent.dbn.device = torch.device("cpu")
        seen = []
        _stub_float64_hidden(parent.dbn.rbm_layers[0], seen)
        data = np.random.rand(2, 4).astype(np.float32)
        visible, errors = parent.dbn.reconstruct(data)

        self.assertEqual(parent.dbn.device, torch.device("cpu"))
        self.assertEqual(weight.device.type, "cuda")
        self.assertEqual(seen, [(weight.device, torch.float32)])
        self.assertEqual(visible.shape, (2, 4))
        self.assertEqual(visible.dtype, np.float64)
        self.assertEqual(errors.shape, (2,))
        self.assertTrue(np.isfinite(visible).all())


if __name__ == "__main__":
    unittest.main()
