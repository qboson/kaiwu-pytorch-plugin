"""Regression tests for DBN activations versus RBM parameter devices.

``UnsupervisedDBN.device`` is a plain attribute captured at construction.
Parent ``_apply`` moves the stacked RBMs (merged #164 updates each RBM's own
cache) without refreshing the DBN attribute. ``reconstruct`` then builds the
visible batch on that stale device and subtracts it from a reconstruction that
lives on the RBM parameters.

RBM ``get_hidden`` input coercion (open #1334) does not repair this: the error
term keeps the caller's tensor. These tests follow the meta-device bar in
``tests/test_bm_device_sync.py`` so they run without a GPU.
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


def _moved_dbn(visible=4, hidden=None):
    """Build a trained DBN whose parameters were moved to meta.

    ``device`` is forced back to CPU afterwards so a later ``_apply`` sync of
    the cache cannot hide a stale-attribute bug.
    """
    if hidden is None:
        hidden = [3]
    parent = nn.Module()
    parent.dbn = UnsupervisedDBN(hidden)
    parent.dbn.create_rbm_layer(visible)
    parent.dbn.mark_as_trained()
    parent._apply(_move_to_meta)
    parent.dbn.device = torch.device("cpu")
    return parent.dbn


class TestDbnParameterDevice(unittest.TestCase):
    """Reconstruction and forward placement must follow live RBM parameters."""

    def test_reconstruct_and_forward_on_construction_device(self):
        """Unmoved CPU models still return finite NumPy arrays."""
        dbn = UnsupervisedDBN([3])
        dbn.create_rbm_layer(4)
        dbn.mark_as_trained()
        data = np.random.rand(2, 4).astype(np.float32)

        visible, errors = dbn.reconstruct(data)
        features = dbn.forward(data)

        self.assertEqual(visible.shape, (2, 4))
        self.assertEqual(errors.shape, (2,))
        self.assertEqual(features.shape, (2, 3))
        self.assertTrue(np.isfinite(visible).all())
        self.assertTrue(np.isfinite(features).all())

    def test_reconstruct_after_parent_apply_accepts_cpu_batch(self):
        """Visible error must not mix a stale CPU batch with meta parameters."""
        dbn = _moved_dbn()
        data = np.random.rand(2, 4).astype(np.float32)

        visible, errors = _call_exporting_meta(lambda: dbn.reconstruct(data))

        self.assertEqual(dbn.device, torch.device("cpu"))
        self.assertEqual(next(dbn.parameters()).device.type, "meta")
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
        data = np.random.rand(2, 4).astype(np.float32)

        visible, errors = _call_exporting_meta(
            lambda: dbn.reconstruct(data, layer_index=0)
        )

        self.assertEqual(dbn.rbm_layers[0].quadratic_coef.device.type, "meta")
        self.assertEqual(dbn.rbm_layers[1].quadratic_coef.device.type, "cpu")
        self.assertEqual(visible.shape, (2, 4))
        self.assertEqual(errors.shape, (2,))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDbnParameterDeviceCuda(unittest.TestCase):
    """End-to-end coverage of construct-on-CUDA / parent-move-to-CPU."""

    def test_parent_move_to_cpu_reconstructs_and_transforms(self):
        """Stale ``self.device`` must not survive a parent move off CUDA."""
        parent = nn.Module()
        parent.dbn = UnsupervisedDBN([3])
        parent.dbn.create_rbm_layer(4)
        parent.dbn.mark_as_trained()
        self.assertEqual(next(parent.dbn.parameters()).device.type, "cuda")

        parent.to("cpu")
        parent.dbn.device = torch.device("cuda")
        data = np.random.rand(2, 4).astype(np.float32)

        visible, errors = parent.dbn.reconstruct(data)
        features = parent.dbn.forward(data)

        self.assertEqual(next(parent.dbn.parameters()).device.type, "cpu")
        self.assertEqual(parent.dbn.device.type, "cuda")
        self.assertEqual(visible.shape, (2, 4))
        self.assertEqual(errors.shape, (2,))
        self.assertTrue(np.isfinite(visible).all())
        self.assertTrue(np.isfinite(errors).all())
        self.assertEqual(features.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
