"""Regression tests for ``MLPClassifier`` device alignment.

``MLPClassifier`` caches ``device`` at construction. A parent ``_apply``
(merged #164) moves ``self.model`` without calling the child ``.to`` or
updating that attribute. ``predict`` and ``predict_proba`` follow the live
parameter device and dtype.

NumPy is copied, so a float32 buffer is not aliased by ``torch.FloatTensor``.
An already-aligned tensor keeps identity and ``data_ptr``. Cache placement
after ``parent._apply(meta)`` raises ``RuntimeError`` naming both devices.
The opposite mix points the cache at ``meta`` while weights stay on CPU.
``Module.to(cpu)`` cannot copy out of ``meta``. CUDA tests are collected
and skipped without a GPU.

``RBMRunner`` (#1400) and feature-selection classifiers (#1393) differ.
"""

# Parent moves go through ``_apply`` (see merged #164).
# pylint: disable=protected-access

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

META = torch.device("meta")
CPU = torch.device("cpu")


def _load_classifier():
    """Load the example classifier without the MNIST helper stack."""
    helpers = types.ModuleType("utils.helpers")
    helpers.plot_training_curves = lambda *args, **kwargs: None
    utils_mod = types.ModuleType("utils")
    utils_mod.helpers = helpers
    sys.modules["utils"] = utils_mod
    sys.modules["utils.helpers"] = helpers
    path = (
        Path(__file__).resolve().parents[1]
        / "example"
        / "qvae_mnist"
        / "downstream"
        / "classifier.py"
    )
    spec = importlib.util.spec_from_file_location("mlp_classifier_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLF = _load_classifier()


def _classifier(**kwargs):
    """Build a tiny classifier on the construction device."""
    kwargs.setdefault("input_dim", 4)
    kwargs.setdefault("hidden_dims", [4])
    kwargs.setdefault("output_dim", 3)
    kwargs.setdefault("device", CPU)
    kwargs.setdefault("epochs_mlp", 1)
    kwargs.setdefault("batch_size_mlp", 8)
    return CLF.MLPClassifier(**kwargs)


def _batch(rows: int = 5) -> np.ndarray:
    """Return a deterministic float32 feature matrix."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((rows, 4)).astype(np.float32)


def _apply_device(module: nn.Module, device: torch.device) -> None:
    """Move parameters through ``_apply``, the parent-module path."""
    module._apply(lambda tensor: tensor.to(device))


def _retarget_storage(device: torch.device):
    """Return an ``_apply`` callback that places zero storage on ``device``.

    ``Module.to`` cannot copy out of ``meta``. Replacing storage still goes
    through ``_apply``, so the parameter device changes without ``.to()``.
    Zeros are readable, which is what a later CPU score needs.
    """

    def _move(tensor: torch.Tensor) -> torch.Tensor:
        """Place ``tensor`` on ``device`` without reading its data."""
        if tensor.device == device:
            return tensor
        return torch.zeros(tensor.shape, dtype=tensor.dtype, device=device)

    return _move


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


def _ready(**kwargs):
    """Return a classifier, a parent that owns its network, and ``.to`` calls."""
    clf = _classifier(**kwargs)
    clf.model = clf._create_model()
    calls = _record_to(clf.model)
    parent = nn.Module()
    parent.model = clf.model
    return clf, parent, calls


def _spy(module: nn.Module) -> dict:
    """Record the tensor ``module`` actually receives."""
    seen: dict = {}
    original = module.forward

    def _wrapped(inputs):
        """Store the batch, then run the real forward."""
        seen["inputs"] = inputs
        return original(inputs)

    module.forward = _wrapped
    return seen


def _reference(model: nn.Module, batch: np.ndarray | torch.Tensor):
    """Labels and probabilities from the live weights, in eval mode.

    The batch uses the parameter device and dtype. A tensor that already
    matches is not copied. NumPy is copied.
    """
    parameter = next(model.parameters())
    if torch.is_tensor(batch):
        features = batch.detach().to(dtype=parameter.dtype, device=parameter.device)
    else:
        features = torch.tensor(
            np.asarray(batch), dtype=parameter.dtype, device=parameter.device
        )
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(features)
    finally:
        model.train(training)
    probabilities = torch.softmax(logits, dim=1)
    return (
        logits.argmax(dim=1).detach().cpu().numpy(),
        probabilities.detach().cpu().numpy(),
    )


def _cpu_clone(clf) -> nn.Module:
    """Return a CPU eval copy of ``clf.model`` with the same dtype and values."""
    clone = clf._create_model()
    source = clf.model.state_dict()
    dtype = next(iter(source.values())).dtype
    if dtype != torch.float32:
        clone = clone.to(dtype=dtype)
    clone.load_state_dict(
        {key: value.detach().cpu().clone() for key, value in source.items()}
    )
    clone.eval()
    return clone


def _snapshot(array: np.ndarray):
    """Capture NumPy storage identity. Values are copied for a later compare."""
    return (
        array.__array_interface__["data"][0],
        array.shape,
        array.dtype,
        array.flags.c_contiguous,
        array.copy(),
    )


def _assert_unchanged(case: unittest.TestCase, array: np.ndarray, snap) -> None:
    """Assert ``array`` kept its storage, shape, dtype, layout, and values."""
    pointer, shape, dtype, contiguous, values = snap
    case.assertEqual(array.__array_interface__["data"][0], pointer)
    case.assertEqual(array.shape, shape)
    case.assertEqual(array.dtype, dtype)
    case.assertEqual(array.flags.c_contiguous, contiguous)
    np.testing.assert_array_equal(array, values)


class TestMLPClassifierParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_cpu_predict_matches_reference_and_keeps_input(self):
        """An unmoved classifier matches the linear head and does not copy in place."""
        clf, _parent, _calls = _ready()
        batch = _batch()
        original = _snapshot(batch)
        labels, probabilities = _reference(clf.model, batch)

        predicted = clf.predict(batch)
        scored = clf.predict_proba(batch)

        np.testing.assert_array_equal(predicted, labels)
        np.testing.assert_allclose(scored, probabilities)
        self.assertEqual(predicted.shape, (batch.shape[0],))
        self.assertEqual(scored.shape, (batch.shape[0], clf.output_dim))
        self.assertTrue(np.isfinite(scored).all())
        np.testing.assert_allclose(
            scored.sum(axis=1), np.ones(batch.shape[0]), atol=1e-5
        )
        _assert_unchanged(self, batch, original)
        self.assertEqual(clf.device, CPU)
        self.assertEqual(clf._parameter_device(), CPU)

    def test_noncontiguous_float64_numpy_stays_float32(self):
        """A column slice of float64 values is copied as float32 and left intact."""
        clf, _parent, _calls = _ready()
        wide = np.random.default_rng(1).standard_normal((6, 8)).astype(np.float64)
        batch = np.ascontiguousarray(wide)[:, :4]
        self.assertFalse(batch.flags.c_contiguous)
        original = _snapshot(batch)

        scored = clf.predict_proba(batch)

        self.assertEqual(scored.dtype, np.float32)
        np.testing.assert_allclose(scored, _reference(clf.model, batch)[1])
        _assert_unchanged(self, batch, original)

    def test_parent_to_does_not_call_model_to(self):
        """Positional and keyword ``Module.to(meta)`` do not call child ``.to``."""
        for kwargs in (None, {"device": META}):
            with self.subTest(kwargs=kwargs):
                clf, parent, calls = _ready()
                if kwargs is None:
                    parent.to(META)
                else:
                    parent.to(**kwargs)

                self.assertEqual(calls, [])
                self.assertEqual(list(clf.model.named_buffers()), [])
                for parameter in clf.model.parameters():
                    self.assertEqual(parameter.device, META)
                self.assertEqual(clf.device, CPU)
                self.assertEqual(clf._parameter_device(), META)

    def test_parent_apply_places_predict_batch_on_parameter_device(self):
        """``predict`` must feed the linear stack on the live parameter device."""
        clf, parent, calls = _ready()
        batch = _batch()
        original = _snapshot(batch)
        seen = _spy(clf.model)
        _apply_device(parent, META)

        with self.assertRaises(NotImplementedError) as caught:
            clf.predict(batch)

        self.assertIn("meta", str(caught.exception).lower())
        self.assertEqual(seen["inputs"].device, META)
        self.assertEqual(seen["inputs"].dtype, torch.float32)
        _assert_unchanged(self, batch, original)
        self.assertEqual(clf.device, CPU)
        self.assertEqual(calls, [])

    def test_parent_apply_places_proba_batch_on_parameter_device(self):
        """``predict_proba`` follows the same parameter device as ``predict``."""
        clf, parent, calls = _ready()
        batch = _batch()
        seen = _spy(clf.model)
        _apply_device(parent, META)

        with self.assertRaises(NotImplementedError) as caught:
            clf.predict_proba(batch)

        self.assertIn("meta", str(caught.exception).lower())
        self.assertEqual(seen["inputs"].device, META)
        self.assertEqual(seen["inputs"].dtype, torch.float32)
        self.assertEqual(clf.device, CPU)
        self.assertEqual(calls, [])

    def test_stale_device_attribute_does_not_redirect_inference(self):
        """Assigning ``device`` must not pull a CPU network onto another device."""
        clf, _parent, _calls = _ready()
        batch = _batch()
        labels, probabilities = _reference(clf.model, batch)
        clf.device = META

        predicted = clf.predict(batch)
        scored = clf.predict_proba(batch)

        np.testing.assert_array_equal(predicted, labels)
        np.testing.assert_allclose(scored, probabilities)
        self.assertEqual(next(clf.model.parameters()).device, CPU)
        self.assertEqual(clf.device, META)
        self.assertEqual(clf._parameter_device(), CPU)

    def test_cuda_preference_does_not_redirect_cpu_weights(self):
        """A CUDA construction preference cannot move inference off CPU weights."""
        clf, _parent, _calls = _ready()
        batch = _batch()
        labels, probabilities = _reference(clf.model, batch)
        clf.device = torch.device("cuda")

        np.testing.assert_array_equal(clf.predict(batch), labels)
        np.testing.assert_allclose(clf.predict_proba(batch), probabilities)
        self.assertEqual(next(clf.model.parameters()).device, CPU)

    def test_dtype_only_move_keeps_cpu_device(self):
        """A direct dtype move stays on CPU, scores, and does not alias NumPy."""
        clf, _parent, calls = _ready()
        batch = np.arange(20, dtype=np.float32).reshape(5, 4).copy()
        original = _snapshot(batch)

        clf.model.to(dtype=torch.float64)

        weight = next(clf.model.parameters())
        self.assertEqual(weight.dtype, torch.float64)
        self.assertEqual(weight.device, CPU)
        self.assertEqual(clf.device, CPU)
        self.assertEqual(clf._parameter_device(), CPU)
        self.assertEqual(calls, [((), {"dtype": torch.float64})])
        placed = clf._features_on_parameter_device(batch)
        self.assertEqual(placed.dtype, torch.float64)
        self.assertEqual(placed.device, CPU)
        self.assertNotEqual(placed.data_ptr(), original[0])
        labels, probabilities = _reference(clf.model, batch)
        seen = _spy(clf.model)
        np.testing.assert_array_equal(clf.predict(batch), labels)
        scored = clf.predict_proba(batch)
        self.assertEqual(seen["inputs"].dtype, torch.float64)
        self.assertEqual(seen["inputs"].device, CPU)
        self.assertNotEqual(seen["inputs"].data_ptr(), original[0])
        np.testing.assert_allclose(scored, probabilities)
        self.assertEqual(scored.dtype, np.float64)
        self.assertTrue(np.isfinite(scored).all())
        _assert_unchanged(self, batch, original)
        self.assertEqual(calls, [((), {"dtype": torch.float64})])

    def test_new_model_stays_on_construction_device(self):
        """A replacement network is built on the preference, not the moved one."""
        clf, parent, _calls = _ready()
        _apply_device(parent, META)
        fresh = clf._create_model()

        self.assertEqual(next(fresh.parameters()).device, CPU)
        self.assertEqual(next(clf.model.parameters()).device, META)
        self.assertEqual(clf.device, CPU)

    def test_construction_device_attribute_survives_parent_move(self):
        """``device`` stays the preference passed to ``__init__``.

        It is an estimator parameter. Replacing it with a live view of the
        weights would make a later ``fit`` follow whatever device a parent
        last applied. ``get_params`` is not asserted: this estimator already
        stores ``batch_size_mlp`` as ``batch_size``.
        """
        clf, parent, _calls = _ready()

        _apply_device(parent, META)

        self.assertEqual(clf.device, CPU)
        self.assertEqual(clf._parameter_device(), META)

    def test_parameter_device_is_reread_after_a_later_move(self):
        """``.to(cpu)`` cannot leave meta; a later zero-storage move is reread."""
        clf, parent, calls = _ready()
        batch = _batch()
        original = _snapshot(batch)
        _apply_device(parent, META)
        self.assertEqual(clf._parameter_device(), META)

        with self.assertRaises(NotImplementedError) as blocked:
            parent.to(CPU)
        self.assertIn("meta", str(blocked.exception).lower())
        self.assertEqual(next(clf.model.parameters()).device, META)
        self.assertEqual(calls, [])

        # ``.to(cpu)`` cannot copy out of meta; ``_apply`` still retargets.
        parent._apply(_retarget_storage(CPU))
        clf.device = META
        labels, probabilities = _reference(clf.model, batch)

        np.testing.assert_array_equal(clf.predict(batch), labels)
        np.testing.assert_allclose(clf.predict_proba(batch), probabilities)
        self.assertEqual(clf._parameter_device(), CPU)
        self.assertEqual(clf.device, META)
        self.assertEqual(calls, [])
        _assert_unchanged(self, batch, original)

        seen = _spy(clf.model)
        _apply_device(parent, META)
        for method_name in ("predict", "predict_proba"):
            with self.subTest(method=method_name):
                with self.assertRaises(NotImplementedError) as caught:
                    getattr(clf, method_name)(batch)
                self.assertIn("meta", str(caught.exception).lower())
                self.assertEqual(seen["inputs"].device, META)
                self.assertEqual(seen["inputs"].dtype, torch.float32)
        self.assertEqual(calls, [])
        self.assertEqual(clf.device, META)
        self.assertEqual(clf._parameter_device(), META)
        _assert_unchanged(self, batch, original)


class TestMLPClassifierCallerStorage(unittest.TestCase):
    """Caller storage, dtype, and the construction-cache mix.

    These run on every machine. ``meta`` stands in for an accelerator.
    """

    def test_numpy_float32_is_not_aliased(self):
        """Float32 NumPy is copied. ``torch.FloatTensor`` would alias it."""
        clf, _parent, _calls = _ready()
        batch = np.arange(20, dtype=np.float32).reshape(5, 4).copy()
        pointer = batch.__array_interface__["data"][0]
        aliased = torch.FloatTensor(batch)
        self.assertEqual(aliased.data_ptr(), pointer)

        for method_name in ("predict", "predict_proba"):
            with self.subTest(method=method_name):
                seen = _spy(clf.model)
                getattr(clf, method_name)(batch)
                seen["inputs"].view(-1)[0] = -1
                self.assertEqual(batch.__array_interface__["data"][0], pointer)
                self.assertNotEqual(seen["inputs"].data_ptr(), pointer)
                self.assertEqual(float(batch.ravel()[0]), 0.0)
                self.assertEqual(seen["inputs"].dtype, torch.float32)
                self.assertEqual(seen["inputs"].device, CPU)

    def test_same_device_cpu_tensor_keeps_identity(self):
        """A float32 CPU tensor already on the parameter device keeps storage."""
        clf, _parent, _calls = _ready()
        batch = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        pointer = batch.data_ptr()
        labels, probabilities = _reference(clf.model, batch)
        placed = clf._features_on_parameter_device(batch)
        self.assertIs(placed, batch)

        seen = _spy(clf.model)
        head = _spy(clf.model[-1])
        predicted = clf.predict(batch)
        scored = clf.predict_proba(batch)

        self.assertIs(seen["inputs"], batch)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(batch.device, CPU)
        self.assertEqual(batch.dtype, torch.float32)
        self.assertTrue(
            torch.equal(batch, torch.arange(20, dtype=torch.float32).reshape(5, 4))
        )
        self.assertEqual(head["inputs"].device, CPU)
        self.assertIsNot(head["inputs"], batch)
        np.testing.assert_array_equal(predicted, labels)
        np.testing.assert_allclose(scored, probabilities)

    def test_noncontiguous_cpu_tensor_keeps_storage(self):
        """A noncontiguous CPU view keeps its storage when the device matches."""
        clf, _parent, _calls = _ready()
        base = torch.arange(40, dtype=torch.float32).reshape(5, 8)
        view = base[:, :4]
        self.assertFalse(view.is_contiguous())
        pointer = view.data_ptr()
        labels, probabilities = _reference(clf.model, view)
        placed = clf._features_on_parameter_device(view)
        self.assertIs(placed, view)
        self.assertFalse(placed.is_contiguous())

        seen = _spy(clf.model)
        predicted = clf.predict(view)
        scored = clf.predict_proba(view)

        self.assertIs(seen["inputs"], view)
        self.assertFalse(seen["inputs"].is_contiguous())
        self.assertEqual(view.data_ptr(), pointer)
        self.assertFalse(view.is_contiguous())
        self.assertEqual(view.device, CPU)
        self.assertTrue(
            torch.equal(base, torch.arange(40, dtype=torch.float32).reshape(5, 8))
        )
        np.testing.assert_array_equal(predicted, labels)
        np.testing.assert_allclose(scored, probabilities)

    def test_meta_batch_already_aligned_keeps_storage(self):
        """A meta batch that already matches the weights keeps its storage."""
        clf, parent, calls = _ready()
        _apply_device(parent, META)
        batch = torch.empty(5, 4, dtype=torch.float32, device=META)
        pointer = batch.data_ptr()
        try:
            placed = clf._features_on_parameter_device(batch)
        except TypeError as exc:
            self.fail(f"aligned meta tensor was sent through NumPy: {exc}")
        self.assertIs(placed, batch)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(placed.dtype, torch.float32)
        self.assertEqual(placed.device, META)

        seen = _spy(clf.model)
        head = _spy(clf.model[-1])
        for method_name in ("predict", "predict_proba"):
            with self.subTest(method=method_name):
                with self.assertRaises(NotImplementedError) as caught:
                    getattr(clf, method_name)(batch)
                self.assertIn("meta", str(caught.exception).lower())
                self.assertIs(seen["inputs"], batch)
                self.assertEqual(batch.data_ptr(), pointer)
                self.assertEqual(seen["inputs"].dtype, torch.float32)
                self.assertEqual(head["inputs"].device, META)
                self.assertEqual(head["inputs"].dtype, torch.float32)
                self.assertIsNot(head["inputs"], batch)
        self.assertEqual(calls, [])
        self.assertEqual(clf.device, CPU)
        self.assertEqual(clf._parameter_device(), META)

    def test_cpu_tensor_storage_survives_meta_apply(self):
        """A CPU tensor stays put while the linear stack observes a meta copy."""
        clf, parent, calls = _ready()
        batch = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        pointer = batch.data_ptr()
        before = batch.clone()
        _apply_device(parent, META)
        seen = _spy(clf.model)
        for method_name in ("predict", "predict_proba"):
            with self.subTest(method=method_name):
                with self.assertRaises(NotImplementedError) as caught:
                    getattr(clf, method_name)(batch)
                self.assertIn("meta", str(caught.exception).lower())
                self.assertIsNot(seen["inputs"], batch)
                self.assertEqual(seen["inputs"].device, META)
                self.assertEqual(seen["inputs"].dtype, torch.float32)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(batch.device, CPU)
        self.assertEqual(batch.dtype, torch.float32)
        self.assertTrue(torch.equal(batch, before))
        self.assertEqual(calls, [])
        self.assertEqual(clf.device, CPU)

        base = torch.arange(40, dtype=torch.float32).reshape(5, 8)
        view = base[:, :4]
        view_pointer = view.data_ptr()
        self.assertFalse(view.is_contiguous())
        with self.assertRaises(NotImplementedError):
            clf.predict(view)
        self.assertEqual(view.data_ptr(), view_pointer)
        self.assertFalse(view.is_contiguous())
        self.assertEqual(view.device, CPU)
        self.assertIsNot(seen["inputs"], view)
        self.assertEqual(seen["inputs"].device, META)

    def test_parent_dtype_only_move_scores_without_child_to(self):
        """``parent.to(dtype=...)`` keeps CPU and does not call the child ``.to``.

        Inference follows the parameter dtype. The NumPy caller is not aliased.
        """
        clf, parent, calls = _ready()
        batch = np.arange(20, dtype=np.float32).reshape(5, 4).copy()
        pointer = batch.__array_interface__["data"][0]
        parent.to(dtype=torch.float64)
        placed = clf._features_on_parameter_device(batch)
        weight = next(clf.model.parameters())
        self.assertEqual(calls, [])
        self.assertEqual(weight.dtype, torch.float64)
        self.assertEqual(weight.device, CPU)
        self.assertEqual(clf.device, CPU)
        self.assertEqual(clf._parameter_device(), CPU)
        self.assertEqual(placed.dtype, torch.float64)
        self.assertEqual(placed.device, CPU)
        self.assertNotEqual(placed.data_ptr(), pointer)

        labels, probabilities = _reference(clf.model, batch)
        seen = _spy(clf.model)
        np.testing.assert_array_equal(clf.predict(batch), labels)
        scored = clf.predict_proba(batch)
        self.assertEqual(seen["inputs"].dtype, torch.float64)
        self.assertEqual(seen["inputs"].device, CPU)
        self.assertNotEqual(seen["inputs"].data_ptr(), pointer)
        self.assertEqual(batch.__array_interface__["data"][0], pointer)
        self.assertEqual(float(batch.ravel()[0]), 0.0)
        np.testing.assert_allclose(scored, probabilities)
        self.assertEqual(scored.dtype, np.float64)
        self.assertTrue(np.isfinite(scored).all())
        np.testing.assert_allclose(scored.sum(axis=1), np.ones(5), atol=1e-5)
        self.assertEqual(calls, [])

    def test_float64_tensor_keeps_identity(self):
        """A float64 tensor matching float64 weights keeps object identity."""
        clf, parent, calls = _ready()
        parent.to(dtype=torch.float64)
        batch = torch.arange(20, dtype=torch.float64).reshape(5, 4)
        pointer = batch.data_ptr()
        placed = clf._features_on_parameter_device(batch)
        self.assertIs(placed, batch)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(batch.dtype, torch.float64)
        self.assertEqual(calls, [])

        labels, probabilities = _reference(clf.model, batch)
        seen = _spy(clf.model)
        np.testing.assert_array_equal(clf.predict(batch), labels)
        scored = clf.predict_proba(batch)
        self.assertIs(seen["inputs"], batch)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(scored.dtype, np.float64)
        self.assertTrue(np.isfinite(scored).all())
        np.testing.assert_allclose(scored, probabilities)
        self.assertEqual(clf.device, CPU)
        self.assertEqual(clf._parameter_device(), CPU)

    def test_float64_parameters_survive_meta_apply(self):
        """float64 weights stay float64 on meta and do not rewrite the caller."""
        clf, parent, calls = _ready()
        parent.to(dtype=torch.float64)
        _apply_device(parent, META)
        batch = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        pointer = batch.data_ptr()
        placed = clf._features_on_parameter_device(batch)
        self.assertEqual(placed.dtype, torch.float64)
        self.assertEqual(placed.device, META)
        self.assertIsNot(placed, batch)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(batch.dtype, torch.float32)
        self.assertEqual(batch.device, CPU)

        seen = _spy(clf.model)
        for method_name in ("predict", "predict_proba"):
            with self.subTest(method=method_name):
                with self.assertRaises(NotImplementedError) as caught:
                    getattr(clf, method_name)(batch)
                self.assertIn("meta", str(caught.exception).lower())
                self.assertEqual(seen["inputs"].dtype, torch.float64)
                self.assertEqual(seen["inputs"].device, META)
                self.assertIsNot(seen["inputs"], batch)
        parameter = next(clf.model.parameters())
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(parameter.device, META)
        self.assertEqual(calls, [])
        self.assertEqual(clf.device, CPU)
        self.assertEqual(clf._parameter_device(), META)

    def test_cached_placement_mixes_meta_and_cpu(self):
        """Placing the batch on the construction cache reproduces the mix.

        Weights are on ``meta``. ``self.device`` is still CPU. The pre-fix
        placement raises ``RuntimeError`` and the message names both devices.
        """
        clf, parent, calls = _ready()
        batch = _batch()
        original = _snapshot(batch)
        _apply_device(parent, META)
        features = torch.tensor(
            np.asarray(batch), dtype=torch.float32, device=clf.device
        )

        with self.assertRaises(RuntimeError) as caught:
            clf.model(features)

        message = str(caught.exception).lower()
        self.assertIn("meta", message)
        self.assertIn("cpu", message)
        self.assertEqual(clf.device, CPU)
        self.assertEqual(clf._parameter_device(), META)
        self.assertEqual(calls, [])
        _assert_unchanged(self, batch, original)

    def test_opposite_cache_mixes_meta_batch_with_cpu_weights(self):
        """A cache pointed at ``meta`` must not place inference for a CPU model.

        Feeding that cache to the linear stack raises ``RuntimeError`` and
        names both devices. ``predict`` and ``predict_proba`` still follow the
        weights.
        """
        clf, _parent, _calls = _ready()
        batch = _batch()
        original = _snapshot(batch)
        labels, probabilities = _reference(clf.model, batch)
        clf.device = META
        features = torch.tensor(
            np.asarray(batch), dtype=torch.float32, device=clf.device
        )

        with self.assertRaises(RuntimeError) as caught:
            clf.model(features)

        message = str(caught.exception).lower()
        self.assertIn("meta", message)
        self.assertIn("cpu", message)
        np.testing.assert_array_equal(clf.predict(batch), labels)
        np.testing.assert_allclose(clf.predict_proba(batch), probabilities)
        self.assertEqual(next(clf.model.parameters()).device, CPU)
        self.assertEqual(clf.device, META)
        self.assertEqual(clf._parameter_device(), CPU)
        _assert_unchanged(self, batch, original)

    def test_restored_cache_helper_raises_device_mix(self):
        """Restoring construction-cache placement reproduces the meta/CPU error.

        The live helper does not. It feeds ``meta`` and then cannot copy the
        result back to the host.
        """
        clf, parent, calls = _ready()
        batch = _batch()
        original = _snapshot(batch)
        _apply_device(parent, META)

        def _on_cache(features):
            """Return the pre-fix batch, still on the construction cache."""
            return torch.tensor(
                np.asarray(features), dtype=torch.float32, device=clf.device
            )

        clf._features_on_parameter_device = _on_cache
        for method_name in ("predict", "predict_proba"):
            with self.subTest(method=method_name, placement="cache"):
                with self.assertRaises(RuntimeError) as caught:
                    getattr(clf, method_name)(batch)
                message = str(caught.exception).lower()
                self.assertIn("meta", message)
                self.assertIn("cpu", message)

        del clf._features_on_parameter_device
        seen = _spy(clf.model)
        for method_name in ("predict", "predict_proba"):
            with self.subTest(method=method_name, placement="parameters"):
                with self.assertRaises(NotImplementedError) as caught:
                    getattr(clf, method_name)(batch)
                self.assertIn("meta", str(caught.exception).lower())
                self.assertEqual(seen["inputs"].device, META)
                self.assertEqual(seen["inputs"].dtype, torch.float32)
        self.assertEqual(calls, [])
        self.assertEqual(clf.device, CPU)
        self.assertEqual(next(clf.model.parameters()).device, META)
        _assert_unchanged(self, batch, original)

    def test_parameter_device_falls_back_without_parameters(self):
        """With no network, the inference device is the construction preference.

        Real weights win over a cache pointed at the other device.
        """
        clf = _classifier()
        self.assertIsNone(clf.model)
        self.assertEqual(clf._parameter_device(), CPU)
        clf.device = META
        self.assertEqual(clf._parameter_device(), META)

        clf.model = nn.Sequential()
        clf.device = CPU
        self.assertIsNone(next(clf.model.parameters(), None))
        self.assertEqual(clf._parameter_device(), CPU)

        clf.model = clf._create_model()
        clf.device = META
        self.assertEqual(next(clf.model.parameters()).device, CPU)
        self.assertEqual(clf._parameter_device(), CPU)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMLPClassifierCuda(unittest.TestCase):
    """End-to-end coverage of a real accelerator move."""

    def test_parent_move_to_cuda_accepts_cpu_batch(self):
        """CPU arrays are scored on CUDA weights without calling ``model.to``."""
        clf, parent, calls = _ready()
        batch = _batch()
        original = _snapshot(batch)

        parent.to("cuda")

        self.assertEqual(calls, [])
        self.assertEqual(next(clf.model.parameters()).device.type, "cuda")
        self.assertEqual(clf.device, CPU)
        labels, probabilities = _reference(clf.model, batch)
        np.testing.assert_array_equal(clf.predict(batch), labels)
        np.testing.assert_allclose(clf.predict_proba(batch), probabilities)
        _assert_unchanged(self, batch, original)

    def test_stale_cpu_cache_still_scores_on_cuda(self):
        """Forcing the cache back to CPU must not pull CUDA weights to the host."""
        clf, parent, _calls = _ready()
        parent.to("cuda")
        clf.device = CPU
        batch = _batch()
        labels, probabilities = _reference(clf.model, batch)

        np.testing.assert_array_equal(clf.predict(batch), labels)
        np.testing.assert_allclose(clf.predict_proba(batch), probabilities)
        self.assertEqual(clf._parameter_device().type, "cuda")

    def test_parent_move_back_to_cpu_ignores_cuda_cache(self):
        """After the weights return to CPU, a CUDA cache must not win."""
        clf, parent, calls = _ready()
        parent.to("cuda")
        parent.to("cpu")
        clf.device = torch.device("cuda")
        batch = _batch()
        labels, probabilities = _reference(clf.model, batch)

        np.testing.assert_array_equal(clf.predict(batch), labels)
        np.testing.assert_allclose(clf.predict_proba(batch), probabilities)
        self.assertEqual(next(clf.model.parameters()).device, CPU)
        self.assertEqual(calls, [])

    def test_parent_apply_and_keyword_to_skip_child_to(self):
        """``_apply(cuda)`` and ``to(device='cuda')`` score NumPy without ``.to``."""
        cuda = torch.device("cuda")
        for how in ("apply", "keyword"):
            with self.subTest(how=how):
                clf, parent, calls = _ready()
                batch = _batch()
                original = _snapshot(batch)
                fresh = _cpu_clone(clf)
                labels, probabilities = _reference(fresh, batch)
                if how == "apply":
                    _apply_device(parent, cuda)
                else:
                    parent.to(device="cuda")

                self.assertEqual(calls, [])
                self.assertEqual(next(clf.model.parameters()).device.type, "cuda")
                self.assertEqual(clf.device, CPU)
                np.testing.assert_array_equal(clf.predict(batch), labels)
                np.testing.assert_allclose(
                    clf.predict_proba(batch), probabilities, rtol=1e-4, atol=1e-5
                )
                _assert_unchanged(self, batch, original)

    def test_cpu_tensor_keeps_storage_across_cuda_apply(self):
        """A CPU tensor keeps ``data_ptr`` while the linear stack sees CUDA."""
        clf, parent, calls = _ready()
        batch = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        pointer = batch.data_ptr()
        before = batch.clone()
        fresh = _cpu_clone(clf)
        _labels, probabilities = _reference(fresh, batch)
        _apply_device(parent, torch.device("cuda"))
        seen = _spy(clf.model)

        predicted = clf.predict(batch)
        scored = clf.predict_proba(batch)

        self.assertEqual(calls, [])
        self.assertIsNot(seen["inputs"], batch)
        self.assertEqual(seen["inputs"].device.type, "cuda")
        self.assertEqual(seen["inputs"].dtype, torch.float32)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(batch.device, CPU)
        self.assertTrue(torch.equal(batch, before))
        self.assertEqual(predicted.shape, (5,))
        np.testing.assert_allclose(scored, probabilities, rtol=1e-4, atol=1e-5)

    def test_batch_already_on_cuda_keeps_storage(self):
        """A CUDA batch that already matches the weights keeps its storage."""
        cuda = torch.device("cuda")
        for noncontiguous in (False, True):
            with self.subTest(noncontiguous=noncontiguous):
                clf, parent, calls = _ready()
                _apply_device(parent, cuda)
                if noncontiguous:
                    batch = torch.arange(40, device=cuda, dtype=torch.float32).reshape(
                        5, 8
                    )[:, :4]
                    self.assertFalse(batch.is_contiguous())
                else:
                    batch = torch.arange(20, device=cuda, dtype=torch.float32).reshape(
                        5, 4
                    )
                pointer = batch.data_ptr()
                placed = clf._features_on_parameter_device(batch)
                self.assertIs(placed, batch)
                self.assertEqual(batch.data_ptr(), pointer)
                seen = _spy(clf.model)

                scored = clf.predict_proba(batch)
                predicted = clf.predict(batch)

                self.assertIs(seen["inputs"], batch)
                self.assertEqual(calls, [])
                self.assertEqual(predicted.shape, (5,))
                self.assertEqual(scored.dtype, np.float32)
                self.assertTrue(np.isfinite(scored).all())
                if noncontiguous:
                    self.assertFalse(batch.is_contiguous())
                    self.assertFalse(seen["inputs"].is_contiguous())

    def test_cuda_batch_is_copied_onto_cpu_parameters(self):
        """A CUDA batch is copied onto CPU parameters. Caller storage stays put."""
        clf, _parent, _calls = _ready()
        batch = torch.arange(20, device="cuda", dtype=torch.float32).reshape(5, 4)
        pointer = batch.data_ptr()
        before = batch.detach().cpu().clone()
        labels, probabilities = _reference(clf.model, before)
        seen = _spy(clf.model)

        predicted = clf.predict(batch)
        scored = clf.predict_proba(batch)

        self.assertIsNot(seen["inputs"], batch)
        self.assertEqual(seen["inputs"].device, CPU)
        self.assertEqual(seen["inputs"].dtype, torch.float32)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(batch.device.type, "cuda")
        self.assertTrue(torch.equal(batch.detach().cpu(), before))
        np.testing.assert_array_equal(predicted, labels)
        np.testing.assert_allclose(scored, probabilities)

    def test_device_is_reread_across_cuda_and_cpu(self):
        """CUDA, then CPU, then CUDA follows parameters and does not call ``.to``."""
        cuda = torch.device("cuda")
        clf, parent, calls = _ready()
        batch = _batch()
        original = _snapshot(batch)
        fresh = _cpu_clone(clf)
        labels, probabilities = _reference(fresh, batch)

        _apply_device(parent, cuda)
        np.testing.assert_allclose(
            clf.predict_proba(batch), probabilities, rtol=1e-4, atol=1e-5
        )
        self.assertEqual(clf._parameter_device().type, "cuda")

        _apply_device(parent, CPU)
        np.testing.assert_array_equal(clf.predict(batch), labels)
        np.testing.assert_allclose(clf.predict_proba(batch), probabilities)
        self.assertEqual(clf._parameter_device(), CPU)
        self.assertEqual(clf.device, CPU)

        _apply_device(parent, cuda)
        np.testing.assert_allclose(
            clf.predict_proba(batch), probabilities, rtol=1e-4, atol=1e-5
        )
        self.assertEqual(calls, [])
        self.assertEqual(clf._parameter_device().type, "cuda")
        _assert_unchanged(self, batch, original)

    def test_float64_survives_cuda_apply(self):
        """float64 weights stay float64 on CUDA. The NumPy caller is not rewritten."""
        clf, parent, calls = _ready()
        parent.to(dtype=torch.float64)
        batch = _batch()
        pointer = batch.__array_interface__["data"][0]
        fresh = _cpu_clone(clf)
        _labels, probabilities = _reference(fresh, batch)
        _apply_device(parent, torch.device("cuda"))
        seen = _spy(clf.model)

        scored = clf.predict_proba(batch)
        predicted = clf.predict(batch)

        parameter = next(clf.model.parameters())
        self.assertEqual(calls, [])
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(parameter.device.type, "cuda")
        self.assertEqual(seen["inputs"].dtype, torch.float64)
        self.assertEqual(seen["inputs"].device.type, "cuda")
        self.assertEqual(batch.__array_interface__["data"][0], pointer)
        self.assertEqual(batch.dtype, np.float32)
        self.assertEqual(predicted.shape, (batch.shape[0],))
        self.assertEqual(scored.dtype, np.float64)
        np.testing.assert_allclose(scored, probabilities, rtol=1e-4, atol=1e-5)

    def test_dtype_only_after_cuda_stays_cuda(self):
        """``parent.to(dtype=...)`` after a CUDA move stays on CUDA.

        The child ``.to`` is not called. A matching float64 CUDA batch keeps
        identity.
        """
        clf, parent, calls = _ready()
        _apply_device(parent, torch.device("cuda"))
        parent.to(dtype=torch.float64)
        batch = torch.arange(20, device="cuda", dtype=torch.float64).reshape(5, 4)
        pointer = batch.data_ptr()
        placed = clf._features_on_parameter_device(batch)
        seen = _spy(clf.model)

        scored = clf.predict_proba(batch)

        parameter = next(clf.model.parameters())
        self.assertEqual(calls, [])
        self.assertEqual(parameter.device.type, "cuda")
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertIs(placed, batch)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertIs(seen["inputs"], batch)
        self.assertEqual(scored.dtype, np.float64)
        self.assertTrue(np.isfinite(scored).all())


if __name__ == "__main__":
    unittest.main()
