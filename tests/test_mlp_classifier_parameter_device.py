"""Regression tests for ``MLPClassifier`` device alignment.

``MLPClassifier`` is a scikit-learn estimator, not an ``nn.Module``. It caches
``device`` at construction and sends ``predict`` / ``predict_proba`` batches
there. The network itself is an ``nn.Sequential``. Nesting that module in a
parent and moving the parent through ``_apply`` (merged #164) relocates the
linear weights without updating the estimator attribute. The CPU batch then
meets parameters on ``meta`` or CUDA:

``RuntimeError: Tensor on device meta is not on the expected device cpu!``

Inference rereads the first parameter device. ``self.device`` stays the
construction preference used by ``fit`` and ``get_params``, so a later ``fit``
does not follow a module that has already been moved. Dtypes stay float32.
The caller array is not written. CPU cases use ``meta``. CUDA cases are
skipped unless a GPU is present.

``SimpleEncoder`` (#1374) and ``RBMRunner`` (#1400) are different types.
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


def _reference(model: nn.Module, batch: np.ndarray):
    """Labels and probabilities from the live weights, in eval mode."""
    device = next(model.parameters()).device
    features = torch.tensor(np.asarray(batch), dtype=torch.float32, device=device)
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
        """A parent move relocates weights without calling the child ``.to()``."""
        clf, parent, calls = _ready()

        parent.to(META)

        self.assertEqual(calls, [])
        self.assertEqual(next(clf.model.parameters()).device, META)
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
        """``.to(dtype=...)`` changes the weight dtype and leaves the device."""
        clf, _parent, calls = _ready()

        clf.model.to(dtype=torch.float64)

        weight = next(clf.model.parameters())
        self.assertEqual(weight.dtype, torch.float64)
        self.assertEqual(weight.device, CPU)
        self.assertEqual(clf.device, CPU)
        self.assertEqual(clf._parameter_device(), CPU)
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
        """A move back onto CPU is visible even if the cache still says ``meta``."""
        clf, parent, calls = _ready()
        batch = _batch()
        _apply_device(parent, META)
        self.assertEqual(clf._parameter_device(), META)

        # ``.to(cpu)`` cannot copy out of meta; ``_apply`` still retargets.
        parent._apply(_retarget_storage(CPU))
        clf.device = META
        labels, probabilities = _reference(clf.model, batch)

        np.testing.assert_array_equal(clf.predict(batch), labels)
        np.testing.assert_allclose(clf.predict_proba(batch), probabilities)
        self.assertEqual(clf._parameter_device(), CPU)
        self.assertEqual(clf.device, META)
        self.assertEqual(calls, [])


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


if __name__ == "__main__":
    unittest.main()
