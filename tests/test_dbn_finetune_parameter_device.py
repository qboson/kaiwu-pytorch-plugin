"""Fine-tune batches must follow the network parameter device.

``SupervisedDBNClassification`` trains and scores ``fine_tune_network`` after
placing the batch on ``unsupervised_dbn.device``. That value is a plain
attribute on the DBN. A parent ``nn.Module._apply`` moves the fine-tune
parameters without calling ``fine_tune_network.to`` (merged #164 /
``tests/test_bm_device_sync.py``). The batch then stays on the cache:

``RuntimeError: Tensor on device meta is not on the expected device cpu!``

Class indices are the other tensor in that mix. ``CrossEntropyLoss`` rejects
a CPU target against meta logits with the same error, including when dropout
is on. ``int32`` indices are cast to ``long``. A ``long`` tensor that is
already on the parameter device keeps its storage.

The same mix happens when the cache is forced to ``meta`` and the network is
still on CPU: the cache must not choose the device. These tests stop at the
fine-tune network. They do not enter RBM ``get_hidden`` (#1334) or
``UnsupervisedDBN`` activation placement (#1349 / #242).

Out of scope: layer-weight initialisation (#1353), the classifier-mode
predict path (#1369), and ``get_layer_activations`` (#1367).
"""

# Parent moves go through ``_apply`` (see merged #164).
# pylint: disable=protected-access

from __future__ import annotations

import copy
import importlib
import os
import sys
import types
import unittest
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

# The example imports pyplot at module scope.
os.environ.setdefault("MPLBACKEND", "Agg")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "example", "dbn_digits"))


def _ensure_sampler_imports() -> None:
    """Allow the example to import when the Kaiwu SDK is not installed.

    ``dbn_trainer`` constructs a sampler at import time. The SDK is not on
    PyPI; a stand-in is enough because these tests never solve an Ising model.
    """
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
_INPUT_DIM = 4
_ROWS = 2


def _numpy_features(rows: int = _ROWS) -> np.ndarray:
    """Return a deterministic float32 feature matrix."""
    values = np.arange(rows * _INPUT_DIM, dtype=np.float32) / float(_INPUT_DIM)
    return values.reshape(rows, _INPUT_DIM)


def _labels(rows: int) -> np.ndarray:
    """Return alternating class ids of length ``rows``."""
    return np.arange(rows, dtype=np.int64) % 2


def _classifier() -> SupervisedDBNClassification:
    """Build one fitted-enough classifier with a real fine-tune network."""
    torch.manual_seed(0)
    model = SupervisedDBNClassification(
        hidden_layers_structure=[3],
        learning_rate_rbm=0.1,
        n_epochs_rbm=1,
        batch_size=4,
        verbose=False,
        plot_img=False,
        fine_tuning=True,
        n_iter_backprop=1,
        learning_rate=0.1,
        random_state=0,
    )
    model.unsupervised_dbn._dbn.create_rbm_layer(_INPUT_DIM)
    model.classes_ = np.array([0, 1])
    model._build_fine_tune_network()
    return model


def _nest(model: SupervisedDBNClassification):
    """Parent the fine-tune network and record direct ``.to`` calls."""
    parent = nn.Module()
    parent.network = model.fine_tune_network
    calls = []
    network = model.fine_tune_network
    original = network.to

    def _recorded_to(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    network.to = _recorded_to
    return parent, calls


def _move_parameters(module: nn.Module, device: torch.device) -> None:
    """Move parameters the way a parent does: ``_apply``, not ``.to``."""
    module._apply(lambda tensor: tensor.to(device))


def _retarget_storage(device: torch.device):
    """Return an ``_apply`` callback that places new storage on ``device``.

    ``Module.to`` cannot copy out of ``meta``. Replacing storage still goes
    through ``_apply``, so the parameter device changes without ``.to()``.
    Zeros are readable, which is what a later CPU score needs. The callback
    does not read the source tensor.
    """

    def _apply_fn(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == device:
            return tensor
        return torch.zeros(tensor.shape, dtype=tensor.dtype, device=device)

    return _apply_fn


def _assert_cannot_copy_out_of_meta(case: unittest.TestCase) -> None:
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


@contextmanager
def _capture_training_batch(model: SupervisedDBNClassification):
    """Record the feature batch and class indices the loss actually sees."""
    seen = {}
    network = model.fine_tune_network
    original_forward = network.forward
    original_loss = nn.CrossEntropyLoss.forward

    def _forward(batch):
        seen["features"] = batch
        return original_forward(batch)

    def _loss(self, logits, target):
        seen["logits"] = logits
        seen["labels"] = target
        return original_loss(self, logits, target)

    network.forward = _forward
    nn.CrossEntropyLoss.forward = _loss
    try:
        yield seen
    finally:
        network.forward = original_forward
        nn.CrossEntropyLoss.forward = original_loss


def _assert_meta_training_reads_indices(case, class_ids, dropout: bool) -> None:
    """One meta step must see long indices on the parameter device.

    The scalar ``Tensor.item`` error is past the device check. A CPU target
    would have raised the meta/CPU ``RuntimeError`` first.
    """
    trained = _classifier()
    if dropout:
        trained.dropout_p = 0.5
        trained._build_fine_tune_network()
    trained.n_iter_backprop = 1
    _parent, calls = _nest(trained)
    _move_parameters(_parent, _META)
    trained.unsupervised_dbn._dbn.device = _CPU
    class_pointer = None
    class_dtype = None
    if torch.is_tensor(class_ids):
        class_pointer = class_ids.data_ptr()
        class_dtype = class_ids.dtype

    with _capture_training_batch(trained) as seen:
        with case.assertRaises(RuntimeError) as caught:
            trained._train_fine_tune_network(_numpy_features(rows=8), class_ids)

    message = str(caught.exception).lower()
    case.assertIn("item", message)
    case.assertIn("meta", message)
    case.assertEqual(calls, [])
    case.assertEqual(seen["features"].device, _META)
    case.assertEqual(seen["features"].dtype, torch.float32)
    case.assertEqual(seen["labels"].device, _META)
    case.assertEqual(seen["labels"].dtype, torch.long)
    case.assertEqual(seen["logits"].device, _META)
    case.assertEqual(next(trained.fine_tune_network.parameters()).device, _META)
    case.assertEqual(trained.unsupervised_dbn.device, _CPU)
    if torch.is_tensor(class_ids):
        case.assertIsNot(seen["labels"], class_ids)
        case.assertEqual(class_ids.data_ptr(), class_pointer)
        case.assertEqual(class_ids.dtype, class_dtype)
        case.assertEqual(class_ids.device, _CPU)


def _cpu_logits(
    model: SupervisedDBNClassification, features: np.ndarray
) -> torch.Tensor:
    """Score ``features`` on a CPU weight copy, in eval mode."""
    cpu_network = copy.deepcopy(model.fine_tune_network).cpu().eval()
    with torch.no_grad():
        return cpu_network(torch.as_tensor(features, dtype=torch.float32))


class TestFineTuneParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine, including CPU-only CI."""

    def test_cpu_predictions_match_the_network(self) -> None:
        """An unmoved CPU network keeps the previous predict and proba results."""
        model = _classifier()
        features = _numpy_features()
        expected = _cpu_logits(model, features)
        predictions = model._predict_with_fine_tuning(features)
        probabilities = model._predict_proba_fine_tuning(features)

        np.testing.assert_array_equal(predictions, expected.argmax(dim=1).numpy())
        np.testing.assert_allclose(
            probabilities, torch.softmax(expected, dim=1).numpy()
        )
        self.assertEqual(predictions.shape, (_ROWS,))
        self.assertEqual(probabilities.shape, (_ROWS, 2))

    def test_numpy_features_are_not_mutated(self) -> None:
        """Scoring copies NumPy inputs instead of writing them."""
        model = _classifier()
        features = _numpy_features()
        original = features.copy()

        model._predict_with_fine_tuning(features)
        model._predict_proba_fine_tuning(features)

        np.testing.assert_array_equal(features, original)

    def test_stale_dbn_device_does_not_move_batches(self) -> None:
        """A meta DBN cache must not pull a CPU network off CPU."""
        model = _classifier()
        features = _numpy_features()
        expected = _cpu_logits(model, features)
        model.unsupervised_dbn._dbn.device = _META

        predictions = model._predict_with_fine_tuning(features)
        probabilities = model._predict_proba_fine_tuning(features)

        self.assertEqual(next(model.fine_tune_network.parameters()).device, _CPU)
        self.assertEqual(model.unsupervised_dbn.device, _META)
        np.testing.assert_array_equal(predictions, expected.argmax(dim=1).numpy())
        np.testing.assert_allclose(
            probabilities, torch.softmax(expected, dim=1).numpy()
        )

    def test_parent_apply_meta_reaches_the_network(self) -> None:
        """CPU features meet meta weights through predict and predict_proba.

        Copying the result out of meta is impossible. Before the fix the
        linear layer raises ``Tensor on device meta is not on the expected
        device cpu!`` and this assertion does not see ``NotImplementedError``.
        """
        model = _classifier()
        parent, calls = _nest(model)
        features = _numpy_features()
        original = features.copy()

        _move_parameters(parent, _META)

        self.assertEqual(calls, [])
        self.assertEqual(next(model.fine_tune_network.parameters()).device, _META)
        self.assertEqual(model.unsupervised_dbn.device, _CPU)
        for method in (
            model._predict_with_fine_tuning,
            model._predict_proba_fine_tuning,
        ):
            with self.assertRaises(NotImplementedError) as caught:
                method(features)
            self.assertIn("meta", str(caught.exception).lower())
        np.testing.assert_array_equal(features, original)

    def test_dtype_only_parent_move_keeps_cpu(self) -> None:
        """``parent.to(dtype=...)`` does not call the child ``.to``."""
        model = _classifier()
        parent, calls = _nest(model)

        parent.to(dtype=torch.float64)

        parameter = next(model.fine_tune_network.parameters())
        self.assertEqual(calls, [])
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(parameter.device, _CPU)
        aligned = model._align_fine_tune_features(_numpy_features())
        self.assertEqual(aligned.dtype, torch.float64)
        self.assertEqual(aligned.device, _CPU)
        self.assertTrue(torch.isfinite(model.fine_tune_network(aligned)).all())
        labels = torch.tensor([0, 1])
        pointer = labels.data_ptr()
        aligned_labels = model._align_fine_tune_labels(labels, aligned.device)
        self.assertIs(aligned_labels, labels)
        self.assertEqual(labels.data_ptr(), pointer)
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(labels.device, _CPU)

    def test_direct_dtype_move_keeps_cpu(self) -> None:
        """A direct dtype conversion stays on CPU and scores finite logits."""
        model = _classifier()
        features = _numpy_features()
        original = features.copy()

        model.fine_tune_network.to(dtype=torch.float64)

        parameter = next(model.fine_tune_network.parameters())
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(parameter.device, _CPU)
        aligned = model._align_fine_tune_features(features)
        self.assertEqual(aligned.dtype, torch.float64)
        self.assertEqual(aligned.device, _CPU)
        self.assertTrue(torch.isfinite(model.fine_tune_network(aligned)).all())
        np.testing.assert_array_equal(features, original)

    def test_parameter_device_is_reread_after_round_trip(self) -> None:
        """The device is reread after meta and after ``_apply`` returns to CPU.

        ``parent.to(cpu)`` raises ``NotImplementedError`` and leaves the
        parameters on meta. The return trip retargets storage with zeros.
        """
        _assert_cannot_copy_out_of_meta(self)
        model = _classifier()
        parent, calls = _nest(model)
        features = _numpy_features()

        _move_parameters(parent, _META)
        meta_features = model._align_fine_tune_features(features)
        self.assertEqual(calls, [])
        self.assertEqual(meta_features.device, _META)
        self.assertEqual(model.fine_tune_network(meta_features).device, _META)

        with self.assertRaises(NotImplementedError) as caught:
            parent.to(_CPU)
        self.assertIn("meta", str(caught.exception).lower())
        self.assertEqual(calls, [])
        for parameter in model.fine_tune_network.parameters():
            self.assertEqual(parameter.device, _META)
        self.assertEqual(model._align_fine_tune_features(features).device, _META)

        parent._apply(_retarget_storage(_CPU))
        cpu_features = model._align_fine_tune_features(features)
        self.assertEqual(calls, [])
        self.assertEqual(cpu_features.device, _CPU)
        self.assertEqual(cpu_features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(model.fine_tune_network(cpu_features)).all())

        parent._apply(_retarget_storage(_META))
        self.assertEqual(model._align_fine_tune_features(features).device, _META)
        self.assertEqual(calls, [])

    def test_same_device_features_keep_identity_and_grad(self) -> None:
        """A CPU batch that is already on the parameter device is not copied."""
        model = _classifier()
        features = torch.randn(_ROWS, _INPUT_DIM, requires_grad=True)
        pointer = features.data_ptr()

        aligned = model._align_fine_tune_features(features)

        self.assertIs(aligned, features)
        self.assertEqual(features.data_ptr(), pointer)
        model.fine_tune_network(aligned).sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertEqual(features.grad.device, _CPU)
        weight = next(model.fine_tune_network.parameters())
        self.assertIsNotNone(weight.grad)
        self.assertEqual(weight.grad.device, _CPU)

    def test_noncontiguous_cpu_features_keep_storage(self) -> None:
        """Same-device alignment keeps a noncontiguous view's storage."""
        model = _classifier()
        base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        features = base.t()
        self.assertFalse(features.is_contiguous())
        pointer = features.data_ptr()

        aligned = model._align_fine_tune_features(features)

        self.assertIs(aligned, features)
        self.assertEqual(aligned.data_ptr(), pointer)
        self.assertFalse(aligned.is_contiguous())

    def test_parameter_free_network_follows_the_feature_device(self) -> None:
        """An empty network follows a tensor input, not a stale DBN cache."""
        model = _classifier()
        model.fine_tune_network = nn.Sequential()
        model.unsupervised_dbn._dbn.device = _META
        features = torch.randn(_ROWS, _INPUT_DIM)

        aligned = model._align_fine_tune_features(features)

        self.assertIs(aligned, features)
        self.assertEqual(aligned.device, _CPU)

        model.unsupervised_dbn._dbn.device = _CPU
        meta_features = torch.empty(_ROWS, _INPUT_DIM, device=_META)
        meta_pointer = meta_features.data_ptr()
        aligned_meta = model._align_fine_tune_features(meta_features)
        self.assertIs(aligned_meta, meta_features)
        self.assertEqual(aligned_meta.data_ptr(), meta_pointer)
        self.assertEqual(aligned_meta.device, _META)

    def test_parameter_free_numpy_can_use_the_cache(self) -> None:
        """With no parameters and no tensor device, the DBN cache is the fallback."""
        model = _classifier()
        model.fine_tune_network = nn.Sequential()
        model.unsupervised_dbn._dbn.device = _META

        aligned = model._align_fine_tune_features(_numpy_features())
        labels = model._align_fine_tune_labels(_labels(_ROWS), aligned.device)

        self.assertEqual(aligned.device, _META)
        self.assertEqual(aligned.dtype, torch.float32)
        self.assertEqual(labels.device, _META)
        self.assertEqual(labels.dtype, torch.long)

    def test_fine_tuning_ignores_stale_meta_device(self) -> None:
        """One training step stays on the new CPU network when the cache is meta."""
        model = _classifier()
        model.unsupervised_dbn._dbn.device = _META
        model.n_iter_backprop = 1

        model._fine_tuning(_numpy_features(rows=8), _labels(8))

        parameter = next(model.fine_tune_network.parameters())
        self.assertEqual(parameter.device, _CPU)
        self.assertEqual(model.unsupervised_dbn.device, _META)
        self.assertTrue(torch.isfinite(parameter).all())

    def test_parent_to_meta_skips_child_to(self) -> None:
        """Positional and keyword ``Module.to(meta)`` do not call child ``.to``."""
        for kwargs in ({}, {"device": _META}):
            with self.subTest(kwargs=bool(kwargs)):
                model = _classifier()
                parent, calls = _nest(model)
                if kwargs:
                    parent.to(**kwargs)
                else:
                    parent.to(_META)

                self.assertEqual(calls, [])
                for parameter in model.fine_tune_network.parameters():
                    self.assertEqual(parameter.device, _META)
                self.assertEqual(model.unsupervised_dbn.device, _CPU)

    def test_cpu_caller_storage_survives_meta_apply(self) -> None:
        """A CPU batch stays put while the linear layer observes a meta copy."""
        model = _classifier()
        parent, calls = _nest(model)
        features = torch.randn(_ROWS, _INPUT_DIM)
        before = features.clone()
        pointer = features.data_ptr()
        _move_parameters(parent, _META)

        with self.assertRaises(NotImplementedError) as caught:
            model._predict_with_fine_tuning(features)
        self.assertIn("meta", str(caught.exception).lower())

        aligned = model._align_fine_tune_features(features)
        self.assertEqual(calls, [])
        self.assertEqual(features.data_ptr(), pointer)
        self.assertEqual(features.device, _CPU)
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.equal(features, before))
        self.assertIsNot(aligned, features)
        self.assertEqual(aligned.device, _META)
        self.assertEqual(aligned.dtype, torch.float32)
        self.assertEqual(model.fine_tune_network(aligned).device, _META)

        base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        view = base.t()
        view_pointer = view.data_ptr()
        self.assertFalse(view.is_contiguous())
        moved = model._align_fine_tune_features(view)
        self.assertEqual(view.data_ptr(), view_pointer)
        self.assertFalse(view.is_contiguous())
        self.assertEqual(view.device, _CPU)
        self.assertIsNot(moved, view)
        self.assertEqual(moved.device, _META)

    def test_class_indices_follow_parameter_device(self) -> None:
        """Class indices follow the live weights, including dropout and int32.

        Meta logits reject a CPU target. A same-device ``long`` tensor keeps
        its storage. ``int32`` is copied to ``long``. Training reaches
        ``Tensor.item`` on the meta scalar, which is past the device check.
        """
        model = _classifier()
        labels = torch.tensor(_labels(_ROWS))
        pointer = labels.data_ptr()
        view = torch.arange(8).reshape(4, 2)[:, 0]
        self.assertFalse(view.is_contiguous())
        view_pointer = view.data_ptr()

        aligned = model._align_fine_tune_labels(labels, _CPU)
        aligned_view = model._align_fine_tune_labels(view, _CPU)

        self.assertIs(aligned, labels)
        self.assertEqual(labels.data_ptr(), pointer)
        self.assertEqual(labels.dtype, torch.long)
        self.assertIs(aligned_view, view)
        self.assertEqual(view.data_ptr(), view_pointer)
        self.assertFalse(view.is_contiguous())

        model.n_iter_backprop = 1
        narrow = torch.tensor(_labels(8), dtype=torch.int32)
        narrow_pointer = narrow.data_ptr()
        cast = model._align_fine_tune_labels(narrow, _CPU)
        self.assertIsNot(cast, narrow)
        self.assertEqual(cast.dtype, torch.long)
        self.assertEqual(cast.device, _CPU)
        self.assertEqual(narrow.dtype, torch.int32)
        self.assertEqual(narrow.data_ptr(), narrow_pointer)
        model._train_fine_tune_network(_numpy_features(rows=8), narrow)
        self.assertEqual(narrow.dtype, torch.int32)
        self.assertEqual(narrow.data_ptr(), narrow_pointer)
        self.assertEqual(next(model.fine_tune_network.parameters()).device, _CPU)
        self.assertTrue(
            torch.isfinite(next(model.fine_tune_network.parameters())).all()
        )

        cases = (
            ("long", torch.tensor(_labels(8)), False),
            ("int32", torch.tensor(_labels(8), dtype=torch.int32), False),
            ("numpy", _labels(8), False),
            ("dropout", torch.tensor(_labels(8)), True),
        )
        for name, class_ids, dropout in cases:
            with self.subTest(labels=name):
                _assert_meta_training_reads_indices(self, class_ids, dropout)

    def test_meta_batch_already_aligned_keeps_storage(self) -> None:
        """A meta batch that already matches the weights keeps its storage."""
        model = _classifier()
        _move_parameters(model.fine_tune_network, _META)
        features = torch.empty(_ROWS, _INPUT_DIM, device=_META)
        labels = torch.empty(_ROWS, dtype=torch.long, device=_META)
        feature_pointer = features.data_ptr()
        label_pointer = labels.data_ptr()

        aligned = model._align_fine_tune_features(features)
        aligned_labels = model._align_fine_tune_labels(labels, aligned.device)

        self.assertIs(aligned, features)
        self.assertIs(aligned_labels, labels)
        self.assertEqual(features.data_ptr(), feature_pointer)
        self.assertEqual(labels.data_ptr(), label_pointer)
        self.assertEqual(model.fine_tune_network(aligned).device, _META)
        self.assertEqual(model.fine_tune_network(aligned).dtype, torch.float32)

    def test_float64_parameters_survive_meta_apply(self) -> None:
        """float64 weights stay float64 on meta and do not rewrite callers."""
        model = _classifier()
        parent, calls = _nest(model)
        parent.to(dtype=torch.float64)
        _move_parameters(parent, _META)
        features = torch.randn(_ROWS, _INPUT_DIM)
        labels = torch.tensor(_labels(_ROWS))
        feature_pointer = features.data_ptr()
        label_pointer = labels.data_ptr()

        aligned = model._align_fine_tune_features(features)
        aligned_labels = model._align_fine_tune_labels(labels, aligned.device)
        logits = model.fine_tune_network(aligned)

        self.assertEqual(calls, [])
        for parameter in model.fine_tune_network.parameters():
            self.assertEqual(parameter.dtype, torch.float64)
            self.assertEqual(parameter.device, _META)
        self.assertEqual(features.data_ptr(), feature_pointer)
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(features.device, _CPU)
        self.assertIsNot(aligned, features)
        self.assertEqual(aligned.dtype, torch.float64)
        self.assertEqual(aligned.device, _META)
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(logits.device, _META)
        self.assertEqual(labels.data_ptr(), label_pointer)
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(labels.device, _CPU)
        self.assertIsNot(aligned_labels, labels)
        self.assertEqual(aligned_labels.dtype, torch.long)
        self.assertEqual(aligned_labels.device, _META)

    def test_stale_cache_placement_raises_device_mix(self) -> None:
        """Restoring cache placement reproduces the meta/CPU ``RuntimeError``."""
        model = _classifier()
        _move_parameters(_nest(model)[0], _META)
        self.assertEqual(model.unsupervised_dbn.device, _CPU)
        original = model._align_fine_tune_features

        def _on_cached_device(features):
            cached = model.unsupervised_dbn.device
            if torch.is_tensor(features):
                return features.to(device=cached)
            return torch.tensor(features, dtype=torch.float32, device=cached)

        model._align_fine_tune_features = _on_cached_device
        try:
            with self.assertRaises(RuntimeError) as caught:
                model._predict_with_fine_tuning(_numpy_features())
        finally:
            model._align_fine_tune_features = original
        message = str(caught.exception).lower()
        self.assertIn("meta", message)
        self.assertIn("cpu", message)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFineTuneParameterDeviceCuda(unittest.TestCase):
    """End-to-end coverage of a real second device. Skipped on CPU-only CI."""

    def test_parent_apply_cuda_matches_cpu_weights(self) -> None:
        """A parent CUDA move accepts CPU features and matches a CPU copy."""
        model = _classifier()
        features = _numpy_features()
        expected = _cpu_logits(model, features)
        parent, calls = _nest(model)

        _move_parameters(parent, torch.device("cuda"))
        predictions = model._predict_with_fine_tuning(features)
        probabilities = model._predict_proba_fine_tuning(features)

        self.assertEqual(calls, [])
        self.assertEqual(next(model.fine_tune_network.parameters()).device.type, "cuda")
        np.testing.assert_array_equal(predictions, expected.argmax(dim=1).numpy())
        np.testing.assert_allclose(
            probabilities,
            torch.softmax(expected, dim=1).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_stale_cpu_cache_does_not_pull_cuda_network(self) -> None:
        """Forcing the DBN cache back to CPU leaves CUDA parameters in charge."""
        model = _classifier()
        features = torch.randn(_ROWS, _INPUT_DIM)
        pointer = features.data_ptr()
        parent, calls = _nest(model)
        _move_parameters(parent, torch.device("cuda"))
        model.unsupervised_dbn._dbn.device = _CPU

        aligned = model._align_fine_tune_features(features)

        self.assertEqual(calls, [])
        self.assertEqual(aligned.device.type, "cuda")
        self.assertEqual(features.device, _CPU)
        self.assertEqual(features.data_ptr(), pointer)
        self.assertEqual(model.fine_tune_network(aligned).device.type, "cuda")

    def test_device_is_reread_across_cuda_and_cpu(self) -> None:
        """CUDA, CPU, then CUDA are each reread. ``.to`` is not called."""
        model = _classifier()
        parent, calls = _nest(model)
        features = _numpy_features()

        _move_parameters(parent, torch.device("cuda"))
        self.assertEqual(model._align_fine_tune_features(features).device.type, "cuda")
        _move_parameters(parent, _CPU)
        cpu_features = model._align_fine_tune_features(features)
        self.assertEqual(cpu_features.device, _CPU)
        self.assertTrue(torch.isfinite(model.fine_tune_network(cpu_features)).all())
        _move_parameters(parent, torch.device("cuda"))
        self.assertEqual(model._align_fine_tune_features(features).device.type, "cuda")
        self.assertEqual(calls, [])

    def test_float64_survives_cuda_move(self) -> None:
        """float64 parameters stay float64 after a parent CUDA ``_apply``."""
        model = _classifier()
        parent, calls = _nest(model)
        model.fine_tune_network.to(dtype=torch.float64)

        _move_parameters(parent, torch.device("cuda"))
        aligned = model._align_fine_tune_features(_numpy_features())

        self.assertEqual(calls, [])
        self.assertEqual(aligned.dtype, torch.float64)
        self.assertEqual(aligned.device.type, "cuda")
        logits = model.fine_tune_network(aligned)
        self.assertEqual(logits.dtype, torch.float64)
        self.assertTrue(torch.isfinite(logits).all())

    def test_backward_from_cpu_batch_reaches_cuda_weights(self) -> None:
        """A CPU batch backpropagates into CUDA weights and a CPU input grad."""
        model = _classifier()
        parent, _calls = _nest(model)
        _move_parameters(parent, torch.device("cuda"))
        features = torch.randn(_ROWS, _INPUT_DIM, requires_grad=True)
        pointer = features.data_ptr()

        aligned = model._align_fine_tune_features(features)
        model.fine_tune_network(aligned).sum().backward()

        self.assertEqual(features.data_ptr(), pointer)
        self.assertEqual(features.grad.device, _CPU)
        weight = next(model.fine_tune_network.parameters())
        self.assertEqual(weight.grad.device.type, "cuda")

    def test_cuda_features_follow_cpu_parameters(self) -> None:
        """A CUDA batch is copied onto a CPU network. Caller storage stays put."""
        model = _classifier()
        features = torch.randn(_ROWS, _INPUT_DIM, device=torch.device("cuda"))
        pointer = features.data_ptr()

        aligned = model._align_fine_tune_features(features)

        self.assertEqual(aligned.device, _CPU)
        self.assertEqual(features.device.type, "cuda")
        self.assertEqual(features.data_ptr(), pointer)
        self.assertEqual(model.fine_tune_network(aligned).device, _CPU)

    def test_parent_to_cuda_skips_child_to(self) -> None:
        """``parent.to('cuda')`` scores CPU features without calling child ``.to``."""
        model = _classifier()
        features = _numpy_features()
        expected = _cpu_logits(model, features)
        parent, calls = _nest(model)

        parent.to("cuda")
        predictions = model._predict_with_fine_tuning(features)
        probabilities = model._predict_proba_fine_tuning(features)

        self.assertEqual(calls, [])
        for parameter in model.fine_tune_network.parameters():
            self.assertEqual(parameter.device.type, "cuda")
        np.testing.assert_array_equal(predictions, expected.argmax(dim=1).numpy())
        np.testing.assert_allclose(
            probabilities,
            torch.softmax(expected, dim=1).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_batch_already_on_cuda_keeps_storage(self) -> None:
        """A CUDA batch that already matches the weights keeps its storage."""
        model = _classifier()
        cuda = torch.device("cuda")
        parent, calls = _nest(model)
        _move_parameters(parent, cuda)
        features = torch.randn(_ROWS, _INPUT_DIM, device=cuda)
        labels = torch.tensor(_labels(_ROWS), device=cuda)
        feature_pointer = features.data_ptr()
        label_pointer = labels.data_ptr()

        aligned = model._align_fine_tune_features(features)
        aligned_labels = model._align_fine_tune_labels(labels, aligned.device)
        logits = model.fine_tune_network(aligned)

        self.assertEqual(calls, [])
        self.assertIs(aligned, features)
        self.assertIs(aligned_labels, labels)
        self.assertEqual(features.data_ptr(), feature_pointer)
        self.assertEqual(labels.data_ptr(), label_pointer)
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(logits.device.type, "cuda")
        self.assertTrue(torch.isfinite(logits).all())

    def test_dtype_only_move_keeps_cuda(self) -> None:
        """``parent.to(dtype=...)`` after a CUDA move stays on CUDA."""
        model = _classifier()
        parent, calls = _nest(model)
        _move_parameters(parent, torch.device("cuda"))

        parent.to(dtype=torch.float64)
        aligned = model._align_fine_tune_features(_numpy_features())
        labels = torch.tensor(_labels(_ROWS))
        pointer = labels.data_ptr()
        aligned_labels = model._align_fine_tune_labels(labels, aligned.device)
        logits = model.fine_tune_network(aligned)

        self.assertEqual(calls, [])
        for parameter in model.fine_tune_network.parameters():
            self.assertEqual(parameter.dtype, torch.float64)
            self.assertEqual(parameter.device.type, "cuda")
        self.assertEqual(aligned.dtype, torch.float64)
        self.assertEqual(aligned.device.type, "cuda")
        self.assertEqual(logits.dtype, torch.float64)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(aligned_labels.dtype, torch.long)
        self.assertEqual(aligned_labels.device.type, "cuda")
        self.assertEqual(labels.data_ptr(), pointer)
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(labels.device, _CPU)

    def test_training_step_puts_cpu_labels_on_cuda(self) -> None:
        """One step reads CPU int32 indices on the CUDA parameter device."""
        torch.manual_seed(0)
        model = _classifier()
        model.n_iter_backprop = 1
        parent, calls = _nest(model)
        _move_parameters(parent, torch.device("cuda"))
        model.unsupervised_dbn._dbn.device = _CPU
        features = torch.randn(8, _INPUT_DIM)
        labels = torch.tensor(_labels(8), dtype=torch.int32)
        feature_pointer = features.data_ptr()
        label_pointer = labels.data_ptr()

        with _capture_training_batch(model) as seen:
            model._train_fine_tune_network(features, labels)

        weight = next(model.fine_tune_network.parameters())
        self.assertEqual(calls, [])
        self.assertEqual(seen["features"].device.type, "cuda")
        self.assertEqual(seen["features"].dtype, torch.float32)
        self.assertEqual(seen["labels"].device.type, "cuda")
        self.assertEqual(seen["labels"].dtype, torch.long)
        self.assertIsNot(seen["labels"], labels)
        self.assertEqual(seen["logits"].device.type, "cuda")
        self.assertEqual(features.data_ptr(), feature_pointer)
        self.assertEqual(features.device, _CPU)
        self.assertEqual(labels.data_ptr(), label_pointer)
        self.assertEqual(labels.dtype, torch.int32)
        self.assertEqual(labels.device, _CPU)
        self.assertEqual(weight.device.type, "cuda")
        self.assertTrue(torch.isfinite(weight).all())
        self.assertIsNotNone(weight.grad)
        self.assertEqual(weight.grad.device.type, "cuda")
        self.assertTrue(torch.isfinite(weight.grad).all())
        self.assertGreater(float(weight.grad.abs().sum()), 0.0)
        self.assertEqual(model.unsupervised_dbn.device, _CPU)


if __name__ == "__main__":
    unittest.main()
