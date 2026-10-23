"""Regression tests for the cell-example Trainer device cache.

``Trainer`` stores the constructor device and later used to send every batch
through ``tensor.to(self.device)``. A parent ``nn.Module.to`` / ``_apply`` moves
the model parameters without calling the child ``.to`` (merged #164 /
``tests/test_bm_device_sync.py``) and without updating that cache. The next
``run_epoch``, ``extract_representation`` or ``compute_energy`` then feeds a
CPU batch to weights that already left CPU:

``RuntimeError: Tensor on device meta is not on the expected device cpu!``

The opposite mix, a cache pointed at ``meta`` while the weights stay on CPU,
names both devices the other way around. ``Module.to(cpu)`` cannot copy out of
``meta``; the return trip retargets storage with ``_apply`` and ``zeros_like``.

Placement follows the live parameter device and is reread on every batch.
The constructor cache is not consulted once the model has parameters. A
parameter-free model keeps the caller batch's device. Callers are not updated
in place. Same-device tensors keep identity, including a noncontiguous view.
Meta identity is the Python object: every meta tensor reports ``data_ptr() == 0``.
Dtypes stay put. A dtype-only parent move does not leave the current device.

The probe module is intentional. ``CellQVAE.forward`` draws an ``Exponential``
sample whose beta tensor is a separate open defect; these tests stop at the
trainer's batch placement, which is what the three entry points actually
control. The probe's linear layer is a real parameter matmul, so a stale
device still raises the runtime error above.
"""

# Parent moves use ``_apply``. The cache helper is ``_parameter_device``.
# pylint: disable=protected-access

import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn

_ENTRIES = ("run_epoch", "extract_representation", "compute_energy")
CPU = torch.device("cpu")
META = torch.device("meta")
CUDA = torch.device("cuda")
F32 = torch.float32
F64 = torch.float64


def _install_sdk_stubs(kaiwu):
    """Stub Kaiwu SDK modules that ``trainer.py`` imports but never calls here."""
    stubs = {
        "kaiwu.classical": {"SimulatedAnnealingOptimizer": object},
        "kaiwu.cim": {"CIMOptimizer": object},
        "kaiwu.preprocess": {"PrecisionReducer": object},
    }
    for name, attrs in stubs.items():
        leaf = name.rsplit(".", maxsplit=1)[-1]
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        for key, value in attrs.items():
            setattr(module, key, value)
        setattr(kaiwu, leaf, module)


def _load_trainer():
    """Load the example trainer without a static import pylint cannot resolve."""
    root = Path(__file__).resolve().parents[1]
    for path in (root / "src", root / "example" / "qvae_cell"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    kaiwu = importlib.import_module("kaiwu")
    _install_sdk_stubs(kaiwu)
    spec = importlib.util.spec_from_file_location(
        "cell_trainer_under_test",
        root / "example" / "qvae_cell" / "trainer.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader = spec.loader
    if loader is None:
        raise ImportError("unable to load example/qvae_cell/trainer.py")
    loader.exec_module(module)
    return module


CELL = _load_trainer()


class _RecordingModel(nn.Module):  # pylint: disable=abstract-method
    """Shared bookkeeping for trainer entry points."""

    def __init__(self):
        super().__init__()
        self.direct_to_calls = []
        self.seen_inputs = []
        self.last_kl_loss = torch.zeros(())
        self.last_recon_loss = torch.zeros(())

    def to(self, *args, **kwargs):
        """Record a direct ``.to`` call, which parent ``_apply`` must not make."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)

    def loss(self, x, output, posterior):
        """Return a concrete scalar. Meta activations cannot be read with ``.item``."""
        del x, output, posterior
        return torch.zeros(())

    def bm_loss(self, q, weight_decay):
        """Return a differentiable scalar so the BM step does not raise."""
        del q, weight_decay
        return torch.zeros((), requires_grad=True)


class ProbeModel(_RecordingModel):
    """Linear probe whose matmul crashes when the batch device is stale."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3)
        self.connect_loss = False

    def forward(self, x, batch_idx=None):
        """Matmul on the live parameter device, then hand numpy a CPU placeholder."""
        activated = self.proj(x)
        self.seen_inputs.append((x, batch_idx, activated))
        placeholder = torch.zeros(x.shape[0], 3)
        return placeholder, None, placeholder, placeholder

    def loss(self, x, output, posterior):
        """Optionally keep the caller batch in the autograd graph."""
        del output, posterior
        if self.connect_loss:
            return self.proj(x).sum()
        return torch.zeros(())

    def energy(self, x, loss_type):
        """Score on the parameter device and return a concrete CPU vector."""
        del loss_type
        activated = self.proj(x)
        self.seen_inputs.append((x, None, activated))
        return torch.zeros(x.shape[0])


class ParameterFreeModel(_RecordingModel):
    """Model with no parameters. Placement must follow the caller batch."""

    def forward(self, x, batch_idx=None):
        """Record the batch without moving it."""
        self.seen_inputs.append((x, batch_idx, x))
        placeholder = torch.zeros(x.shape[0], 1)
        return placeholder, None, placeholder, placeholder

    def energy(self, x, loss_type):
        """Record the energy batch without moving it."""
        del loss_type
        self.seen_inputs.append((x, None, x))
        return torch.zeros(x.shape[0])


class _MoveDuringIteration:
    """Yield one batch, retarget ``parent`` through ``_apply``, yield it again."""

    def __init__(self, parent, values, index, device):
        self.parent = parent
        self.values = values
        self.index = index
        self.device = device

    def __iter__(self):
        """Yield, move parameters, then yield the same caller tensors."""
        yield self.values, self.index
        _apply_device(self.parent, self.device)
        yield self.values, self.index

    def __len__(self):
        """``run_epoch`` divides totals by the loader length."""
        return 2


def _trainer(device="cpu"):
    """Build a trainer whose cached device is independent of later moves."""
    args = types.SimpleNamespace(
        representation="q",
        loss_type="mse",
        bm_weight_decay=0.0,
    )
    return CELL.Trainer(args, torch.device(device))


def _batch(
    rows=2,
    features=4,
    dtype=torch.float32,
    device="cpu",
    noncontiguous=False,
    requires_grad=False,
):
    """Return one ``(features, batch index)`` pair.

    The view is sliced on ``device``. ``Tensor.to(meta)`` would materialize a
    contiguous tensor and make a same-device identity check vacuous.
    """
    destination = torch.device(device)
    count = rows * features
    if noncontiguous:
        base = torch.arange(count * 2, dtype=dtype, device=destination)
        values = base.reshape(rows, features * 2)[:, ::2]
    else:
        values = torch.arange(count, dtype=dtype, device=destination).reshape(
            rows, features
        )
    if requires_grad:
        values.requires_grad_(True)
    index = torch.arange(rows, dtype=torch.long, device=destination)
    return values, index


def _apply_device(module, device):
    """Move parameters the way a parent module does.

    Leaving ``meta`` cannot copy values out, so that direction allocates fresh
    zeros with ``zeros_like`` (the source values are not read). Every other hop
    uses ``Tensor.to``.
    """
    destination = torch.device(device)

    def _move(tensor):
        if tensor.device.type == "meta" and destination.type != "meta":
            return torch.zeros_like(tensor, device=destination)
        return tensor.to(destination)

    module._apply(_move)


def _exercise(trainer, model, values, index):
    """Run the three entry points that place a batch."""
    model.seen_inputs.clear()
    loader = [(values, index)]
    metrics = trainer.run_epoch(model, loader, None, None, False)
    representations = trainer.extract_representation(model, loader)
    energies = trainer.compute_energy(model, loader)
    return metrics, representations, energies


def _invoke(trainer, model, values, index, entry):
    """Run one placement entry point on a single batch."""
    model.seen_inputs.clear()
    loader = [(values, index)]
    if entry == "run_epoch":
        return trainer.run_epoch(model, loader, None, None, False)
    if entry == "extract_representation":
        return trainer.extract_representation(model, loader)
    if entry == "compute_energy":
        return trainer.compute_energy(model, loader)
    raise AssertionError(entry)


def _assert_caller_unchanged(test, tensor, snapshot):
    """The trainer must not rewrite the caller's storage."""
    test.assertIs(tensor, snapshot["tensor"])
    test.assertEqual(tensor.data_ptr(), snapshot["ptr"])
    test.assertEqual(tensor.device, snapshot["device"])
    test.assertEqual(tensor.dtype, snapshot["dtype"])
    test.assertEqual(tensor.is_contiguous(), snapshot["contiguous"])
    if snapshot["values"] is not None:
        test.assertTrue(torch.equal(tensor, snapshot["values"]))


def _snapshot(tensor):
    """Capture storage identity. Meta tensors have no values to clone.

    Every meta tensor reports ``data_ptr() == 0``, so meta identity has to be
    the Python object itself.
    """
    values = None
    if tensor.device.type != "meta":
        values = tensor.detach().clone()
    return {
        "tensor": tensor,
        "ptr": tensor.data_ptr(),
        "device": tensor.device,
        "dtype": tensor.dtype,
        "contiguous": tensor.is_contiguous(),
        "values": values,
    }


def _assert_entries(case, model, values, index, device, dtype, same_storage):
    """Check the batch each of the three entry points handed the probe.

    ``_exercise`` records ``run_epoch``, then ``extract_representation``, then
    ``compute_energy``. The energy path has no batch index. ``same_storage``
    is true when device and dtype already matched, so ``Tensor.to`` must return
    the caller, including a noncontiguous view.
    """
    case.assertEqual(len(model.seen_inputs), 3)
    for nth, (seen_values, seen_index, activated) in enumerate(model.seen_inputs):
        case.assertEqual(seen_values.device, device)
        case.assertEqual(seen_values.dtype, dtype)
        case.assertEqual(activated.device, device)
        case.assertEqual(activated.dtype, dtype)
        if same_storage:
            case.assertIs(seen_values, values)
            case.assertEqual(seen_values.is_contiguous(), values.is_contiguous())
        else:
            case.assertIsNot(seen_values, values)
            if values.device.type != "meta":
                case.assertNotEqual(seen_values.data_ptr(), values.data_ptr())
        if device.type != "meta":
            case.assertTrue(torch.isfinite(activated).all())
        if nth < 2:
            case.assertEqual(seen_index.device, device)
            case.assertEqual(seen_index.dtype, torch.long)
            if same_storage:
                case.assertIs(seen_index, index)
        else:
            case.assertIsNone(seen_index)


class TestTrainerParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_cpu_methods_keep_caller_storage(self):
        """An unmoved CPU model keeps a noncontiguous caller and returns results."""
        trainer = _trainer()
        model = ProbeModel()
        values, index = _batch(noncontiguous=True)
        value_snapshot = _snapshot(values)
        index_snapshot = _snapshot(index)
        self.assertFalse(values.is_contiguous())

        metrics, representations, energies = _exercise(trainer, model, values, index)

        _assert_caller_unchanged(self, values, value_snapshot)
        _assert_caller_unchanged(self, index, index_snapshot)
        _assert_entries(self, model, values, index, CPU, F32, True)
        self.assertEqual(representations.shape, (2, 3))
        self.assertEqual(len(energies), 2)
        self.assertTrue(all(item == 0.0 for item in metrics.values()))
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(trainer.device, torch.device("cpu"))

    def test_parent_to_meta_skips_child_to(self):
        """Positional and keyword ``Module.to(meta)`` do not call child ``.to``."""
        moves = (
            ("positional", lambda parent: parent.to("meta")),
            ("keyword", lambda parent: parent.to(device="meta")),
        )
        for label, move in moves:
            with self.subTest(move=label):
                trainer = _trainer()
                model = ProbeModel()
                parent = nn.Module()
                parent.model = model
                values, index = _batch(noncontiguous=True)
                snapshot = _snapshot(values)

                move(parent)
                _exercise(trainer, model, values, index)

                self.assertEqual(model.direct_to_calls, [])
                self.assertEqual(next(model.parameters()).device, torch.device("meta"))
                self.assertEqual(trainer.device, torch.device("cpu"))
                _assert_caller_unchanged(self, values, snapshot)
                _assert_entries(self, model, values, index, META, F32, False)

    def test_parent_apply_places_cpu_batch_on_meta(self):
        """A CPU batch is placed on meta without rewriting the caller."""
        trainer = _trainer()
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model
        values, index = _batch(dtype=torch.float64, noncontiguous=True)
        value_snapshot = _snapshot(values)
        index_snapshot = _snapshot(index)
        _apply_device(parent, "meta")

        _exercise(trainer, model, values, index)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(trainer.device, torch.device("cpu"))
        self.assertEqual(next(model.parameters()).dtype, torch.float32)
        _assert_caller_unchanged(self, values, value_snapshot)
        _assert_caller_unchanged(self, index, index_snapshot)
        _assert_entries(self, model, values, index, META, F64, False)

    def test_meta_batch_keeps_storage(self):
        """A meta batch that already matches the weights stays the same object."""
        for noncontiguous in (False, True):
            with self.subTest(noncontiguous=noncontiguous):
                trainer = _trainer()
                model = ProbeModel()
                _apply_device(model, "meta")
                values, index = _batch(device="meta", noncontiguous=noncontiguous)
                self.assertEqual(values.is_contiguous(), not noncontiguous)

                _exercise(trainer, model, values, index)

                self.assertEqual(model.direct_to_calls, [])
                self.assertEqual(trainer.device, torch.device("cpu"))
                _assert_entries(self, model, values, index, META, F32, True)

    def test_float64_parameters_keep_feature_dtype(self):
        """``_apply(meta)`` after a dtype change must not recast the batch."""
        trainer = _trainer()
        model = ProbeModel()
        _apply_device(model, "cpu")
        model.proj.to(dtype=torch.float64)
        _apply_device(model, "meta")
        values, index = _batch(dtype=torch.float64, noncontiguous=True)
        snapshot = _snapshot(values)

        _exercise(trainer, model, values, index)

        _assert_caller_unchanged(self, values, snapshot)
        _assert_entries(self, model, values, index, META, F64, False)
        parameter = next(model.parameters())
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(parameter.device, torch.device("meta"))
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(trainer.device, torch.device("cpu"))

    def test_dtype_only_move_keeps_cpu_device(self):
        """``parent.to(dtype=...)`` stays on CPU and does not call child ``.to``.

        A poisoned ``trainer.device`` must not pull the float64 batch onto meta.
        A noncontiguous caller that already matches keeps its storage.
        """
        trainer = _trainer()
        trainer.device = torch.device("meta")
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model
        values, index = _batch(dtype=torch.float64, noncontiguous=True)
        snapshot = _snapshot(values)

        parent.to(dtype=torch.float64)
        _exercise(trainer, model, values, index)

        parameter = next(model.parameters())
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(parameter.device, torch.device("cpu"))
        self.assertEqual(trainer.device, torch.device("meta"))
        _assert_caller_unchanged(self, values, snapshot)
        _assert_entries(self, model, values, index, CPU, F64, True)

    def test_direct_dtype_move_keeps_cpu_and_identity(self):
        """A direct ``model.to(dtype=...)`` is the one recorded call and stays on CPU."""
        trainer = _trainer()
        trainer.device = torch.device("meta")
        model = ProbeModel()
        values, index = _batch(dtype=torch.float64, noncontiguous=True)

        model.to(dtype=torch.float64)
        _exercise(trainer, model, values, index)

        parameter = next(model.parameters())
        self.assertEqual(model.direct_to_calls, [((), {"dtype": torch.float64})])
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(parameter.device, torch.device("cpu"))
        self.assertEqual(trainer.device, torch.device("meta"))
        _assert_entries(self, model, values, index, CPU, F64, True)

    def test_device_is_reread_after_meta_round_trip(self):
        """Meta, a rejected ``.to(cpu)``, zeros, then meta again.

        ``parent.to(cpu)`` raises and leaves the parameters on meta. The return
        trip uses ``zeros_like`` through ``_apply``. The cache is poisoned on
        the CPU stop and must still lose to the live weights. All three entry
        points are reread on every leg.
        """
        trainer = _trainer()
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model
        values, index = _batch(noncontiguous=True)
        value_snapshot = _snapshot(values)
        index_snapshot = _snapshot(index)

        _apply_device(parent, "meta")
        _exercise(trainer, model, values, index)
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(trainer.device, torch.device("cpu"))
        _assert_entries(self, model, values, index, META, F32, False)

        with self.assertRaises(NotImplementedError) as blocked:
            parent.to("cpu")
        self.assertIn("meta", str(blocked.exception).lower())
        self.assertEqual(next(model.parameters()).device, torch.device("meta"))
        self.assertEqual(model.direct_to_calls, [])

        _apply_device(parent, "cpu")
        weight = next(model.parameters())
        self.assertEqual(weight.device, torch.device("cpu"))
        self.assertTrue(torch.isfinite(weight).all())
        trainer.device = torch.device("meta")
        _exercise(trainer, model, values, index)
        self.assertEqual(model.direct_to_calls, [])
        _assert_entries(self, model, values, index, CPU, F32, True)

        _apply_device(parent, "meta")
        _exercise(trainer, model, values, index)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(trainer.device, torch.device("meta"))
        self.assertEqual(next(model.parameters()).device, torch.device("meta"))
        _assert_caller_unchanged(self, values, value_snapshot)
        _assert_caller_unchanged(self, index, index_snapshot)
        _assert_entries(self, model, values, index, META, F32, False)

    def test_each_batch_rereads_parameter_device(self):
        """A move between batches is visible to every entry point."""
        trainer = _trainer()
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model
        values, index = _batch(noncontiguous=True)
        snapshot = _snapshot(values)
        self.assertEqual(trainer.device, torch.device("cpu"))

        for entry in _ENTRIES:
            with self.subTest(entry=entry):
                _apply_device(parent, "cpu")
                weight = next(model.parameters())
                self.assertEqual(weight.device, torch.device("cpu"))
                self.assertTrue(torch.isfinite(weight).all())
                model.direct_to_calls.clear()
                model.seen_inputs.clear()
                loader = _MoveDuringIteration(parent, values, index, "meta")
                if entry == "run_epoch":
                    trainer.run_epoch(model, loader, None, None, False)
                elif entry == "extract_representation":
                    trainer.extract_representation(model, loader)
                else:
                    trainer.compute_energy(model, loader)

                self.assertEqual(model.direct_to_calls, [])
                self.assertEqual(len(model.seen_inputs), 2)
                first_values, first_index, first_activated = model.seen_inputs[0]
                second_values, second_index, second_activated = model.seen_inputs[1]
                self.assertIs(first_values, values)
                self.assertFalse(first_values.is_contiguous())
                self.assertEqual(first_activated.device, torch.device("cpu"))
                self.assertTrue(torch.isfinite(first_activated).all())
                self.assertIsNot(second_values, values)
                self.assertEqual(second_values.device, torch.device("meta"))
                self.assertEqual(second_activated.device, torch.device("meta"))
                if entry == "compute_energy":
                    self.assertIsNone(first_index)
                    self.assertIsNone(second_index)
                else:
                    self.assertIs(first_index, index)
                    self.assertEqual(second_index.device, torch.device("meta"))
                    self.assertEqual(second_index.dtype, torch.long)
                _assert_caller_unchanged(self, values, snapshot)

    def test_stale_cache_does_not_win(self):
        """A poisoned ``trainer.device`` must not pull a CPU model onto meta."""
        trainer = _trainer()
        model = ProbeModel()
        trainer.device = torch.device("meta")
        values, index = _batch(noncontiguous=True)
        snapshot = _snapshot(values)

        _exercise(trainer, model, values, index)

        self.assertEqual(trainer.device, torch.device("meta"))
        self.assertEqual(next(model.parameters()).device, torch.device("cpu"))
        self.assertEqual(model.direct_to_calls, [])
        _assert_caller_unchanged(self, values, snapshot)
        _assert_entries(self, model, values, index, CPU, F32, True)

    def test_cuda_preference_does_not_redirect_cpu_weights(self):
        """A CUDA cache cannot move CPU weights. No GPU is required to reject it."""
        trainer = _trainer()
        model = ProbeModel()
        trainer.device = torch.device("cuda")
        values, index = _batch(noncontiguous=True)
        snapshot = _snapshot(values)

        _exercise(trainer, model, values, index)

        self.assertEqual(trainer.device, torch.device("cuda"))
        self.assertEqual(next(model.parameters()).device, torch.device("cpu"))
        self.assertEqual(model.direct_to_calls, [])
        _assert_caller_unchanged(self, values, snapshot)
        _assert_entries(self, model, values, index, CPU, F32, True)

    def test_parameter_free_model_follows_the_batch(self):
        """No parameters: a meta batch stays on meta, a CPU batch stays on CPU."""
        trainer = _trainer()
        trainer.device = torch.device("meta")
        model = ParameterFreeModel()
        _apply_device(model, "meta")
        meta_values, meta_index = _batch(device="meta", noncontiguous=True)
        self.assertFalse(meta_values.is_contiguous())

        _exercise(trainer, model, meta_values, meta_index)

        self.assertEqual(trainer.device, torch.device("meta"))
        self.assertIsNone(next(model.parameters(), None))
        _assert_entries(self, model, meta_values, meta_index, META, F32, True)

        cpu_values, cpu_index = _batch(noncontiguous=True)
        cpu_snapshot = _snapshot(cpu_values)
        _exercise(trainer, model, cpu_values, cpu_index)
        _assert_caller_unchanged(self, cpu_values, cpu_snapshot)
        _assert_entries(self, model, cpu_values, cpu_index, CPU, F32, True)

    def test_parameter_free_model_ignores_cuda_preference(self):
        """With no parameters, a CUDA cache must not be the placement device."""
        trainer = _trainer()
        trainer.device = torch.device("cuda")
        model = ParameterFreeModel()
        values, index = _batch(noncontiguous=True)
        snapshot = _snapshot(values)

        _exercise(trainer, model, values, index)

        self.assertEqual(trainer.device, torch.device("cuda"))
        self.assertIsNone(next(model.parameters(), None))
        _assert_caller_unchanged(self, values, snapshot)
        _assert_entries(self, model, values, index, CPU, F32, True)

    def test_same_device_alignment_keeps_autograd(self):
        """A poisoned cache still backprops into an already-aligned CPU batch."""
        trainer = _trainer()
        trainer.device = torch.device("meta")
        model = ProbeModel()
        model.connect_loss = True
        values, index = _batch(requires_grad=True)
        pointer = values.data_ptr()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer.run_epoch(model, [(values, index)], optimizer, optimizer, True)

        self.assertIs(model.seen_inputs[0][0], values)
        self.assertEqual(values.data_ptr(), pointer)
        self.assertEqual(trainer.device, torch.device("meta"))
        self.assertEqual(next(model.parameters()).device, torch.device("cpu"))
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertGreater(values.grad.abs().sum().item(), 0.0)

    def test_probe_matmul_names_both_devices(self):
        """Cache placement crashes the linear probe and names both devices.

        This locks the probe. A stub that skipped the matmul would let a stale
        device look green. The message contains ``meta`` and ``cpu``.
        """
        cpu_batch, _index = _batch()
        meta_model = ProbeModel()
        _apply_device(meta_model, "meta")
        with self.assertRaises(RuntimeError) as meta_weights:
            meta_model.proj(cpu_batch)
        meta_message = str(meta_weights.exception).lower()
        self.assertIn("meta", meta_message)
        self.assertIn("cpu", meta_message)

        cpu_model = ProbeModel()
        meta_batch, _meta_index = _batch(device="meta")
        with self.assertRaises(RuntimeError) as cpu_weights:
            cpu_model.proj(meta_batch)
        cpu_message = str(cpu_weights.exception).lower()
        self.assertIn("meta", cpu_message)
        self.assertIn("cpu", cpu_message)

    def test_restored_cache_helper_raises_on_every_entry(self):
        """Restoring construction-cache placement reproduces the device mix.

        The live helper does not. After the override is removed, every entry
        point places the batch on meta and does not call child ``.to``.
        """
        trainer = _trainer()
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model
        _apply_device(parent, "meta")
        values, index = _batch(noncontiguous=True)
        snapshot = _snapshot(values)

        def _on_cache(module, fallback):
            """Return the pre-fix device, the construction cache."""
            del module, fallback
            return trainer.device

        trainer._parameter_device = _on_cache
        for entry in _ENTRIES:
            with self.subTest(entry=entry, placement="cache"):
                with self.assertRaises(RuntimeError) as caught:
                    _invoke(trainer, model, values, index, entry)
                message = str(caught.exception).lower()
                self.assertIn("meta", message)
                self.assertIn("cpu", message)

        del trainer._parameter_device
        _exercise(trainer, model, values, index)
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(trainer.device, torch.device("cpu"))
        self.assertEqual(next(model.parameters()).device, torch.device("meta"))
        _assert_caller_unchanged(self, values, snapshot)
        _assert_entries(self, model, values, index, META, F32, False)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestTrainerParameterDeviceCuda(unittest.TestCase):
    """CUDA coverage of the construct-on-CPU / parent-move path."""

    def test_parent_to_cuda_skips_child_to(self):
        """``parent.to('cuda')`` and ``to(device='cuda')`` do not call child ``.to``."""
        moves = (
            ("positional", lambda parent: parent.to("cuda")),
            ("keyword", lambda parent: parent.to(device="cuda")),
        )
        for label, move in moves:
            with self.subTest(move=label):
                trainer = _trainer()
                model = ProbeModel()
                parent = nn.Module()
                parent.model = model
                values, index = _batch(noncontiguous=True)
                snapshot = _snapshot(values)

                move(parent)
                _exercise(trainer, model, values, index)

                self.assertEqual(model.direct_to_calls, [])
                self.assertEqual(next(model.parameters()).device.type, "cuda")
                self.assertEqual(trainer.device, torch.device("cpu"))
                _assert_caller_unchanged(self, values, snapshot)
                _assert_entries(self, model, values, index, CUDA, F32, False)

    def test_parent_apply_places_cpu_batch_on_cuda(self):
        """A CPU batch stays put. The matmul runs on CUDA and is finite."""
        trainer = _trainer()
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model
        values, index = _batch(noncontiguous=True)
        snapshot = _snapshot(values)
        _apply_device(parent, "cuda")

        _exercise(trainer, model, values, index)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(trainer.device, torch.device("cpu"))
        _assert_caller_unchanged(self, values, snapshot)
        _assert_entries(self, model, values, index, CUDA, F32, False)

    def test_cuda_batch_keeps_storage(self):
        """A CUDA batch that already matches the weights keeps its storage."""
        for noncontiguous in (False, True):
            with self.subTest(noncontiguous=noncontiguous):
                trainer = _trainer()
                model = ProbeModel()
                _apply_device(model, "cuda")
                values, index = _batch(device="cuda", noncontiguous=noncontiguous)
                self.assertEqual(values.is_contiguous(), not noncontiguous)

                _exercise(trainer, model, values, index)

                self.assertEqual(model.direct_to_calls, [])
                _assert_entries(self, model, values, index, CUDA, F32, True)

    def test_device_is_reread_across_cuda_and_cpu(self):
        """CUDA, CPU, CUDA. Child ``.to`` is not called. Every entry point follows."""
        trainer = _trainer()
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model
        values, index = _batch(noncontiguous=True)
        snapshot = _snapshot(values)

        _apply_device(parent, "cuda")
        _exercise(trainer, model, values, index)
        _assert_entries(self, model, values, index, CUDA, F32, False)

        _apply_device(parent, "cpu")
        trainer.device = torch.device("cuda")
        _exercise(trainer, model, values, index)
        _assert_entries(self, model, values, index, CPU, F32, True)

        _apply_device(parent, "cuda")
        _exercise(trainer, model, values, index)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(trainer.device, torch.device("cuda"))
        _assert_caller_unchanged(self, values, snapshot)
        _assert_entries(self, model, values, index, CUDA, F32, False)

    def test_float64_cuda_move_keeps_dtype(self):
        """float64 survives a CUDA ``_apply`` and is not recast on any entry point."""
        trainer = _trainer()
        model = ProbeModel()
        model.proj.to(dtype=torch.float64)
        _apply_device(model, "cuda")
        values, index = _batch(dtype=torch.float64, noncontiguous=True)
        snapshot = _snapshot(values)

        _exercise(trainer, model, values, index)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(next(model.parameters()).dtype, torch.float64)
        self.assertEqual(next(model.parameters()).device.type, "cuda")
        self.assertEqual(trainer.device, torch.device("cpu"))
        _assert_caller_unchanged(self, values, snapshot)
        _assert_entries(self, model, values, index, CUDA, F64, False)

    def test_dtype_only_move_keeps_cuda_device(self):
        """A dtype-only parent move does not pull CUDA parameters back to CPU.

        The construction cache stays CPU. A float64 CUDA batch that matches the
        weights keeps identity on every entry point.
        """
        trainer = _trainer()
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model
        _apply_device(parent, "cuda")
        parent.to(dtype=torch.float64)
        values, index = _batch(dtype=torch.float64, device="cuda", noncontiguous=True)
        self.assertFalse(values.is_contiguous())

        _exercise(trainer, model, values, index)

        parameter = next(model.parameters())
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(parameter.device.type, "cuda")
        self.assertEqual(parameter.dtype, torch.float64)
        self.assertEqual(trainer.device, torch.device("cpu"))
        _assert_entries(self, model, values, index, CUDA, F64, True)

    def test_stale_cpu_cache_does_not_pull_cuda_weights(self):
        """Forcing the cache back to CPU must not pull CUDA weights to the host."""
        trainer = _trainer()
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model
        _apply_device(parent, "cuda")
        trainer.device = torch.device("cpu")
        values, index = _batch(noncontiguous=True)
        snapshot = _snapshot(values)

        _exercise(trainer, model, values, index)

        self.assertEqual(trainer.device, torch.device("cpu"))
        self.assertEqual(next(model.parameters()).device.type, "cuda")
        self.assertEqual(model.direct_to_calls, [])
        _assert_caller_unchanged(self, values, snapshot)
        _assert_entries(self, model, values, index, CUDA, F32, False)

    def test_cuda_batch_on_cpu_model_keeps_caller_storage(self):
        """A CUDA batch is scored on a CPU model. Caller storage stays on CUDA."""
        for noncontiguous in (False, True):
            with self.subTest(noncontiguous=noncontiguous):
                trainer = _trainer()
                trainer.device = torch.device("cuda")
                model = ProbeModel()
                values, index = _batch(device="cuda", noncontiguous=noncontiguous)
                pointer = values.data_ptr()
                self.assertEqual(values.is_contiguous(), not noncontiguous)

                _exercise(trainer, model, values, index)

                self.assertEqual(values.data_ptr(), pointer)
                self.assertEqual(values.device.type, "cuda")
                self.assertEqual(values.is_contiguous(), not noncontiguous)
                self.assertEqual(trainer.device.type, "cuda")
                self.assertEqual(next(model.parameters()).device, torch.device("cpu"))
                _assert_entries(self, model, values, index, CPU, F32, False)


if __name__ == "__main__":
    unittest.main()
