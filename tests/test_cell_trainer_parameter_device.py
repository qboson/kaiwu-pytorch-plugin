"""Regression tests for the cell-example Trainer device cache.

``Trainer`` stores the constructor device and later sends every batch through
``tensor.to(self.device)``. A parent ``nn.Module.to`` / ``_apply`` moves the
model parameters without calling the child ``.to`` (merged #164 /
``tests/test_bm_device_sync.py``) and without updating that cache. The next
``run_epoch``, ``extract_representation`` or ``compute_energy`` then feeds a
CPU batch to weights that already left CPU:

``RuntimeError: Tensor on device meta is not on the expected device cpu!``

Placement follows the live parameter device and is reread on every call. The
constructor cache is not consulted once the model has parameters. A
parameter-free model keeps the caller batch's device. Callers are not updated
in place, and dtypes stay put. ``.to(cpu)`` cannot copy out of a meta tensor,
so a meta round trip retargets storage with ``_apply``.

The probe module is intentional. ``CellQVAE.forward`` draws an
``Exponential`` sample whose beta tensor is a separate open defect; these
tests stop at the trainer's batch placement, which is what the three entry
points actually control. The probe's linear layer is a real parameter matmul,
so a stale device still raises the runtime error above.
"""

import os
import sys
import types
import unittest

import torch
from torch import nn

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
CELL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../example/qvae_cell")
)
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, CELL_ROOT)

import kaiwu  # pylint: disable=wrong-import-position


def _install_sdk_stubs():
    """Stub Kaiwu SDK modules that ``trainer.py`` imports but never calls here."""
    stubs = {
        "kaiwu.classical": {"SimulatedAnnealingOptimizer": object},
        "kaiwu.cim": {"CIMOptimizer": object},
        "kaiwu.preprocess": {"PrecisionReducer": object},
    }
    for name, attrs in stubs.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        setattr(kaiwu, name.rsplit(".", maxsplit=1)[-1], module)


_install_sdk_stubs()

# pylint: disable-next=wrong-import-position,wrong-import-order
from trainer import Trainer  # noqa: E402


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


def _trainer(device="cpu"):
    """Build a trainer whose cached device is independent of later moves."""
    args = types.SimpleNamespace(
        representation="q",
        loss_type="mse",
        bm_weight_decay=0.0,
    )
    return Trainer(args, torch.device(device))


def _batch(
    rows=2,
    features=4,
    dtype=torch.float32,
    device="cpu",
    noncontiguous=False,
    requires_grad=False,
):
    """Return one ``(features, batch index)`` pair."""
    if noncontiguous:
        base = torch.arange(rows * features * 2, dtype=dtype).reshape(
            rows, features * 2
        )
        values = base[:, ::2]
    else:
        values = torch.arange(rows * features, dtype=dtype).reshape(rows, features)
    if requires_grad:
        values.requires_grad_(True)
    index = torch.arange(rows, dtype=torch.long)
    destination = torch.device(device)
    if destination.type != "cpu":
        values = values.to(destination)
        index = index.to(destination)
    return values, index


def _apply_device(module, device):
    """Move parameters the way a parent module does.

    Leaving ``meta`` cannot copy values out, so that direction allocates fresh
    zeros. Every other hop uses ``Tensor.to``.
    """
    destination = torch.device(device)

    def _move(tensor):
        if tensor.device.type == "meta" and destination.type != "meta":
            return torch.zeros_like(tensor, device=destination)
        return tensor.to(destination)

    module._apply(_move)  # pylint: disable=protected-access


def _exercise(trainer, model, values, index):
    """Run the three entry points that place a batch."""
    model.seen_inputs.clear()
    loader = [(values, index)]
    metrics = trainer.run_epoch(model, loader, None, None, False)
    representations = trainer.extract_representation(model, loader)
    energies = trainer.compute_energy(model, loader)
    return metrics, representations, energies


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


class TestTrainerParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_cpu_methods_keep_caller_storage(self):
        """An unmoved CPU model keeps the caller batch and returns finite results."""
        trainer = _trainer()
        model = ProbeModel()
        values, index = _batch(noncontiguous=True)
        value_snapshot = _snapshot(values)
        index_snapshot = _snapshot(index)

        metrics, representations, energies = _exercise(trainer, model, values, index)

        _assert_caller_unchanged(self, values, value_snapshot)
        _assert_caller_unchanged(self, index, index_snapshot)
        self.assertEqual(len(model.seen_inputs), 3)
        for seen_values, seen_index, activated in model.seen_inputs:
            self.assertEqual(seen_values.device, torch.device("cpu"))
            self.assertEqual(activated.device, torch.device("cpu"))
            self.assertTrue(torch.isfinite(activated).all())
            if seen_index is not None:
                self.assertEqual(seen_index.data_ptr(), index.data_ptr())
                self.assertEqual(seen_values.data_ptr(), values.data_ptr())
        self.assertEqual(representations.shape, (2, 3))
        self.assertEqual(len(energies), 2)
        self.assertTrue(all(item == 0.0 for item in metrics.values()))
        self.assertEqual(model.direct_to_calls, [])

    def test_parent_to_meta_skips_child_to(self):
        """``parent.to`` moves parameters through ``_apply`` and skips ``.to``."""
        trainer = _trainer()
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model

        parent.to("meta")

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(next(model.parameters()).device, torch.device("meta"))
        self.assertEqual(trainer.device, torch.device("cpu"))

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
        _assert_caller_unchanged(self, values, value_snapshot)
        _assert_caller_unchanged(self, index, index_snapshot)
        self.assertEqual(len(model.seen_inputs), 3)
        for seen_values, seen_index, activated in model.seen_inputs:
            self.assertEqual(seen_values.device, torch.device("meta"))
            self.assertEqual(seen_values.dtype, torch.float64)
            self.assertEqual(activated.device, torch.device("meta"))
            self.assertEqual(activated.dtype, torch.float64)
            self.assertNotEqual(seen_values.data_ptr(), values.data_ptr())
            if seen_index is not None:
                self.assertEqual(seen_index.device, torch.device("meta"))
                self.assertEqual(seen_index.dtype, torch.long)

    def test_meta_batch_keeps_storage(self):
        """A batch that is already on meta stays the same object."""
        trainer = _trainer()
        model = ProbeModel()
        _apply_device(model, "meta")
        values, index = _batch(device="meta")
        value_ptr = values.data_ptr()
        index_ptr = index.data_ptr()

        _exercise(trainer, model, values, index)

        self.assertIs(values, model.seen_inputs[0][0])
        self.assertIs(index, model.seen_inputs[0][1])
        self.assertEqual(values.data_ptr(), value_ptr)
        self.assertEqual(index.data_ptr(), index_ptr)

    def test_float64_parameters_keep_feature_dtype(self):
        """``_apply(meta)`` after a dtype change must not recast the batch."""
        trainer = _trainer()
        model = ProbeModel()
        _apply_device(model, "cpu")
        model.proj.to(dtype=torch.float64)  # direct, still CPU
        _apply_device(model, "meta")
        values, index = _batch(dtype=torch.float64)
        snapshot = _snapshot(values)

        _exercise(trainer, model, values, index)

        _assert_caller_unchanged(self, values, snapshot)
        seen_values, _, activated = model.seen_inputs[0]
        self.assertEqual(seen_values.dtype, torch.float64)
        self.assertEqual(activated.dtype, torch.float64)
        self.assertEqual(next(model.parameters()).dtype, torch.float64)
        self.assertEqual(next(model.parameters()).device, torch.device("meta"))

    def test_dtype_only_move_keeps_cpu_device(self):
        """``parent.to(dtype=...)`` does not call the child ``.to`` or leave CPU."""
        trainer = _trainer()
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model
        values, index = _batch(dtype=torch.float64)

        parent.to(dtype=torch.float64)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(next(model.parameters()).dtype, torch.float64)
        self.assertEqual(next(model.parameters()).device, torch.device("cpu"))
        _exercise(trainer, model, values, index)
        seen_values, _, activated = model.seen_inputs[0]
        self.assertEqual(seen_values.device, torch.device("cpu"))
        self.assertEqual(seen_values.dtype, torch.float64)
        self.assertEqual(activated.device, torch.device("cpu"))
        self.assertTrue(torch.isfinite(activated).all())

    def test_device_is_reread_after_meta_round_trip(self):
        """Meta, CPU zeros, then meta again. The cache is never the source."""
        trainer = _trainer()
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model
        values, index = _batch()
        snapshot = _snapshot(values)

        _apply_device(parent, "meta")
        _exercise(trainer, model, values, index)
        self.assertEqual(model.seen_inputs[-1][0].device, torch.device("meta"))

        _apply_device(parent, "cpu")
        _exercise(trainer, model, values, index)
        self.assertEqual(model.seen_inputs[-1][0].device, torch.device("cpu"))
        self.assertTrue(torch.isfinite(model.seen_inputs[-1][2]).all())

        _apply_device(parent, "meta")
        _exercise(trainer, model, values, index)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(trainer.device, torch.device("cpu"))
        self.assertEqual(model.seen_inputs[-1][0].device, torch.device("meta"))
        _assert_caller_unchanged(self, values, snapshot)

    def test_stale_cache_does_not_win(self):
        """A poisoned ``trainer.device`` must not pull a CPU model onto meta."""
        trainer = _trainer()
        model = ProbeModel()
        trainer.device = torch.device("meta")
        values, index = _batch()

        _exercise(trainer, model, values, index)

        self.assertEqual(trainer.device, torch.device("meta"))
        self.assertEqual(next(model.parameters()).device, torch.device("cpu"))
        for seen_values, _, activated in model.seen_inputs:
            self.assertEqual(seen_values.device, torch.device("cpu"))
            self.assertEqual(activated.device, torch.device("cpu"))
            self.assertTrue(torch.isfinite(activated).all())

    def test_parameter_free_model_follows_the_batch(self):
        """No parameters: a meta batch stays on meta, a CPU batch stays on CPU."""
        trainer = _trainer()
        trainer.device = torch.device("meta")
        model = ParameterFreeModel()
        _apply_device(model, "meta")
        meta_values, meta_index = _batch(device="meta")
        meta_ptr = meta_values.data_ptr()

        _exercise(trainer, model, meta_values, meta_index)

        self.assertIs(model.seen_inputs[0][0], meta_values)
        self.assertEqual(model.seen_inputs[0][0].device, torch.device("meta"))
        self.assertEqual(meta_values.data_ptr(), meta_ptr)

        cpu_values, cpu_index = _batch()
        cpu_ptr = cpu_values.data_ptr()
        _exercise(trainer, model, cpu_values, cpu_index)
        self.assertEqual(cpu_values.data_ptr(), cpu_ptr)
        self.assertEqual(model.seen_inputs[0][0].data_ptr(), cpu_ptr)
        self.assertEqual(model.seen_inputs[0][0].device, torch.device("cpu"))

    def test_same_device_alignment_keeps_autograd(self):
        """A CPU batch that is already aligned still receives ``backward``."""
        trainer = _trainer()
        model = ProbeModel()
        model.connect_loss = True
        values, index = _batch(requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer.run_epoch(model, [(values, index)], optimizer, optimizer, True)

        self.assertEqual(model.seen_inputs[0][0].data_ptr(), values.data_ptr())
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertGreater(values.grad.abs().sum().item(), 0.0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestTrainerParameterDeviceCuda(unittest.TestCase):
    """CUDA coverage of the construct-on-CPU / parent-move path."""

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
        _assert_caller_unchanged(self, values, snapshot)
        for _, _, activated in model.seen_inputs:
            self.assertEqual(activated.device.type, "cuda")
            self.assertTrue(torch.isfinite(activated.detach().cpu()).all())

    def test_cuda_batch_keeps_storage(self):
        """A batch that is already on CUDA stays the same object."""
        trainer = _trainer()
        model = ProbeModel()
        _apply_device(model, "cuda")
        values, index = _batch(device="cuda")
        pointer = values.data_ptr()

        _exercise(trainer, model, values, index)

        self.assertIs(model.seen_inputs[0][0], values)
        self.assertEqual(values.data_ptr(), pointer)

    def test_device_is_reread_across_cuda_and_cpu(self):
        """CUDA, CPU, CUDA. Child ``.to`` is not called."""
        trainer = _trainer()
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model
        values, index = _batch()
        snapshot = _snapshot(values)

        _apply_device(parent, "cuda")
        _exercise(trainer, model, values, index)
        self.assertEqual(model.seen_inputs[-1][0].device.type, "cuda")

        _apply_device(parent, "cpu")
        _exercise(trainer, model, values, index)
        self.assertEqual(model.seen_inputs[-1][0].device.type, "cpu")

        _apply_device(parent, "cuda")
        _exercise(trainer, model, values, index)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(model.seen_inputs[-1][0].device.type, "cuda")
        _assert_caller_unchanged(self, values, snapshot)

    def test_float64_cuda_move_keeps_dtype(self):
        """float64 survives a CUDA ``_apply``."""
        trainer = _trainer()
        model = ProbeModel()
        model.proj.to(dtype=torch.float64)
        _apply_device(model, "cuda")
        values, index = _batch(dtype=torch.float64)

        _exercise(trainer, model, values, index)

        seen_values, _, activated = model.seen_inputs[0]
        self.assertEqual(seen_values.dtype, torch.float64)
        self.assertEqual(activated.dtype, torch.float64)
        self.assertEqual(activated.device.type, "cuda")

    def test_dtype_only_move_keeps_cuda_device(self):
        """A dtype-only parent move does not pull CUDA parameters back to CPU."""
        trainer = _trainer()
        model = ProbeModel()
        parent = nn.Module()
        parent.model = model
        _apply_device(parent, "cuda")
        values, index = _batch(dtype=torch.float64, device="cuda")

        parent.to(dtype=torch.float64)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(next(model.parameters()).device.type, "cuda")
        self.assertEqual(next(model.parameters()).dtype, torch.float64)
        _exercise(trainer, model, values, index)
        _, _, activated = model.seen_inputs[0]
        self.assertEqual(activated.device.type, "cuda")
        self.assertEqual(activated.dtype, torch.float64)

    def test_cuda_batch_on_cpu_model_keeps_caller_storage(self):
        """A CUDA batch is scored on a CPU model. Caller storage stays on CUDA."""
        trainer = _trainer()
        model = ProbeModel()
        values, index = _batch(device="cuda")
        pointer = values.data_ptr()

        _exercise(trainer, model, values, index)

        self.assertEqual(values.data_ptr(), pointer)
        self.assertEqual(values.device.type, "cuda")
        for seen_values, _, activated in model.seen_inputs:
            self.assertEqual(seen_values.device.type, "cpu")
            self.assertEqual(activated.device.type, "cpu")
            self.assertTrue(torch.isfinite(activated).all())


if __name__ == "__main__":
    unittest.main()
