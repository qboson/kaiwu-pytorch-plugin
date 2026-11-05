"""Regression tests for conditioned BM energy device placement.

``BMConditionedEnergyModel`` feeds caller tokens into the encoder and the
pooled features into ``feature_projector``. A parent ``nn.Module.to`` /
``_apply`` moves those parameters without calling the energy model's ``.to``
(merged #164 / ``tests/test_bm_device_sync.py``). CPU tokens then meet
embedding or linear weights that already moved:

``RuntimeError: Tensor on device cpu is not on the expected device meta!``

Visible logits follow the live parameter device. The device is reread on
every call. ``energy_bm.device`` is not consulted: a stale Boltzmann-machine
cache must not pull features onto itself. ``.to(cpu)`` cannot copy out of a
meta tensor, so a meta round trip retargets storage with ``_apply``.

Same-device inputs stay the same objects, which is the CPU identity contract
pinned by open #1244. Callers are not updated in place, and dtypes stay put.
These tests stop at ``build_visible_logits``. They do not enter
``score_visible_logits`` or BM sampling.

Out of scope: example token placement (#1292), BM sampling / ``forward``
(#1268 / #1347), mixed-precision pooling (#259), and library ``qdiffusion.py``
metadata / constructor dtype (#488 / #310).
"""

# Parent moves go through ``_apply`` (see merged #164).
# pylint: disable=protected-access

from __future__ import annotations

import copy
import os
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/qdiffusion")),
)

from dplm.models.bm import (  # pylint: disable=wrong-import-position
    BMConditionedEnergyModel,
    _parameter_device,
)

_CPU = torch.device("cpu")
_META = torch.device("meta")
_LOGIT_SHAPE = (2, 3)
_FEATURE_SHAPE = (2, 8)


class _ExplodingSampler:  # pylint: disable=too-few-public-methods
    """Sampler that fails if scoring leaves the projector."""

    def solve(self, ising_mat):
        """Refuse BM sampling; this file stops at visible logits."""
        raise AssertionError(f"sampler must not run, got {type(ising_mat).__name__}")


class RecordingEncoder(nn.Module):
    """Encoder that records every ``encode_tokens`` argument."""

    def __init__(self, vocab_size: int = 8, hidden_size: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        self.seen_tokens: list[torch.Tensor] = []
        self.seen_masks: list[torch.Tensor] = []

    def forward(self, tokens, attention_mask=None):
        """Delegate to ``encode_tokens`` so the module is concrete."""
        return self.encode_tokens(tokens, attention_mask=attention_mask)

    def encode_tokens(self, tokens, attention_mask=None):
        """Embed tokens and remember the objects that arrived."""
        self.seen_tokens.append(tokens)
        self.seen_masks.append(attention_mask)
        return self.embed(tokens)


class ParameterFreeEncoder(nn.Module):
    """Encoder with no parameters; outputs follow the token device."""

    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.seen_tokens: list[torch.Tensor] = []
        self.seen_masks: list[torch.Tensor] = []

    def forward(self, tokens, attention_mask=None):
        """Delegate to ``encode_tokens`` so the module is concrete."""
        return self.encode_tokens(tokens, attention_mask=attention_mask)

    def encode_tokens(self, tokens, attention_mask=None):
        """Broadcast token ids into a hidden-sized feature."""
        self.seen_tokens.append(tokens)
        self.seen_masks.append(attention_mask)
        values = tokens.to(dtype=torch.float32)
        expanded = values.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        return expanded.contiguous()


class RecordingEnergy(
    BMConditionedEnergyModel
):  # pylint: disable=too-few-public-methods
    """Energy model that records direct ``.to`` invocations."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__(
            encoder,
            bm_num_visible=3,
            bm_num_hidden=2,
            sampler=_ExplodingSampler(),
        )
        self.direct_to_calls: list[tuple] = []

    def to(self, *args, **kwargs):
        """Record the call, then move like any other module."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def _build(encoder: nn.Module | None = None) -> RecordingEnergy:
    """Build a tiny conditioned energy model around ``encoder``."""
    if encoder is None:
        encoder = RecordingEncoder()
    return RecordingEnergy(encoder)


def _nest(energy: nn.Module) -> nn.Module:
    """Return a parent whose ``_apply`` moves ``energy``."""
    parent = nn.Module()
    parent.energy = energy
    return parent


def _batch(device: torch.device = _CPU):
    """Return noncontiguous token batches and a boolean attention mask.

    The second sequence hides its last position so pooling is not a plain
    mean. Token ids stay inside the recording encoder's vocabulary.
    """
    noisy_table = torch.tensor(
        [
            [[1, 8], [2, 8], [3, 8], [0, 8], [1, 8]],
            [[4, 8], [5, 8], [6, 8], [7, 8], [0, 8]],
        ],
        dtype=torch.long,
        device=device,
    )
    candidate_table = torch.tensor(
        [
            [[7, 8], [6, 8], [5, 8], [4, 8], [3, 8]],
            [[1, 8], [0, 8], [2, 8], [3, 8], [4, 8]],
        ],
        dtype=torch.long,
        device=device,
    )
    mask_table = torch.ones(2, 5, 2, dtype=torch.bool, device=device)
    mask_table[1, 4, :] = False
    return noisy_table[:, :, 0], candidate_table[:, :, 0], mask_table[:, :, 0]


def _callers(device: torch.device = _CPU):
    """Return a batch plus clones and pointers taken before the call."""
    noisy, candidate, mask = _batch(device)
    before = (noisy.clone(), candidate.clone(), mask.clone())
    pointers = (noisy.data_ptr(), candidate.data_ptr(), mask.data_ptr())
    return noisy, candidate, mask, before, pointers


def _move_parameters(module: nn.Module, device: torch.device) -> None:
    """Move parameters the way a parent does: ``_apply``, not ``.to``."""
    module._apply(lambda tensor: tensor.to(device))


def _retarget_storage(device: torch.device):
    """Return an ``_apply`` callback that places new storage on ``device``.

    ``Module.to`` cannot copy out of ``meta``. Replacing storage still goes
    through ``_apply``, so the parameter device changes without ``.to()``.
    Zeros are readable, which is what a later CPU forward needs. The callback
    does not read the source tensor.
    """

    def _apply_fn(tensor: torch.Tensor) -> torch.Tensor:
        """Place ``tensor`` on ``device`` without reading its data."""
        if tensor.device == device:
            return tensor
        return torch.zeros(tensor.shape, dtype=tensor.dtype, device=device)

    return _apply_fn


def _logits(model: RecordingEnergy, noisy, candidate, mask) -> torch.Tensor:
    """Score one batch and keep only this call's encoder observations."""
    model.encoder.seen_tokens.clear()
    model.encoder.seen_masks.clear()
    return model.build_visible_logits(noisy, candidate, mask)


def _features(model: RecordingEnergy, noisy, candidate, mask) -> torch.Tensor:
    """Pool one batch and keep only this call's encoder observations."""
    model.encoder.seen_tokens.clear()
    model.encoder.seen_masks.clear()
    return model.build_conditioned_features(noisy, candidate, mask)


def _assert_unchanged(case, tensor, before, pointer, values=True) -> None:
    """Assert ``tensor`` kept its storage, device, dtype, and layout.

    ``values`` is false for meta tensors, which have no readable data.
    """
    case.assertEqual(tensor.data_ptr(), pointer)
    case.assertEqual(tensor.device, before.device)
    case.assertEqual(tensor.dtype, before.dtype)
    case.assertFalse(tensor.is_contiguous())
    if values:
        case.assertTrue(torch.equal(tensor, before))


def _assert_callers(case, tensors, before, pointers, values=True) -> None:
    """Assert noisy tokens, candidate tokens, and the mask stayed put."""
    for tensor, previous, pointer in zip(tensors, before, pointers):
        _assert_unchanged(case, tensor, previous, pointer, values=values)


def _assert_seen(case, encoder, device, same=None) -> None:
    """Both encodes must observe tokens and the mask on ``device``."""
    case.assertEqual(len(encoder.seen_tokens), 2)
    case.assertEqual(len(encoder.seen_masks), 2)
    for tokens, mask in zip(encoder.seen_tokens, encoder.seen_masks):
        case.assertEqual(tokens.device, device)
        case.assertEqual(mask.device, device)
    if same is not None:
        noisy, candidate, mask = same
        case.assertIs(encoder.seen_tokens[0], noisy)
        case.assertIs(encoder.seen_tokens[1], candidate)
        case.assertIs(encoder.seen_masks[0], mask)
        case.assertIs(encoder.seen_masks[1], mask)


def _assert_logits(case, logits, device, dtype, finite) -> None:
    """Assert logit device, dtype, shape, and, when requested, finiteness."""
    case.assertEqual(logits.device, device)
    case.assertEqual(logits.dtype, dtype)
    case.assertEqual(logits.shape, _LOGIT_SHAPE)
    if finite:
        case.assertTrue(torch.isfinite(logits).all().item())


def _assert_projector_grad(case, model, device) -> None:
    """Autograd must land on the projector and the encoder embedding."""
    weight = model.feature_projector.weight
    bias = model.feature_projector.bias
    embed = model.encoder.embed.weight
    for gradient in (weight.grad, bias.grad, embed.grad):
        case.assertIsNotNone(gradient)
        case.assertEqual(gradient.device, device)
        case.assertTrue(torch.isfinite(gradient).all().item())
        case.assertGreater(float(gradient.abs().sum()), 0.0)


def _assert_cannot_copy_out_of_meta(case) -> None:
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


class TestBmEnergyParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine, including CPU-only CI."""

    def test_cpu_logits_keep_caller_storage_and_identity(self):
        """An unmoved CPU model reuses the caller tensors and stays finite."""
        model = _build()
        noisy, candidate, mask, before, pointers = _callers()

        logits = _logits(model, noisy, candidate, mask)

        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, _CPU, same=(noisy, candidate, mask))
        _assert_logits(self, logits, _CPU, torch.float32, finite=True)
        self.assertEqual(_parameter_device(model.feature_projector, noisy), _CPU)
        self.assertEqual(noisy.dtype, torch.long)
        self.assertEqual(mask.dtype, torch.bool)

    def test_parent_to_meta_skips_energy_to(self):
        """Positional and keyword ``Module.to`` must not call the child ``.to``."""
        for kwargs in ({}, {"device": _META}):
            model = _build()
            parent = _nest(model)
            if kwargs:
                parent.to(**kwargs)
            else:
                parent.to(_META)
            self.assertEqual(model.direct_to_calls, [])
            self.assertEqual(next(model.encoder.parameters()).device, _META)
            self.assertEqual(next(model.feature_projector.parameters()).device, _META)

    def test_parent_apply_meta_ignores_stale_bm_device(self):
        """``parent._apply(meta)`` scores on meta and does not call ``.to``."""
        model = _build()
        parent = _nest(model)
        noisy, candidate, mask, before, pointers = _callers()

        _move_parameters(parent, _META)
        model.energy_bm.device = _CPU
        logits = _logits(model, noisy, candidate, mask)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(model.energy_bm.device, _CPU)
        self.assertEqual(next(model.encoder.parameters()).device, _META)
        self.assertEqual(next(model.feature_projector.parameters()).device, _META)
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, _META)
        self.assertIsNot(model.encoder.seen_tokens[0], noisy)
        self.assertIsNot(model.encoder.seen_masks[0], mask)
        _assert_logits(self, logits, _META, torch.float32, finite=False)

    def test_meta_batch_already_aligned_keeps_storage(self):
        """Tokens that already live on meta stay the objects the encoder sees."""
        model = _build()
        _move_parameters(model, _META)
        noisy, candidate, mask, before, pointers = _callers(_META)

        logits = _logits(model, noisy, candidate, mask)

        self.assertEqual(model.direct_to_calls, [])
        _assert_callers(self, (noisy, candidate, mask), before, pointers, values=False)
        _assert_seen(self, model.encoder, _META, same=(noisy, candidate, mask))
        _assert_logits(self, logits, _META, torch.float32, finite=False)

    def test_projector_only_move_keeps_encoder_on_caller_device(self):
        """Projector weights win even when the encoder and BM cache stay on CPU."""
        model = _build()
        noisy, candidate, mask, before, pointers = _callers()
        _move_parameters(model.feature_projector, _META)
        model.energy_bm.device = _CPU

        logits = _logits(model, noisy, candidate, mask)

        self.assertEqual(next(model.encoder.parameters()).device, _CPU)
        self.assertEqual(next(model.feature_projector.parameters()).device, _META)
        self.assertEqual(model.energy_bm.device, _CPU)
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, _CPU, same=(noisy, candidate, mask))
        _assert_logits(self, logits, _META, torch.float32, finite=False)

    def test_encoder_only_move_places_features_on_encoder_device(self):
        """Pooled features follow the encoder; the CPU projector cannot host meta."""
        model = _build()
        noisy, candidate, mask, before, pointers = _callers()
        _move_parameters(model.encoder, _META)
        model.energy_bm.device = _CPU

        features = _features(model, noisy, candidate, mask)

        self.assertEqual(features.device, _META)
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(features.shape, _FEATURE_SHAPE)
        self.assertEqual(next(model.feature_projector.parameters()).device, _CPU)
        self.assertEqual(model.energy_bm.device, _CPU)
        _assert_seen(self, model.encoder, _META)
        self.assertIsNot(model.encoder.seen_tokens[0], noisy)
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        with self.assertRaises(NotImplementedError) as caught:
            model.build_visible_logits(noisy, candidate, mask)
        self.assertIn("meta", str(caught.exception).lower())
        self.assertEqual(next(model.feature_projector.parameters()).device, _CPU)
        _assert_callers(self, (noisy, candidate, mask), before, pointers)

    def test_stale_bm_device_does_not_move_features(self):
        """A lying BM cache must not take CPU features onto ``meta``."""
        model = _build()
        noisy, candidate, mask, before, pointers = _callers()
        model.energy_bm.device = _META

        logits = _logits(model, noisy, candidate, mask)

        self.assertEqual(model.energy_bm.device, _META)
        self.assertEqual(next(model.feature_projector.parameters()).device, _CPU)
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, _CPU, same=(noisy, candidate, mask))
        _assert_logits(self, logits, _CPU, torch.float32, finite=True)

    def test_parent_dtype_only_move_keeps_cpu_and_skips_child_to(self):
        """``parent.to(dtype=...)`` keeps CPU and does not call the child ``.to``."""
        model = _build()
        parent = _nest(model)
        parent.to(dtype=torch.float64)
        noisy, candidate, mask, before, pointers = _callers()

        logits = _logits(model, noisy, candidate, mask)

        self.assertEqual(model.direct_to_calls, [])
        weight = next(model.feature_projector.parameters())
        self.assertEqual(weight.dtype, torch.float64)
        self.assertEqual(weight.device, _CPU)
        self.assertEqual(next(model.encoder.parameters()).dtype, torch.float64)
        self.assertEqual(next(model.encoder.parameters()).device, _CPU)
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, _CPU, same=(noisy, candidate, mask))
        _assert_logits(self, logits, _CPU, torch.float64, finite=True)
        self.assertEqual(noisy.dtype, torch.long)
        self.assertEqual(mask.dtype, torch.bool)

    def test_direct_dtype_only_stays_on_cpu(self):
        """A direct float64 cast stays on CPU and leaves the caller batch put."""
        model = _build()
        model.to(dtype=torch.float64)
        noisy, candidate, mask, before, pointers = _callers()

        logits = _logits(model, noisy, candidate, mask)

        self.assertEqual(model.direct_to_calls[0][1]["dtype"], torch.float64)
        self.assertEqual(next(model.feature_projector.parameters()).device, _CPU)
        self.assertEqual(
            next(model.feature_projector.parameters()).dtype, torch.float64
        )
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, _CPU, same=(noisy, candidate, mask))
        _assert_logits(self, logits, _CPU, torch.float64, finite=True)

    def test_float64_parameters_survive_meta_apply(self):
        """Moving float64 weights onto meta must not recast or rewrite callers."""
        model = _build()
        parent = _nest(model)
        parent.to(dtype=torch.float64)
        _move_parameters(parent, _META)
        noisy, candidate, mask, before, pointers = _callers()

        logits = _logits(model, noisy, candidate, mask)

        self.assertEqual(model.direct_to_calls, [])
        weight = next(model.feature_projector.parameters())
        self.assertEqual(weight.dtype, torch.float64)
        self.assertEqual(weight.device, _META)
        self.assertEqual(next(model.encoder.parameters()).dtype, torch.float64)
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, _META)
        _assert_logits(self, logits, _META, torch.float64, finite=False)
        self.assertEqual(noisy.dtype, torch.long)

    def test_parameter_device_is_reread_after_meta_round_trip(self):
        """Each move is reread. ``.to(cpu)`` cannot copy out of meta.

        ``_apply`` still retargets storage, and the next logits follow that
        device. ``BMConditionedEnergyModel.to`` is not called.
        """
        _assert_cannot_copy_out_of_meta(self)
        model = _build()
        parent = _nest(model)
        noisy, candidate, mask, before, pointers = _callers()

        logits = _logits(model, noisy, candidate, mask)
        self.assertEqual(_parameter_device(model.feature_projector, noisy), _CPU)
        _assert_logits(self, logits, _CPU, torch.float32, finite=True)

        _move_parameters(parent, _META)
        self.assertEqual(_parameter_device(model.encoder, noisy), _META)
        logits = _logits(model, noisy, candidate, mask)
        self.assertEqual(model.direct_to_calls, [])
        _assert_seen(self, model.encoder, _META)
        _assert_logits(self, logits, _META, torch.float32, finite=False)

        with self.assertRaises(NotImplementedError) as caught:
            parent.to(_CPU)
        self.assertIn("meta", str(caught.exception).lower())
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(next(model.feature_projector.parameters()).device, _META)

        parent._apply(_retarget_storage(_CPU))
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(_parameter_device(model.encoder, noisy), _CPU)
        logits = _logits(model, noisy, candidate, mask)
        self.assertEqual(next(model.feature_projector.parameters()).device, _CPU)
        _assert_seen(self, model.encoder, _CPU, same=(noisy, candidate, mask))
        _assert_logits(self, logits, _CPU, torch.float32, finite=True)

        _move_parameters(parent, _META)
        logits = _logits(model, noisy, candidate, mask)
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(_parameter_device(model.feature_projector, noisy), _META)
        _assert_seen(self, model.encoder, _META)
        _assert_logits(self, logits, _META, torch.float32, finite=False)
        _assert_callers(self, (noisy, candidate, mask), before, pointers)

    def test_parameter_device_helper_rereads_parameters(self):
        """Empty modules fall back; a later ``_apply`` wins over that fallback."""
        empty = ParameterFreeEncoder()
        cpu_tokens = torch.zeros(2, 3, dtype=torch.long)
        meta_tokens = torch.zeros(2, 3, dtype=torch.long, device=_META)
        self.assertIsNone(next(empty.parameters(), None))
        self.assertEqual(_parameter_device(empty, cpu_tokens), _CPU)
        self.assertEqual(_parameter_device(empty, meta_tokens), _META)

        model = _build()
        self.assertEqual(_parameter_device(model.encoder, meta_tokens), _CPU)
        self.assertEqual(_parameter_device(model.feature_projector, cpu_tokens), _CPU)
        _move_parameters(model.encoder, _META)
        self.assertEqual(_parameter_device(model.encoder, cpu_tokens), _META)
        self.assertEqual(_parameter_device(model.feature_projector, meta_tokens), _CPU)

    def test_parameter_free_encoder_follows_tokens_then_projector(self):
        """With no encoder weights, tokens stay put and logits follow the projector."""
        model = _build(ParameterFreeEncoder())
        self.assertEqual(list(model.encoder.parameters()), [])
        noisy, candidate, mask, before, pointers = _callers()
        model.energy_bm.device = _META

        logits = _logits(model, noisy, candidate, mask)

        self.assertEqual(model.energy_bm.device, _META)
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, _CPU, same=(noisy, candidate, mask))
        _assert_logits(self, logits, _CPU, torch.float32, finite=True)

        _move_parameters(model.feature_projector, _META)
        model.energy_bm.device = _CPU
        logits = _logits(model, noisy, candidate, mask)
        self.assertIs(model.encoder.seen_tokens[0], noisy)
        self.assertIs(model.encoder.seen_masks[0], mask)
        self.assertEqual(model.energy_bm.device, _CPU)
        self.assertEqual(list(model.encoder.parameters()), [])
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_logits(self, logits, _META, torch.float32, finite=False)

        meta_model = _build(ParameterFreeEncoder())
        _move_parameters(meta_model.feature_projector, _META)
        meta_noisy, meta_candidate, meta_mask, meta_before, meta_pointers = _callers(
            _META
        )
        logits = _logits(meta_model, meta_noisy, meta_candidate, meta_mask)
        _assert_callers(
            self,
            (meta_noisy, meta_candidate, meta_mask),
            meta_before,
            meta_pointers,
            values=False,
        )
        _assert_seen(
            self,
            meta_model.encoder,
            _META,
            same=(meta_noisy, meta_candidate, meta_mask),
        )
        _assert_logits(self, logits, _META, torch.float32, finite=False)

    def test_same_device_backward_reaches_projector(self):
        """A CPU batch still trains the projector weight, bias, and embedding."""
        torch.manual_seed(0)
        model = _build()
        noisy, candidate, mask, _before, pointers = _callers()

        logits = _logits(model, noisy, candidate, mask)
        self.assertTrue(logits.requires_grad)
        logits.sum().backward()

        self.assertEqual(noisy.data_ptr(), pointers[0])
        self.assertEqual(candidate.data_ptr(), pointers[1])
        self.assertEqual(mask.data_ptr(), pointers[2])
        _assert_logits(self, logits, _CPU, torch.float32, finite=True)
        _assert_projector_grad(self, model, _CPU)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBmEnergyParameterDeviceCuda(unittest.TestCase):
    """Parent moves onto CUDA. Skipped on machines without a GPU."""

    def test_parent_apply_accepts_cpu_batch(self):
        """CPU tokens match a CPU copy after ``parent._apply`` onto CUDA."""
        model = _build()
        reference = copy.deepcopy(model)
        parent = _nest(model)
        cuda = torch.device("cuda")
        noisy, candidate, mask, before, pointers = _callers()

        _move_parameters(parent, cuda)
        model.energy_bm.device = _CPU
        logits = _logits(model, noisy, candidate, mask)
        expected = reference.build_visible_logits(noisy, candidate, mask)

        device = next(model.feature_projector.parameters()).device
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(model.energy_bm.device, _CPU)
        self.assertEqual(device.type, "cuda")
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, device)
        self.assertIsNot(model.encoder.seen_tokens[0], noisy)
        _assert_logits(self, logits, device, torch.float32, finite=True)
        torch.testing.assert_close(logits.cpu(), expected)

    def test_batch_already_on_cuda_keeps_storage(self):
        """A CUDA batch that already matches the weights keeps its storage."""
        model = _build()
        reference = copy.deepcopy(model)
        cuda = torch.device("cuda")
        _move_parameters(model, cuda)
        noisy, candidate, mask, before, pointers = _callers(cuda)
        cpu_noisy = noisy.detach().cpu()
        cpu_candidate = candidate.detach().cpu()
        cpu_mask = mask.detach().cpu()

        logits = _logits(model, noisy, candidate, mask)
        expected = reference.build_visible_logits(cpu_noisy, cpu_candidate, cpu_mask)

        device = next(model.encoder.parameters()).device
        self.assertEqual(model.direct_to_calls, [])
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, device, same=(noisy, candidate, mask))
        _assert_logits(self, logits, device, torch.float32, finite=True)
        torch.testing.assert_close(logits.cpu(), expected)

    def test_round_trip_is_reread(self):
        """CUDA → CPU → CUDA follows parameters and does not call ``.to``."""
        model = _build()
        reference = copy.deepcopy(model)
        parent = _nest(model)
        noisy, candidate, mask, before, pointers = _callers()
        expected = reference.build_visible_logits(noisy, candidate, mask)
        cuda = torch.device("cuda")

        _move_parameters(parent, cuda)
        logits = _logits(model, noisy, candidate, mask)
        device = next(model.feature_projector.parameters()).device
        self.assertEqual(device.type, "cuda")
        _assert_seen(self, model.encoder, device)
        _assert_logits(self, logits, device, torch.float32, finite=True)
        torch.testing.assert_close(logits.cpu(), expected)

        _move_parameters(parent, _CPU)
        logits = _logits(model, noisy, candidate, mask)
        self.assertEqual(_parameter_device(model.encoder, noisy), _CPU)
        _assert_seen(self, model.encoder, _CPU, same=(noisy, candidate, mask))
        _assert_logits(self, logits, _CPU, torch.float32, finite=True)
        torch.testing.assert_close(logits, expected)

        _move_parameters(parent, cuda)
        logits = _logits(model, noisy, candidate, mask)
        device = next(model.feature_projector.parameters()).device
        self.assertEqual(model.direct_to_calls, [])
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, device)
        _assert_logits(self, logits, device, torch.float32, finite=True)
        torch.testing.assert_close(logits.cpu(), expected)

    def test_float64_cuda_move_keeps_dtype(self):
        """A float64 CUDA move must not downcast logits or the caller tokens."""
        model = _build()
        parent = _nest(model)
        parent.to(dtype=torch.float64)
        reference = copy.deepcopy(model)
        noisy, candidate, mask, before, pointers = _callers()
        _move_parameters(parent, torch.device("cuda"))

        logits = _logits(model, noisy, candidate, mask)
        expected = reference.build_visible_logits(noisy, candidate, mask)

        device = next(model.feature_projector.parameters()).device
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(device.type, "cuda")
        self.assertEqual(next(model.encoder.parameters()).dtype, torch.float64)
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_logits(self, logits, device, torch.float64, finite=True)
        self.assertEqual(noisy.dtype, torch.long)
        torch.testing.assert_close(logits.cpu(), expected)

    def test_dtype_only_move_keeps_cuda_device(self):
        """``.to(dtype=...)`` on CUDA keeps CUDA and does not call child ``.to``."""
        model = _build()
        parent = _nest(model)
        _move_parameters(parent, torch.device("cuda"))
        parent.to(dtype=torch.float64)
        noisy, candidate, mask, before, pointers = _callers()

        logits = _logits(model, noisy, candidate, mask)

        device = next(model.feature_projector.parameters()).device
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(device.type, "cuda")
        self.assertEqual(next(model.encoder.parameters()).dtype, torch.float64)
        self.assertEqual(
            next(model.feature_projector.parameters()).dtype, torch.float64
        )
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_logits(self, logits, device, torch.float64, finite=True)

    def test_backward_from_cpu_batch_reaches_cuda_projector(self):
        """CPU caller storage stays put while projector gradients land on CUDA."""
        torch.manual_seed(0)
        model = _build()
        parent = _nest(model)
        _move_parameters(parent, torch.device("cuda"))
        noisy, candidate, mask, _before, pointers = _callers()

        logits = _logits(model, noisy, candidate, mask)
        logits.sum().backward()

        device = model.feature_projector.weight.device
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(noisy.data_ptr(), pointers[0])
        self.assertEqual(mask.data_ptr(), pointers[2])
        self.assertEqual(noisy.device, _CPU)
        self.assertEqual(logits.device, device)
        _assert_projector_grad(self, model, device)

    def test_cuda_batch_follows_cpu_parameters(self):
        """A CUDA batch is encoded on the CPU parameter device."""
        model = _build()
        reference = copy.deepcopy(model)
        noisy, candidate, mask, before, pointers = _callers(torch.device("cuda"))
        cpu_noisy = noisy.detach().cpu()
        cpu_candidate = candidate.detach().cpu()
        cpu_mask = mask.detach().cpu()

        logits = _logits(model, noisy, candidate, mask)
        expected = reference.build_visible_logits(cpu_noisy, cpu_candidate, cpu_mask)

        self.assertEqual(noisy.data_ptr(), pointers[0])
        self.assertEqual(mask.data_ptr(), pointers[2])
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, _CPU)
        self.assertIsNot(model.encoder.seen_tokens[0], noisy)
        self.assertEqual(next(model.feature_projector.parameters()).device, _CPU)
        _assert_logits(self, logits, _CPU, torch.float32, finite=True)
        torch.testing.assert_close(logits, expected)

    def test_encoder_only_cuda_move_projects_on_cpu(self):
        """Encoder-only CUDA features are projected on the CPU projector."""
        model = _build()
        reference = copy.deepcopy(model)
        noisy, candidate, mask, before, pointers = _callers()
        _move_parameters(model.encoder, torch.device("cuda"))
        model.energy_bm.device = _META

        features = _features(model, noisy, candidate, mask)
        logits = _logits(model, noisy, candidate, mask)
        expected = reference.build_visible_logits(noisy, candidate, mask)

        encoder_device = next(model.encoder.parameters()).device
        self.assertEqual(encoder_device.type, "cuda")
        self.assertEqual(features.device, encoder_device)
        self.assertEqual(features.shape, _FEATURE_SHAPE)
        self.assertEqual(next(model.feature_projector.parameters()).device, _CPU)
        self.assertEqual(model.energy_bm.device, _META)
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, encoder_device)
        _assert_logits(self, logits, _CPU, torch.float32, finite=True)
        torch.testing.assert_close(logits, expected)

    def test_projector_only_cuda_move(self):
        """A CUDA projector does not drag encoder inputs off the caller device."""
        model = _build()
        reference = copy.deepcopy(model)
        noisy, candidate, mask, before, pointers = _callers()
        _move_parameters(model.feature_projector, torch.device("cuda"))
        model.energy_bm.device = _CPU

        logits = _logits(model, noisy, candidate, mask)
        expected = reference.build_visible_logits(noisy, candidate, mask)

        device = next(model.feature_projector.parameters()).device
        self.assertEqual(device.type, "cuda")
        self.assertEqual(next(model.encoder.parameters()).device, _CPU)
        self.assertEqual(model.energy_bm.device, _CPU)
        _assert_callers(self, (noisy, candidate, mask), before, pointers)
        _assert_seen(self, model.encoder, _CPU, same=(noisy, candidate, mask))
        _assert_logits(self, logits, device, torch.float32, finite=True)
        torch.testing.assert_close(logits.cpu(), expected)


if __name__ == "__main__":
    unittest.main()
