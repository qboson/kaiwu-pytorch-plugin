"""Regression tests for DPLM ESM attention-mask device placement.

``_ModifiedEsmModel.forward`` builds or forwards an attention mask, token ids,
position ids, and optional fused embeddings into ``EsmEmbeddings``. Those
embeddings multiply the mask, index the word and position tables, and, when
token dropout is on, compare the token ids with the activations. A parent
``nn.Module.to`` moves the weights through ``_apply`` and does not call the
child's ``.to`` (merged #164 / ``tests/test_bm_device_sync.py``), so a CPU
mask or a CPU id meets parameters that already moved:

``RuntimeError: Tensor on device meta is not on the expected device cpu!``

The live embedding parameter device is reread on every call. Caller tensors
are not modified in place, dtypes stay put, and a same-device ``.to`` keeps
the caller's object. ``.to(cpu)`` cannot copy out of a meta tensor, so a meta
round trip retargets storage with ``_apply``.

Out of scope: ``bm.py`` conditioned energy (#1375), ``common.py`` pooling
(#259), example generator token placement (#1292), and library
``qdiffusion.py`` (#310 / #488).
"""

# Parent-initiated moves go through ``_apply`` (see merged #164).
# pylint: disable=protected-access,too-few-public-methods

from __future__ import annotations

import copy
import importlib.util
import unittest
import warnings
from pathlib import Path

import torch
from torch import nn

warnings.filterwarnings("ignore", message=r".*GenerationMixin.*")

try:
    from transformers import EsmConfig
except ImportError:  # pragma: no cover - optional example dependency
    EsmConfig = None


def _load_esm_patch():
    """Load the example module without importing the DPLM package init."""
    path = (
        Path(__file__).resolve().parents[1]
        / "example"
        / "qdiffusion"
        / "dplm"
        / "models"
        / "esm_patch.py"
    )
    spec = importlib.util.spec_from_file_location("esm_patch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if EsmConfig is not None:
    _ESM_PATCH = _load_esm_patch()

    class RecordingEsmModel(_ESM_PATCH._ModifiedEsmModel):
        """ESM backbone that records direct ``.to`` calls."""

        def __init__(self, config) -> None:
            super().__init__(config, add_pooling_layer=False)
            self.direct_to_calls: list[tuple] = []

        def to(self, *args, **kwargs):
            """Record the call, then perform the real move."""
            self.direct_to_calls.append((args, kwargs))
            return super().to(*args, **kwargs)

else:  # pragma: no cover - optional example dependency
    _ESM_PATCH = None

    class RecordingEsmModel(nn.Module):
        """Placeholder so collection succeeds without transformers."""

        direct_to_calls: list[tuple] = []

        def forward(self, *_args, **_kwargs):  # pragma: no cover
            """Unused stand-in. Tests that need the real model are skipped."""
            raise RuntimeError("transformers is required")


_CPU = torch.device("cpu")
_META = torch.device("meta")
_HIDDEN = 16
_SHAPE = (2, 5, _HIDDEN)


def _tiny_config(token_dropout: bool = False):
    """Return a CPU-cheap ESM config with absolute position embeddings."""
    return EsmConfig(
        vocab_size=20,
        hidden_size=_HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        pad_token_id=0,
        mask_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
        token_dropout=token_dropout,
        emb_layer_norm_before=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


def _cpu_model(token_dropout: bool = False) -> RecordingEsmModel:
    """Build one eval-mode tiny backbone on CPU."""
    model = RecordingEsmModel(_tiny_config(token_dropout=token_dropout))
    model.eval()
    return model


def _nest(model: nn.Module) -> nn.Module:
    """Return a parent whose ``_apply`` moves ``model``."""
    parent = nn.Module()
    parent.model = model
    return parent


def _move_parameters(module: nn.Module, device: torch.device) -> None:
    """Move parameters the way a parent does: ``_apply``, not ``.to``."""
    module._apply(lambda tensor: tensor.to(device))


def _retarget_storage(device: torch.device):
    """Return an ``_apply`` callback that places new storage on ``device``.

    ``Module.to`` cannot copy out of ``meta``. Replacing storage still goes
    through ``_apply``, so the parameter device changes without ``.to``.
    Zeros are readable, which is what a later CPU forward needs. The callback
    does not read the source tensor.
    """

    def _apply_fn(tensor: torch.Tensor) -> torch.Tensor:
        """Place ``tensor`` on ``device`` without reading its data."""
        if tensor.device == device:
            return tensor
        return torch.zeros(tensor.shape, dtype=tensor.dtype, device=device)

    return _apply_fn


def _callers(device: torch.device = _CPU, embed_dtype: torch.dtype = torch.float32):
    """Return noncontiguous tokens, a mask, fused embeds, and position ids.

    The second sequence hides its last position. Token ids stay inside the
    tiny vocabulary and include the mask id so token dropout has work to do.
    """
    token_table = torch.tensor(
        [
            [[2, 9], [1, 9], [6, 9], [7, 9], [3, 9]],
            [[4, 9], [5, 9], [8, 9], [1, 9], [2, 9]],
        ],
        dtype=torch.long,
        device=device,
    )
    mask_table = torch.ones(2, 5, 2, dtype=torch.bool, device=device)
    mask_table[1, 4, :] = False
    embed_table = torch.randn(2, 5, 2, _HIDDEN, dtype=embed_dtype, device=device)
    position_table = torch.tensor(
        [
            [[1, 9], [2, 9], [3, 9], [4, 9], [5, 9]],
            [[1, 9], [2, 9], [3, 9], [4, 9], [6, 9]],
        ],
        dtype=torch.long,
        device=device,
    )
    tokens = token_table[:, :, 0]
    mask = mask_table[:, :, 0]
    embeds = embed_table[:, :, 0, :]
    position_ids = position_table[:, :, 0]
    tensors = (tokens, mask, embeds, position_ids)
    before = tuple(tensor.clone() for tensor in tensors)
    pointers = tuple(tensor.data_ptr() for tensor in tensors)
    return tokens, mask, embeds, position_ids, before, pointers


def _install_spy(model: RecordingEsmModel) -> dict:
    """Record the tensors ``EsmEmbeddings.forward`` actually receives."""
    seen: dict = {}
    original = model.embeddings.forward

    def _wrapped(*args, **kwargs):
        """Store embedding inputs, then run the real embedding."""
        del args
        seen["mask"] = kwargs.get("attention_mask")
        seen["inputs_embeds"] = kwargs.get("inputs_embeds")
        seen["input_ids"] = kwargs.get("input_ids")
        seen["position_ids"] = kwargs.get("position_ids")
        return original(**kwargs)

    model.embeddings.forward = _wrapped
    return seen


def _forward(
    model: RecordingEsmModel,
    tokens=None,
    mask=None,
    embeds=None,
    position_ids=None,
) -> torch.Tensor:
    """Run one forward and return the last hidden state."""
    return model(
        input_ids=tokens,
        attention_mask=mask,
        inputs_embeds=embeds,
        position_ids=position_ids,
        return_dict=True,
    ).last_hidden_state


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


def _assert_pair(case, tensors, before, pointers, values=True) -> None:
    """Assert the two caller tensors in ``tensors`` stayed put."""
    for tensor, previous, pointer in zip(tensors, before, pointers):
        _assert_unchanged(case, tensor, previous, pointer, values=values)


def _assert_hidden(case, hidden, device, dtype, finite: bool) -> None:
    """Assert hidden device, dtype, shape, and, when requested, finiteness."""
    case.assertEqual(hidden.device, device)
    case.assertEqual(hidden.dtype, dtype)
    case.assertEqual(hidden.shape, _SHAPE)
    if finite:
        case.assertTrue(torch.isfinite(hidden).all().item())


def _assert_position_grad(case, model, device) -> None:
    """Autograd must land on the position-embedding weight."""
    gradient = model.embeddings.position_embeddings.weight.grad
    case.assertIsNotNone(gradient)
    case.assertEqual(gradient.device, device)
    case.assertTrue(torch.isfinite(gradient).all().item())
    case.assertGreater(float(gradient.abs().sum()), 0.0)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare an accelerator result with the CPU reference."""
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-4, atol=1e-5)


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


@unittest.skipUnless(EsmConfig is not None, "requires transformers")
class TestEsmMaskParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine, including CPU-only CI."""

    def test_cpu_forward_keeps_caller_storage_and_identity(self):
        """An unmoved CPU model reuses the caller tensors and stays finite."""
        model = _cpu_model()
        tokens, mask, embeds, position_ids, before, pointers = _callers()
        seen = _install_spy(model)

        hidden = _forward(model, tokens, mask, embeds, position_ids)

        _assert_pair(self, (tokens, mask), before[:2], pointers[:2])
        _assert_pair(self, (embeds, position_ids), before[2:], pointers[2:])
        self.assertIs(seen["input_ids"], tokens)
        self.assertIs(seen["mask"], mask)
        self.assertIs(seen["inputs_embeds"], embeds)
        self.assertIs(seen["position_ids"], position_ids)
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(_ESM_PATCH._parameter_device(model.embeddings, tokens), _CPU)
        _assert_hidden(self, hidden, _CPU, torch.float32, finite=True)

    def test_parent_to_meta_skips_child_to(self):
        """Positional and keyword ``Module.to`` must not call the child ``.to``."""
        for kwargs in ({}, {"device": _META}):
            with self.subTest(kwargs=bool(kwargs)):
                model = _cpu_model()
                parent = _nest(model)
                if kwargs:
                    parent.to(**kwargs)
                else:
                    parent.to(_META)
                word = model.embeddings.word_embeddings.weight
                position = model.embeddings.position_embeddings.weight
                self.assertEqual(model.direct_to_calls, [])
                self.assertEqual(word.device, _META)
                self.assertEqual(position.device, _META)

    def test_parent_apply_accepts_cpu_token_ids(self):
        """CPU token ids run after ``parent._apply(meta)`` without ``.to``.

        Token dropout is the path that compares those ids with the activations.
        Meta weights accept a foreign index, so this still failed before ids
        were aligned. The generated mask is float32 on the parameter device.
        """
        for token_dropout in (False, True):
            with self.subTest(token_dropout=token_dropout):
                model = _cpu_model(token_dropout=token_dropout)
                parent = _nest(model)
                tokens, _mask, _embeds, _position_ids, before, pointers = _callers()
                _move_parameters(parent, _META)
                seen = _install_spy(model)

                hidden = _forward(model, tokens=tokens)

                self.assertEqual(model.direct_to_calls, [])
                self.assertEqual(
                    model.embeddings.position_embeddings.weight.device, _META
                )
                _assert_unchanged(self, tokens, before[0], pointers[0])
                self.assertEqual(seen["input_ids"].device, _META)
                self.assertEqual(seen["input_ids"].dtype, torch.long)
                self.assertIsNot(seen["input_ids"], tokens)
                self.assertEqual(seen["mask"].device, _META)
                self.assertEqual(seen["mask"].dtype, torch.float32)
                self.assertEqual(seen["mask"].shape, (2, 5))
                self.assertIsNone(seen["inputs_embeds"])
                self.assertIsNone(seen["position_ids"])
                _assert_hidden(self, hidden, _META, torch.float32, finite=False)

    def test_provided_cpu_mask_follows_parameters_not_the_caller(self):
        """A caller mask stays on CPU while the embedding mask is ``meta``."""
        model = _cpu_model()
        tokens, mask, _embeds, _position_ids, before, pointers = _callers()
        _move_parameters(_nest(model), _META)
        seen = _install_spy(model)

        hidden = _forward(model, tokens, mask)

        self.assertEqual(model.direct_to_calls, [])
        _assert_pair(self, (tokens, mask), before[:2], pointers[:2])
        self.assertEqual(seen["mask"].device, _META)
        self.assertEqual(seen["mask"].dtype, torch.bool)
        self.assertIsNot(seen["mask"], mask)
        self.assertEqual(seen["input_ids"].device, _META)
        self.assertIsNot(seen["input_ids"], tokens)
        _assert_hidden(self, hidden, _META, torch.float32, finite=False)

    def test_cpu_inputs_embeds_follow_position_embedding_parameters(self):
        """Fused embeddings must meet position weights on one device."""
        model = _cpu_model()
        _tokens, mask, embeds, _position_ids, before, pointers = _callers()
        _move_parameters(_nest(model), _META)
        seen = _install_spy(model)

        hidden = _forward(model, mask=mask, embeds=embeds)

        self.assertEqual(model.direct_to_calls, [])
        _assert_pair(
            self, (embeds, mask), (before[2], before[1]), (pointers[2], pointers[1])
        )
        self.assertEqual(seen["inputs_embeds"].device, _META)
        self.assertEqual(seen["inputs_embeds"].dtype, torch.float32)
        self.assertIsNot(seen["inputs_embeds"], embeds)
        self.assertEqual(seen["mask"].device, _META)
        self.assertIsNot(seen["mask"], mask)
        self.assertIsNone(seen["input_ids"])
        _assert_hidden(self, hidden, _META, torch.float32, finite=False)

    def test_explicit_cpu_position_ids_follow_parameters(self):
        """Caller position ids stay put while the embedding index is ``meta``."""
        model = _cpu_model()
        tokens, _mask, _embeds, position_ids, before, pointers = _callers()
        _move_parameters(_nest(model), _META)
        seen = _install_spy(model)

        hidden = _forward(model, tokens=tokens, position_ids=position_ids)

        self.assertEqual(model.direct_to_calls, [])
        _assert_pair(
            self,
            (tokens, position_ids),
            (before[0], before[3]),
            (pointers[0], pointers[3]),
        )
        self.assertEqual(seen["position_ids"].device, _META)
        self.assertEqual(seen["position_ids"].dtype, torch.long)
        self.assertIsNot(seen["position_ids"], position_ids)
        self.assertEqual(seen["input_ids"].device, _META)
        _assert_hidden(self, hidden, _META, torch.float32, finite=False)

    def test_meta_batch_already_aligned_keeps_storage(self):
        """A meta batch that already matches the weights keeps its storage."""
        model = _cpu_model()
        _move_parameters(model, _META)
        tokens, mask, embeds, position_ids, before, pointers = _callers(_META)
        seen = _install_spy(model)

        hidden = _forward(model, tokens, mask, embeds, position_ids)

        self.assertEqual(model.direct_to_calls, [])
        _assert_pair(self, (tokens, mask), before[:2], pointers[:2], values=False)
        _assert_pair(
            self, (embeds, position_ids), before[2:], pointers[2:], values=False
        )
        self.assertIs(seen["input_ids"], tokens)
        self.assertIs(seen["mask"], mask)
        self.assertIs(seen["inputs_embeds"], embeds)
        self.assertIs(seen["position_ids"], position_ids)
        _assert_hidden(self, hidden, _META, torch.float32, finite=False)

    def test_parent_dtype_only_move_keeps_cpu_and_skips_child_to(self):
        """``parent.to(dtype=...)`` keeps CPU and does not call the child ``.to``."""
        model = _cpu_model()
        parent = _nest(model)
        parent.to(dtype=torch.float64)
        tokens, mask, _embeds, _position_ids, before, pointers = _callers()
        seen = _install_spy(model)

        hidden = _forward(model, tokens, mask)

        word = model.embeddings.word_embeddings.weight
        position = model.embeddings.position_embeddings.weight
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(word.dtype, torch.float64)
        self.assertEqual(word.device, _CPU)
        self.assertEqual(position.dtype, torch.float64)
        self.assertEqual(position.device, _CPU)
        _assert_pair(self, (tokens, mask), before[:2], pointers[:2])
        self.assertIs(seen["input_ids"], tokens)
        self.assertIs(seen["mask"], mask)
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(mask.dtype, torch.bool)
        _assert_hidden(self, hidden, _CPU, torch.float64, finite=True)

    def test_direct_dtype_only_stays_on_cpu(self):
        """A direct float64 cast stays on CPU and leaves the caller batch put."""
        model = _cpu_model()
        model.to(dtype=torch.float64)
        tokens, mask, _embeds, _position_ids, before, pointers = _callers()

        hidden = _forward(model, tokens, mask)

        self.assertEqual(model.direct_to_calls[0][1]["dtype"], torch.float64)
        position = model.embeddings.position_embeddings.weight
        self.assertEqual(position.device, _CPU)
        self.assertEqual(position.dtype, torch.float64)
        _assert_pair(self, (tokens, mask), before[:2], pointers[:2])
        _assert_hidden(self, hidden, _CPU, torch.float64, finite=True)

    def test_float64_parameters_survive_meta_apply(self):
        """Moving float64 weights onto meta must not recast or rewrite callers."""
        model = _cpu_model()
        parent = _nest(model)
        parent.to(dtype=torch.float64)
        _move_parameters(parent, _META)
        tokens, mask, _embeds, _position_ids, before, pointers = _callers()
        seen = _install_spy(model)

        hidden = _forward(model, tokens, mask)

        position = model.embeddings.position_embeddings.weight
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(position.dtype, torch.float64)
        self.assertEqual(position.device, _META)
        self.assertEqual(model.embeddings.word_embeddings.weight.dtype, torch.float64)
        _assert_pair(self, (tokens, mask), before[:2], pointers[:2])
        self.assertEqual(seen["input_ids"].device, _META)
        self.assertEqual(seen["input_ids"].dtype, torch.long)
        self.assertIsNot(seen["input_ids"], tokens)
        self.assertEqual(seen["mask"].dtype, torch.bool)
        self.assertEqual(tokens.dtype, torch.long)
        _assert_hidden(self, hidden, _META, torch.float64, finite=False)

    def test_parameter_device_is_reread_after_leaving_meta(self):
        """Each move is reread. ``.to(cpu)`` cannot copy out of meta.

        ``_apply`` still retargets storage, and the next hidden state follows
        that device. ``_ModifiedEsmModel.to`` is not called.
        """
        _assert_cannot_copy_out_of_meta(self)
        model = _cpu_model()
        parent = _nest(model)
        tokens, mask, _embeds, _position_ids, before, pointers = _callers()
        seen = _install_spy(model)

        hidden = _forward(model, tokens, mask)
        self.assertEqual(_ESM_PATCH._parameter_device(model.embeddings, tokens), _CPU)
        self.assertIs(seen["input_ids"], tokens)
        self.assertIs(seen["mask"], mask)
        _assert_hidden(self, hidden, _CPU, torch.float32, finite=True)

        _move_parameters(parent, _META)
        hidden = _forward(model, tokens, mask)
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(_ESM_PATCH._parameter_device(model.embeddings, tokens), _META)
        self.assertEqual(seen["input_ids"].device, _META)
        self.assertIsNot(seen["input_ids"], tokens)
        self.assertIsNot(seen["mask"], mask)
        _assert_hidden(self, hidden, _META, torch.float32, finite=False)

        with self.assertRaises(NotImplementedError) as caught:
            parent.to(_CPU)
        self.assertIn("meta", str(caught.exception).lower())
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(model.embeddings.position_embeddings.weight.device, _META)

        parent._apply(_retarget_storage(_CPU))
        hidden = _forward(model, tokens, mask)
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(_ESM_PATCH._parameter_device(model.embeddings, tokens), _CPU)
        self.assertIs(seen["input_ids"], tokens)
        self.assertIs(seen["mask"], mask)
        _assert_hidden(self, hidden, _CPU, torch.float32, finite=True)

        _move_parameters(parent, _META)
        hidden = _forward(model, tokens, mask)
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(_ESM_PATCH._parameter_device(model.embeddings, tokens), _META)
        self.assertIsNot(seen["mask"], mask)
        _assert_hidden(self, hidden, _META, torch.float32, finite=False)
        _assert_pair(self, (tokens, mask), before[:2], pointers[:2])

    def test_parameter_device_falls_back_when_embeddings_have_no_parameters(self):
        """An empty module uses the caller device and does not raise.

        A later ``_apply`` on a real embedding module wins over that fallback.
        Alignment keeps dtypes, and a missing mask covers the past length.
        """
        empty = nn.Identity()
        cpu_fallback = torch.zeros(1)
        meta_fallback = torch.zeros(1, device=_META)
        self.assertIsNone(next(empty.parameters(), None))
        self.assertEqual(_ESM_PATCH._parameter_device(empty, cpu_fallback), _CPU)
        self.assertEqual(_ESM_PATCH._parameter_device(empty, meta_fallback), _META)

        tokens, mask, embeds, position_ids, before, pointers = _callers()
        aligned_mask, aligned_embeds, aligned_ids, aligned_pos = (
            _ESM_PATCH._align_embedding_activations(
                empty,
                cpu_fallback,
                mask,
                embeds,
                tokens,
                position_ids,
                2,
                5,
                3,
            )
        )
        self.assertIs(aligned_mask, mask)
        self.assertIs(aligned_embeds, embeds)
        self.assertIs(aligned_ids, tokens)
        self.assertIs(aligned_pos, position_ids)

        moved_mask, moved_embeds, moved_ids, _moved_pos = (
            _ESM_PATCH._align_embedding_activations(
                empty,
                meta_fallback,
                mask,
                embeds,
                tokens,
                position_ids,
                2,
                5,
                0,
            )
        )
        self.assertEqual(moved_mask.device, _META)
        self.assertEqual(moved_mask.dtype, torch.bool)
        self.assertIsNot(moved_mask, mask)
        self.assertEqual(moved_embeds.device, _META)
        self.assertEqual(moved_embeds.dtype, torch.float32)
        self.assertEqual(moved_ids.device, _META)
        self.assertEqual(moved_ids.dtype, torch.long)
        _assert_pair(self, (tokens, mask), before[:2], pointers[:2])
        _assert_unchanged(self, embeds, before[2], pointers[2])

        model = _cpu_model()
        self.assertEqual(
            _ESM_PATCH._parameter_device(model.embeddings, meta_fallback), _CPU
        )
        _move_parameters(model.embeddings, _META)
        self.assertEqual(
            _ESM_PATCH._parameter_device(model.embeddings, cpu_fallback), _META
        )
        self.assertEqual(
            _ESM_PATCH._parameter_device(model.encoder, meta_fallback), _CPU
        )

        float_mask = torch.ones(2, 5, 2, dtype=torch.float64)[:, :, 0]
        float_ptr = float_mask.data_ptr()
        self.assertFalse(float_mask.is_contiguous())
        cast_mask, _cast_embeds, _cast_ids, _cast_pos = (
            _ESM_PATCH._align_embedding_activations(
                model.embeddings,
                cpu_fallback,
                float_mask,
                None,
                None,
                None,
                2,
                5,
                4,
            )
        )
        self.assertEqual(cast_mask.shape, (2, 5))
        self.assertEqual(cast_mask.dtype, torch.float64)
        self.assertEqual(cast_mask.device, _META)
        self.assertEqual(float_mask.dtype, torch.float64)
        self.assertEqual(float_mask.device, _CPU)
        self.assertEqual(float_mask.data_ptr(), float_ptr)

        created, no_embeds, no_ids, no_pos = _ESM_PATCH._align_embedding_activations(
            model.embeddings,
            cpu_fallback,
            None,
            None,
            None,
            None,
            2,
            5,
            4,
        )
        self.assertEqual(created.shape, (2, 9))
        self.assertEqual(created.device, _META)
        self.assertEqual(created.dtype, torch.float32)
        self.assertIsNone(no_embeds)
        self.assertIsNone(no_ids)
        self.assertIsNone(no_pos)

    def test_same_device_alignment_keeps_autograd(self):
        """A CPU batch must still gradient the position weights and the input."""
        torch.manual_seed(0)
        model = _cpu_model()
        embeds = torch.randn(*_SHAPE, requires_grad=True)
        embed_ptr = embeds.data_ptr()
        _tokens, mask, _embeds, _position_ids, before, pointers = _callers()

        hidden = _forward(model, mask=mask, embeds=embeds)
        self.assertTrue(hidden.requires_grad)
        hidden.sum().backward()

        self.assertEqual(embeds.data_ptr(), embed_ptr)
        self.assertEqual(embeds.device, _CPU)
        self.assertIsNotNone(embeds.grad)
        self.assertEqual(embeds.grad.device, _CPU)
        self.assertTrue(torch.isfinite(embeds.grad).all().item())
        self.assertGreater(float(embeds.grad.abs().sum()), 0.0)
        _assert_unchanged(self, mask, before[1], pointers[1])
        _assert_position_grad(self, model, _CPU)
        _assert_hidden(self, hidden, _CPU, torch.float32, finite=True)

    def test_cpu_mask_without_alignment_raises_on_meta(self):
        """Restoring the caller-device mask reproduces the meta/CPU crash."""
        model = _cpu_model()
        _move_parameters(_nest(model), _META)
        tokens = torch.tensor([[2, 5, 6, 3]])
        original = _ESM_PATCH._align_embedding_activations

        def _unaligned(  # pylint: disable=too-many-arguments
            embeddings,
            fallback,
            attention_mask,
            inputs_embeds,
            input_ids,
            position_ids,
            batch_size,
            seq_length,
            past_key_values_length,
        ):
            """Return the caller-device mask the pre-fix forward used."""
            del embeddings, fallback, inputs_embeds, input_ids, position_ids
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length + past_key_values_length)
                )
            return attention_mask, None, tokens, None

        _ESM_PATCH._align_embedding_activations = _unaligned
        try:
            with self.assertRaises(RuntimeError) as caught:
                model(input_ids=tokens, return_dict=True)
        finally:
            _ESM_PATCH._align_embedding_activations = original
        message = str(caught.exception).lower()
        self.assertIn("meta", message)
        self.assertIn("cpu", message)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipUnless(EsmConfig is not None, "requires transformers")
class TestEsmMaskParameterDeviceCuda(unittest.TestCase):
    """Parent moves onto CUDA. Skipped on machines without a GPU."""

    def test_parent_to_cuda_accepts_cpu_tokens(self):
        """``parent.to('cuda')`` scores CPU tokens on the parameter device."""
        model = _cpu_model()
        reference = copy.deepcopy(model)
        parent = _nest(model)
        tokens, mask, _embeds, position_ids, before, pointers = _callers()
        parent.to("cuda")
        seen = _install_spy(model)

        hidden = _forward(model, tokens, mask, position_ids=position_ids)
        expected = _forward(reference, tokens, mask, position_ids=position_ids)

        device = model.embeddings.word_embeddings.weight.device
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(device.type, "cuda")
        self.assertEqual(model.embeddings.position_embeddings.weight.device, device)
        _assert_pair(self, (tokens, mask), before[:2], pointers[:2])
        _assert_unchanged(self, position_ids, before[3], pointers[3])
        self.assertEqual(seen["input_ids"].device, device)
        self.assertIsNot(seen["input_ids"], tokens)
        self.assertEqual(seen["mask"].device, device)
        self.assertEqual(seen["position_ids"].device, device)
        self.assertIsNot(seen["position_ids"], position_ids)
        self.assertEqual(tokens.device, _CPU)
        self.assertEqual(tokens.dtype, torch.long)
        _assert_hidden(self, hidden, device, torch.float32, finite=True)
        _assert_close(hidden, expected)

    def test_parent_apply_accepts_cpu_inputs_embeds(self):
        """CPU fused embeddings follow a parent ``_apply`` onto CUDA."""
        model = _cpu_model()
        reference = copy.deepcopy(model)
        parent = _nest(model)
        _tokens, mask, embeds, _position_ids, before, pointers = _callers()
        _move_parameters(parent, torch.device("cuda"))
        seen = _install_spy(model)

        hidden = _forward(model, mask=mask, embeds=embeds)
        expected = _forward(reference, mask=mask, embeds=embeds)

        device = model.embeddings.position_embeddings.weight.device
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(device.type, "cuda")
        _assert_pair(
            self, (embeds, mask), (before[2], before[1]), (pointers[2], pointers[1])
        )
        self.assertEqual(seen["inputs_embeds"].device, device)
        self.assertIsNot(seen["inputs_embeds"], embeds)
        self.assertEqual(seen["inputs_embeds"].dtype, torch.float32)
        self.assertEqual(seen["mask"].device, device)
        self.assertIsNone(seen["input_ids"])
        self.assertEqual(embeds.device, _CPU)
        _assert_hidden(self, hidden, device, torch.float32, finite=True)
        _assert_close(hidden, expected)

    def test_batch_already_on_cuda_keeps_storage(self):
        """A CUDA batch that already matches the weights keeps its storage."""
        model = _cpu_model()
        reference = copy.deepcopy(model)
        cuda = torch.device("cuda")
        _move_parameters(model, cuda)
        tokens, mask, embeds, position_ids, before, pointers = _callers(cuda)
        seen = _install_spy(model)

        hidden = _forward(model, tokens, mask, embeds, position_ids)
        expected = _forward(
            reference,
            tokens.detach().cpu(),
            mask.detach().cpu(),
            embeds.detach().cpu(),
            position_ids.detach().cpu(),
        )

        device = model.embeddings.word_embeddings.weight.device
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(device.type, "cuda")
        _assert_pair(self, (tokens, mask), before[:2], pointers[:2])
        _assert_pair(self, (embeds, position_ids), before[2:], pointers[2:])
        self.assertIs(seen["input_ids"], tokens)
        self.assertIs(seen["mask"], mask)
        self.assertIs(seen["inputs_embeds"], embeds)
        self.assertIs(seen["position_ids"], position_ids)
        _assert_hidden(self, hidden, device, torch.float32, finite=True)
        _assert_close(hidden, expected)

    def test_round_trip_is_reread(self):
        """CUDA, then CPU, then CUDA follows parameters and does not call ``.to``."""
        model = _cpu_model()
        reference = copy.deepcopy(model)
        parent = _nest(model)
        tokens, mask, _embeds, position_ids, before, pointers = _callers()
        seen = _install_spy(model)
        expected = _forward(reference, tokens, mask, position_ids=position_ids)
        cuda = torch.device("cuda")

        _move_parameters(parent, cuda)
        hidden = _forward(model, tokens, mask, position_ids=position_ids)
        device = model.embeddings.position_embeddings.weight.device
        self.assertEqual(device.type, "cuda")
        self.assertEqual(seen["input_ids"].device, device)
        self.assertIsNot(seen["input_ids"], tokens)
        _assert_hidden(self, hidden, device, torch.float32, finite=True)
        _assert_close(hidden, expected)

        _move_parameters(parent, _CPU)
        hidden = _forward(model, tokens, mask, position_ids=position_ids)
        self.assertEqual(_ESM_PATCH._parameter_device(model.embeddings, tokens), _CPU)
        self.assertIs(seen["input_ids"], tokens)
        self.assertIs(seen["mask"], mask)
        self.assertIs(seen["position_ids"], position_ids)
        _assert_hidden(self, hidden, _CPU, torch.float32, finite=True)
        _assert_close(hidden, expected)

        _move_parameters(parent, cuda)
        hidden = _forward(model, tokens, mask, position_ids=position_ids)
        device = model.embeddings.word_embeddings.weight.device
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(device.type, "cuda")
        self.assertIsNot(seen["position_ids"], position_ids)
        _assert_pair(self, (tokens, mask), before[:2], pointers[:2])
        _assert_unchanged(self, position_ids, before[3], pointers[3])
        _assert_hidden(self, hidden, device, torch.float32, finite=True)
        _assert_close(hidden, expected)

    def test_float64_cuda_move_keeps_dtype(self):
        """A float64 CUDA move must not downcast hidden states or token ids."""
        model = _cpu_model()
        parent = _nest(model)
        parent.to(dtype=torch.float64)
        reference = copy.deepcopy(model)
        tokens, mask, _embeds, _position_ids, before, pointers = _callers()
        _move_parameters(parent, torch.device("cuda"))

        hidden = _forward(model, tokens, mask)
        expected = _forward(reference, tokens, mask)

        device = model.embeddings.position_embeddings.weight.device
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(device.type, "cuda")
        self.assertEqual(device, model.embeddings.word_embeddings.weight.device)
        self.assertEqual(model.embeddings.word_embeddings.weight.dtype, torch.float64)
        self.assertEqual(
            model.embeddings.position_embeddings.weight.dtype, torch.float64
        )
        _assert_pair(self, (tokens, mask), before[:2], pointers[:2])
        self.assertEqual(tokens.dtype, torch.long)
        _assert_hidden(self, hidden, device, torch.float64, finite=True)
        _assert_close(hidden, expected)

    def test_dtype_only_move_keeps_cuda_device(self):
        """``.to(dtype=...)`` on CUDA keeps CUDA and does not call child ``.to``."""
        model = _cpu_model()
        parent = _nest(model)
        _move_parameters(parent, torch.device("cuda"))
        parent.to(dtype=torch.float64)
        tokens, mask, _embeds, _position_ids, before, pointers = _callers()

        hidden = _forward(model, tokens, mask)

        device = model.embeddings.position_embeddings.weight.device
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(device.type, "cuda")
        self.assertEqual(model.embeddings.word_embeddings.weight.dtype, torch.float64)
        self.assertEqual(
            model.embeddings.position_embeddings.weight.dtype, torch.float64
        )
        _assert_pair(self, (tokens, mask), before[:2], pointers[:2])
        _assert_hidden(self, hidden, device, torch.float64, finite=True)

    def test_backward_from_cpu_batch_reaches_cuda_position_embeddings(self):
        """CPU caller storage stays put while position gradients land on CUDA."""
        torch.manual_seed(0)
        model = _cpu_model()
        parent = _nest(model)
        _move_parameters(parent, torch.device("cuda"))
        embeds = torch.randn(*_SHAPE, requires_grad=True)
        embed_ptr = embeds.data_ptr()
        _tokens, mask, _embeds, _position_ids, before, pointers = _callers()

        hidden = _forward(model, mask=mask, embeds=embeds)
        hidden.sum().backward()

        device = model.embeddings.position_embeddings.weight.device
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(device.type, "cuda")
        self.assertEqual(embeds.data_ptr(), embed_ptr)
        self.assertEqual(embeds.device, _CPU)
        self.assertEqual(embeds.grad.device, _CPU)
        self.assertTrue(torch.isfinite(embeds.grad).all().item())
        _assert_unchanged(self, mask, before[1], pointers[1])
        _assert_position_grad(self, model, device)
        self.assertEqual(hidden.device, device)

    def test_cuda_batch_follows_cpu_parameters(self):
        """A CUDA batch is embedded on the CPU parameter device."""
        model = _cpu_model()
        reference = copy.deepcopy(model)
        tokens, mask, embeds, position_ids, before, pointers = _callers(
            torch.device("cuda")
        )
        seen = _install_spy(model)

        hidden = _forward(model, tokens, mask, embeds, position_ids)
        expected = _forward(
            reference,
            tokens.detach().cpu(),
            mask.detach().cpu(),
            embeds.detach().cpu(),
            position_ids.detach().cpu(),
        )

        self.assertEqual(model.embeddings.position_embeddings.weight.device, _CPU)
        _assert_pair(self, (tokens, mask), before[:2], pointers[:2])
        _assert_pair(self, (embeds, position_ids), before[2:], pointers[2:])
        self.assertEqual(seen["input_ids"].device, _CPU)
        self.assertIsNot(seen["input_ids"], tokens)
        self.assertEqual(seen["mask"].device, _CPU)
        self.assertEqual(seen["inputs_embeds"].device, _CPU)
        self.assertEqual(tokens.device.type, "cuda")
        _assert_hidden(self, hidden, _CPU, torch.float32, finite=True)
        _assert_close(hidden, expected)


if __name__ == "__main__":
    unittest.main()
