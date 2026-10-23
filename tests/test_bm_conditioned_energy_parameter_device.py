# pylint: disable=protected-access
"""Regression tests for conditioned BM energy device placement.

``BMConditionedEnergyModel`` feeds caller tokens into the encoder and the
pooled features into ``feature_projector``. A parent ``nn.Module.to`` /
``_apply`` moves those parameters without calling the energy model's ``.to``
(merged #164). CPU tokens then meet embedding or linear weights that already
moved:

``RuntimeError: Tensor on device cpu is not on the expected device meta!``

Visible logits must follow the live parameter device. ``energy_bm.device`` is
not consulted: a Boltzmann machine constructed on CUDA must not pull CPU
projector features onto that cache.
"""

from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from dplm.models.bm import BMConditionedEnergyModel


class RecordingEncoder(nn.Module):
    """Encoder that records the device of every ``encode_tokens`` call."""

    def __init__(self, vocab_size: int = 8, hidden_size: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        self.token_devices: list[torch.device] = []
        self.mask_devices: list[torch.device] = []

    def forward(self, tokens, attention_mask=None):
        """Delegate to ``encode_tokens`` so the module is concrete."""
        return self.encode_tokens(tokens, attention_mask=attention_mask)

    def encode_tokens(self, tokens, attention_mask=None):
        """Embed tokens and remember the device they arrived on."""
        self.token_devices.append(tokens.device)
        if attention_mask is not None:
            self.mask_devices.append(attention_mask.device)
        return self.embed(tokens)


class ParameterFreeEncoder(nn.Module):
    """Encoder with no parameters; outputs follow the token device."""

    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, tokens, attention_mask=None):
        """Delegate to ``encode_tokens`` so the module is concrete."""
        return self.encode_tokens(tokens, attention_mask=attention_mask)

    def encode_tokens(self, tokens, attention_mask=None):
        """Broadcast token ids into a hidden-sized feature."""
        del attention_mask
        values = tokens.to(dtype=torch.float32)
        expanded = values.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        return expanded.contiguous()


class RecordingEnergy(BMConditionedEnergyModel):
    """Energy model that records direct ``.to`` invocations."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__(
            encoder,
            bm_num_visible=3,
            bm_num_hidden=2,
            sampler=object(),
        )
        self.direct_to_calls: list[tuple] = []

    def to(self, *args, **kwargs):
        """Record the call, then move like any other module."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


class ParentModule(nn.Module):
    """Container whose ``.to`` / ``_apply`` moves the nested energy model."""

    def __init__(self, energy: nn.Module) -> None:
        super().__init__()
        self.energy = energy

    def forward(self, *inputs):
        """Expose the nested energy model without moving it."""
        return self.energy(*inputs)


def _batch():
    """Return deterministic CPU token batches and a boolean attention mask."""
    noisy = torch.tensor([[1, 2, 3, 0, 1], [4, 5, 6, 7, 0]])
    candidate = torch.tensor([[7, 6, 5, 4, 3], [1, 0, 2, 3, 4]])
    mask = torch.ones(noisy.shape, dtype=torch.bool)
    return noisy, candidate, mask


def _retarget_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Replace storage with zeros on CPU, including a trip out of ``meta``."""
    return torch.zeros(tensor.shape, dtype=tensor.dtype, device="cpu")


def _build(encoder: nn.Module | None = None) -> RecordingEnergy:
    """Build a tiny conditioned energy model around ``encoder``."""
    if encoder is None:
        encoder = RecordingEncoder()
    return RecordingEnergy(encoder)


class TestBmEnergyParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_parent_meta_move_places_logits_on_parameter_device(self):
        """CPU tokens must score after a parent move onto ``meta``."""
        model = _build()
        parent = ParentModule(model)
        noisy, candidate, mask = _batch()
        noisy_ptr = noisy.data_ptr()
        candidate_ptr = candidate.data_ptr()
        mask_ptr = mask.data_ptr()

        parent.to(torch.device("meta"))

        logits = model.build_visible_logits(noisy, candidate, mask)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(logits.device, torch.device("meta"))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(noisy.data_ptr(), noisy_ptr)
        self.assertEqual(candidate.data_ptr(), candidate_ptr)
        self.assertEqual(mask.data_ptr(), mask_ptr)
        self.assertEqual(noisy.device, torch.device("cpu"))
        self.assertEqual(model.encoder.token_devices, [torch.device("meta")] * 2)
        self.assertEqual(model.encoder.mask_devices, [torch.device("meta")] * 2)

    def test_apply_move_matches_parent_to(self):
        """``_apply`` is the path parent modules use and must not call ``.to``."""
        model = _build()
        parent = ParentModule(model)
        noisy, candidate, mask = _batch()

        parent._apply(lambda tensor: tensor.to(torch.device("meta")))
        logits = model.build_visible_logits(noisy, candidate, mask)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(logits.device, torch.device("meta"))
        self.assertEqual(
            next(model.feature_projector.parameters()).device, logits.device
        )

    def test_projector_only_move_keeps_encoder_on_caller_device(self):
        """A projector move must not drag encoder inputs off the caller device."""
        model = _build()
        noisy, candidate, mask = _batch()

        model.feature_projector._apply(lambda tensor: tensor.to(torch.device("meta")))
        logits = model.build_visible_logits(noisy, candidate, mask)

        self.assertEqual(logits.device, torch.device("meta"))
        self.assertEqual(next(model.encoder.parameters()).device, torch.device("cpu"))
        self.assertEqual(model.encoder.token_devices, [torch.device("cpu")] * 2)
        self.assertEqual(model.encoder.mask_devices, [torch.device("cpu")] * 2)
        self.assertEqual(noisy.device, torch.device("cpu"))

    def test_encoder_only_move_places_features_on_encoder_device(self):
        """Pooled features follow the encoder when the projector stays put."""
        model = _build()
        noisy, candidate, mask = _batch()

        model.encoder._apply(lambda tensor: tensor.to(torch.device("meta")))
        features = model.build_conditioned_features(noisy, candidate, mask)

        self.assertEqual(features.device, torch.device("meta"))
        self.assertEqual(features.shape, (2, 8))
        self.assertEqual(
            next(model.feature_projector.parameters()).device, torch.device("cpu")
        )
        self.assertEqual(model.encoder.token_devices, [torch.device("meta")] * 2)
        self.assertEqual(noisy.device, torch.device("cpu"))

    def test_parameter_free_encoder_follows_tokens_onto_projector(self):
        """An empty encoder falls back to the token device, then the projector."""
        model = _build(ParameterFreeEncoder())
        noisy, candidate, mask = _batch()

        model.feature_projector._apply(lambda tensor: tensor.to(torch.device("meta")))
        logits = model.build_visible_logits(noisy, candidate, mask)

        self.assertEqual(logits.device, torch.device("meta"))
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(noisy.device, torch.device("cpu"))
        self.assertEqual(list(model.encoder.parameters()), [])

    def test_device_is_reread_after_meta_round_trip(self):
        """A later CPU retarget must win; the helper must not cache ``meta``."""
        model = _build()
        parent = ParentModule(model)
        noisy, candidate, mask = _batch()

        parent._apply(lambda tensor: tensor.to(torch.device("meta")))
        meta_logits = model.build_visible_logits(noisy, candidate, mask)
        parent._apply(_retarget_cpu)
        cpu_logits = model.build_visible_logits(noisy, candidate, mask)

        self.assertEqual(meta_logits.device, torch.device("meta"))
        self.assertEqual(cpu_logits.device, torch.device("cpu"))
        self.assertEqual(cpu_logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(cpu_logits).all().item())
        self.assertEqual(model.encoder.token_devices[-2:], [torch.device("cpu")] * 2)

    def test_stale_bm_device_does_not_move_features(self):
        """The Boltzmann machine cache must not choose the projector device."""
        model = _build()
        noisy, candidate, mask = _batch()
        model.energy_bm.device = torch.device("meta")

        logits = model.build_visible_logits(noisy, candidate, mask)

        self.assertEqual(logits.device, torch.device("cpu"))
        self.assertEqual(model.encoder.token_devices, [torch.device("cpu")] * 2)
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_dtype_only_move_stays_on_cpu(self):
        """``.to(dtype=...)`` keeps CPU and preserves the projector dtype."""
        model = _build()
        noisy, candidate, mask = _batch()

        model.to(dtype=torch.float64)
        logits = model.build_visible_logits(noisy, candidate, mask)

        self.assertEqual(model.direct_to_calls[0][1]["dtype"], torch.float64)
        self.assertEqual(
            next(model.feature_projector.parameters()).device, torch.device("cpu")
        )
        self.assertEqual(logits.device, torch.device("cpu"))
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(noisy.dtype, torch.long)
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_same_device_backward_reaches_projector(self):
        """A CPU batch still trains the projector and the encoder embedding."""
        model = _build()
        noisy, candidate, mask = _batch()

        logits = model.build_visible_logits(noisy, candidate, mask)
        logits.sum().backward()

        self.assertEqual(logits.device, torch.device("cpu"))
        self.assertIsNotNone(model.feature_projector.weight.grad)
        self.assertIsNotNone(model.encoder.embed.weight.grad)
        self.assertGreater(model.feature_projector.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.encoder.embed.weight.grad.abs().sum().item(), 0.0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBmEnergyParameterDeviceCuda(unittest.TestCase):
    """Construct-on-CPU / parent-move-to-CUDA coverage."""

    def test_parent_cuda_move_matches_cpu_reference(self):
        """CPU tokens against CUDA weights match a CPU weight copy."""
        model = _build()
        reference = copy.deepcopy(model).cpu()
        parent = ParentModule(model)
        noisy, candidate, mask = _batch()

        parent.to("cuda")
        logits = model.build_visible_logits(noisy, candidate, mask)
        expected = reference.build_visible_logits(noisy, candidate, mask)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(logits.device.type, "cuda")
        self.assertEqual(noisy.device, torch.device("cpu"))
        torch.testing.assert_close(logits.cpu(), expected)

    def test_backward_from_cpu_tokens_reaches_cuda_weights(self):
        """Gradients land on CUDA parameters while the caller batch stays on CPU."""
        model = _build()
        parent = ParentModule(model)
        noisy, candidate, mask = _batch()

        parent.to("cuda")
        logits = model.build_visible_logits(noisy, candidate, mask)
        logits.sum().backward()

        self.assertEqual(model.feature_projector.weight.grad.device.type, "cuda")
        self.assertEqual(model.encoder.embed.weight.grad.device.type, "cuda")
        self.assertEqual(noisy.device, torch.device("cpu"))

    def test_cuda_round_trip_is_reread(self):
        """CUDA → CPU → CUDA must follow parameters, not the first move."""
        model = _build()
        reference = copy.deepcopy(model).cpu()
        parent = ParentModule(model)
        noisy, candidate, mask = _batch()
        expected = reference.build_visible_logits(noisy, candidate, mask)

        parent.to("cuda")
        parent.to("cpu")
        cpu_logits = model.build_visible_logits(noisy, candidate, mask)
        parent.to("cuda")
        cuda_logits = model.build_visible_logits(noisy, candidate, mask)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(cpu_logits.device, torch.device("cpu"))
        self.assertEqual(cuda_logits.device.type, "cuda")
        torch.testing.assert_close(cpu_logits, expected)
        torch.testing.assert_close(cuda_logits.cpu(), expected)


if __name__ == "__main__":
    unittest.main()
