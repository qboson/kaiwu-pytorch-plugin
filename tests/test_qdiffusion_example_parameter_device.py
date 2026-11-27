"""Regression tests for Q-Diffusion example device placement.

``QDiffusion`` caches ``device`` as plain metadata and only refreshes it when
``.to()`` runs on the wrapper itself. PyTorch moves nested modules through
``_apply`` *without* calling the child's ``.to()`` override, so a generator
nested inside a parent container can keep a stale ``generator.device`` after
its parameters were moved. Example helpers must therefore follow parameter
devices (defense in depth until library PR #488 lands).

Before the example-layer fix, ``encode_sequence`` and collate helpers used
``generator.device``, which left token tensors on CPU while parameters lived on
``meta`` (or CUDA) after a parent-initiated move — the repro tests below assert
that mismatch explicitly.

CUDA-dependent cases mirror ``tests/test_bm_device_sync.py``: they run only
when a GPU is available.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_QDIFFUSION_ROOT = _REPO_ROOT / "example" / "qdiffusion"
_DPLM_ROOT = _QDIFFUSION_ROOT / "dplm"


def _load_runtime():
    runtime_path = _DPLM_ROOT / "utils" / "runtime.py"
    spec = importlib.util.spec_from_file_location("qdiff_runtime", runtime_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TinyProposal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return dummy logits for the toy proposal backbone."""
        batch = tokens.size(0)
        return self.weight.unsqueeze(0).expand(batch, -1, -1)[:, :, :8]


class _TinyEnergy(EnergyModel):
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Return zero energies on the noisy-token device."""
        del candidate_tokens, attention_mask
        return torch.zeros(noisy_tokens.size(0), 1, device=noisy_tokens.device)


class _MockTokenizer:
    """Minimal tokenizer stand-in for example helper tests."""

    def __call__(self, sequence, **kwargs):
        """Encode one sequence or delegate batch encoding to ``batch_encode_plus``."""
        if isinstance(sequence, list):
            return self.batch_encode_plus(sequence, **kwargs)
        length = max(len(sequence), 1)
        return {"input_ids": torch.arange(length, dtype=torch.long).reshape(1, -1)}

    def batch_encode_plus(self, batch, **kwargs):
        """Return padded ``input_ids`` for a batch of raw sequences."""
        del kwargs
        width = max(len(item) for item in batch)
        rows = [torch.arange(max(len(item), 1), dtype=torch.long) for item in batch]
        padded = torch.stack(
            [torch.nn.functional.pad(row, (0, width - row.numel())) for row in rows]
        )
        return {"input_ids": padded}


class RecordingQDiffusion(QDiffusion):
    """QDiffusion that records direct ``.to()`` invocations."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.direct_to_calls: list[tuple[tuple, dict]] = []

    def to(self, *args, **kwargs):
        """Record and forward module moves."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def _token_spec(tokenizer=None) -> SequenceTokenSpec:
    return SequenceTokenSpec(
        mask_id=0,
        pad_id=1,
        bos_id=2,
        eos_id=3,
        tokenizer=tokenizer or _MockTokenizer(),
    )


def _nest_and_move_to_meta(generator: QDiffusion) -> QDiffusion:
    """Nest ``generator`` and move parameters to ``meta`` via a parent module."""
    wrapper = nn.Module()
    wrapper.generator = generator
    wrapper._apply(  # pylint: disable=protected-access
        lambda tensor: tensor.to(torch.device("meta"))
    )
    return generator


class TestExampleDeviceBookkeeping(unittest.TestCase):
    """Mechanism tests that run on every machine (CPU / meta)."""

    def setUp(self) -> None:
        self.runtime = _load_runtime()

    def test_parent_to_does_not_call_child_to_override(self) -> None:
        """Parent ``.to()`` must move parameters without invoking child ``.to()``."""
        generator = RecordingQDiffusion(
            _TinyProposal(),
            _TinyEnergy(),
            _token_spec(),
            QDiffusionConfig(),
        )
        parent = nn.Module()
        parent.generator = generator

        parent.to(torch.device("meta"))

        self.assertEqual(generator.direct_to_calls, [])
        self.assertEqual(
            next(generator.parameters()).device,
            torch.device("meta"),
        )

    def test_parameter_device_follows_parent_apply_move(self) -> None:
        """``parameter_device`` must track parent ``_apply`` moves."""
        generator = QDiffusion(
            _TinyProposal(),
            _TinyEnergy(),
            _token_spec(),
            QDiffusionConfig(),
        )
        _nest_and_move_to_meta(generator)

        self.assertEqual(
            self.runtime.parameter_device(generator),
            torch.device("meta"),
        )
        self.assertEqual(generator.device, torch.device("cpu"))

    def test_stale_cache_repro_before_fix_pattern(self) -> None:
        """Document the bug: ``generator.device`` diverges from parameters."""
        generator = QDiffusion(
            _TinyProposal(),
            _TinyEnergy(),
            _token_spec(),
            QDiffusionConfig(),
        )
        _nest_and_move_to_meta(generator)

        encoded = generator.tokenizer("ACDE", return_tensors="pt")
        stale_tokens = encoded["input_ids"].to(generator.device)
        fixed_tokens = encoded["input_ids"].to(self.runtime.parameter_device(generator))

        self.assertEqual(stale_tokens.device, torch.device("cpu"))
        self.assertEqual(fixed_tokens.device, torch.device("meta"))
        self.assertNotEqual(stale_tokens.device, fixed_tokens.device)

    def test_encode_sequence_uses_parameter_device(self) -> None:
        """``encode_sequence`` must place tokens on the parameter device."""
        generator = QDiffusion(
            _TinyProposal(),
            _TinyEnergy(),
            _token_spec(),
            QDiffusionConfig(),
        )
        _nest_and_move_to_meta(generator)

        tokens = self.runtime.encode_sequence(generator, "ACDE")
        self.assertEqual(tokens.device, torch.device("meta"))

    def test_direct_generator_to_keeps_parameter_device_consistent(self) -> None:
        """Sanity: explicit ``generator.to`` should agree with ``parameter_device``."""
        generator = QDiffusion(
            _TinyProposal(),
            _TinyEnergy(),
            _token_spec(),
            QDiffusionConfig(),
        )
        generator.to(torch.device("meta"))

        self.assertEqual(
            self.runtime.parameter_device(generator),
            generator.device,
        )

    def test_collate_targets_use_parameter_device(self) -> None:
        """Mirror ``workflow_helpers.build_data_loader_from_records`` collate."""
        generator = QDiffusion(
            _TinyProposal(),
            _TinyEnergy(),
            _token_spec(),
            QDiffusionConfig(),
        )
        _nest_and_move_to_meta(generator)

        sequences = ["AC", "GT"]
        encoded = generator.tokenizer(
            sequences,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
        )
        targets = encoded["input_ids"].to(self.runtime.parameter_device(generator))

        self.assertEqual(targets.device, torch.device("meta"))

    def test_build_full_mask_input_uses_parameter_device(self) -> None:
        """Mirror ``esm2_eval_helpers.build_full_mask_input`` device placement."""
        generator = QDiffusion(
            _TinyProposal(),
            _TinyEnergy(),
            _token_spec(),
            QDiffusionConfig(),
        )
        _nest_and_move_to_meta(generator)

        masked_sequence = "".join(["<mask>"] * 4)
        encoded = generator.tokenizer.batch_encode_plus(
            [masked_sequence],
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        tokens = encoded["input_ids"].to(self.runtime.parameter_device(generator))
        self.assertEqual(tokens.device, torch.device("meta"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestExampleDeviceSyncAcrossAccelerators(unittest.TestCase):
    """Parent ``.to("cpu")`` after CUDA construction (CPU-only CI cannot run this)."""

    def setUp(self) -> None:
        self.runtime = _load_runtime()

    def test_encode_sequence_after_parent_move_to_cpu(self) -> None:
        """Encode after CUDA construction and a parent move to CPU."""
        generator = QDiffusion(
            _TinyProposal(),
            _TinyEnergy(),
            _token_spec(),
            QDiffusionConfig(),
            device="cuda",
        )
        self.assertEqual(next(generator.parameters()).device.type, "cuda")

        wrapper = nn.Module()
        wrapper.generator = generator
        wrapper.to("cpu")

        self.assertEqual(next(generator.parameters()).device.type, "cpu")
        tokens = self.runtime.encode_sequence(generator, "ACDE")
        self.assertEqual(tokens.device.type, "cpu")

    def test_collate_after_parent_move_to_cpu(self) -> None:
        """Collate placement after CUDA construction and a parent move to CPU."""
        generator = QDiffusion(
            _TinyProposal(),
            _TinyEnergy(),
            _token_spec(),
            QDiffusionConfig(),
            device="cuda",
        )
        wrapper = nn.Module()
        wrapper.generator = generator
        wrapper.to("cpu")

        encoded = generator.tokenizer(
            ["AC", "GT"],
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
        )
        targets = encoded["input_ids"].to(self.runtime.parameter_device(generator))
        self.assertEqual(targets.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
