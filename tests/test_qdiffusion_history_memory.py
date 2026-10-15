"""Regression tests for the QDiffusion decode-state memory contract.

``state["history"]`` keeps one token snapshot per decode step. On the default
``generate`` path the caller only receives ``output_tokens``, so those
snapshots were allocated and retained without ever being observable. These
tests pin down that the token-only path retains nothing, while every path
that can actually read the history keeps behaving as before.
"""

import os
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class DummyTokenizer:
    """Minimal tokenizer stub used by direct-construction tests."""

    def batch_decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(str(int(token)) for token in row) for row in tokens]


class DummyProposalModel(nn.Module):
    """Tiny proposal model that returns logits over one toy vocabulary."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        del kwargs
        hidden = torch.tanh(self.hidden(self.embedding(input_ids)))
        return self.lm_head(hidden)


class DummyEnergyModel(EnergyModel):
    """Tiny energy model whose parameters are optimized by QDiffusion."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        del vocab_size, hidden_size
        super().__init__()
        self.num_score_calls = 0

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del attention_mask
        self.num_score_calls += 1
        return (
            candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)
            - noisy_tokens.to(torch.float32).sum(dim=1, keepdim=True)
        )


def history_bytes(history) -> int:
    """Returns the retained payload of a decode history in bytes."""
    return sum(tensor.numel() * tensor.element_size() for tensor in history)


class HistorySpy(QDiffusion):
    """QDiffusion that records the retained history size at each decode step."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seen_history_bytes: list[int] = []

    def _decode_step(self, state, partial_masks=None):
        self.seen_history_bytes.append(history_bytes(state["history"]))
        return super()._decode_step(state, partial_masks=partial_masks)


class TestHistoryMemoryContract(unittest.TestCase):
    """Pins the memory contract of ``initialize_state`` / ``generate``."""

    def setUp(self):
        vocab_size = 8
        hidden_size = 12
        self.proposal_model = DummyProposalModel(vocab_size=vocab_size, hidden_size=hidden_size)
        self.energy_model = DummyEnergyModel(vocab_size=vocab_size, hidden_size=hidden_size)
        self.token_spec = SequenceTokenSpec(
            pad_id=0,
            bos_id=1,
            eos_id=2,
            mask_id=3,
            x_id=4,
            tokenizer=DummyTokenizer(),
        )
        self.config = QDiffusionConfig(
            num_diffusion_timesteps=8,
            num_candidates=2,
            proposal_temperature=0.0,
            disable_resample=True,
        )
        self.targets = torch.tensor(
            [
                [1, 5, 6, 2, 0],
                [1, 6, 5, 2, 0],
            ],
            dtype=torch.long,
        )

    def make_model(self) -> HistorySpy:
        return HistorySpy(
            proposal_model=self.proposal_model,
            energy_model=self.energy_model,
            token_spec=self.token_spec,
            config=self.config,
            freeze_proposal=False,
        )

    def test_final_token_path_retains_no_snapshots(self):
        model = self.make_model()
        generated = model.generate(self.targets, max_steps=4)

        self.assertIsInstance(generated, torch.Tensor)
        self.assertEqual(generated.shape, self.targets.shape)
        self.assertTrue(model.seen_history_bytes)
        self.assertTrue(all(size == 0 for size in model.seen_history_bytes))

    def test_return_state_keeps_every_snapshot(self):
        model = self.make_model()
        state = model.generate(self.targets, max_steps=4, return_state=True)

        self.assertEqual(len(state["history"]), 5)
        self.assertEqual(history_bytes(state["history"]), 5 * 2 * 5 * 8)

    def test_record_history_false_for_external_loop(self):
        model = self.make_model()
        state = model.initialize_state(self.targets, max_steps=3, record_history=False)
        self.assertEqual(state["history"], [])

        for _ in range(3):
            state = model.step(state)

        self.assertEqual(state["history"], [])
        self.assertEqual(state["step"], 3)
        self.assertTrue(all(size == 0 for size in model.seen_history_bytes))

    def test_default_external_loop_still_records(self):
        model = self.make_model()
        state = model.initialize_state(self.targets, max_steps=3)
        self.assertEqual(len(state["history"]), 1)

        for _ in range(3):
            state = model.step(state)

        self.assertEqual(len(state["history"]), 4)

    def test_legacy_state_without_flag_still_records(self):
        model = self.make_model()
        state = model.initialize_state(self.targets, max_steps=2)
        del state["record_history"]  # a hand-built state predating the opt-out

        state = model.step(state)
        self.assertEqual(len(state["history"]), 2)

    def test_final_tokens_match_between_paths(self):
        torch.manual_seed(0)
        tokens_only = self.make_model().generate(self.targets, max_steps=3)

        torch.manual_seed(0)
        with_state = self.make_model().generate(self.targets, max_steps=3, return_state=True)

        self.assertTrue(torch.equal(tokens_only, with_state["output_tokens"]))


if __name__ == "__main__":
    unittest.main()
