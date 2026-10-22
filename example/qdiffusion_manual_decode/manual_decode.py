# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Drive Q-Diffusion decoding step by step with ``initialize_state`` / ``step``.

``QDiffusion.generate`` runs the whole decode loop internally. The same loop can
be driven from outside: ``initialize_state`` creates the decode state and each
``step`` call runs one proposal/rerank/remask round and appends one frame to
``state["history"]``. This is the entry point for custom stopping rules,
per-step inspection, or gradient-based decoding.

The example uses a toy proposal and a toy energy model, so it needs no
checkpoint, dataset, or Kaiwu license.
"""

from __future__ import annotations

import torch

from kaiwu.torch_plugin import EnergyModel, QDiffusion, QDiffusionConfig
from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec

BOS_ID, EOS_ID, X_ID, PAD_ID, MASK_ID = 0, 1, 2, 3, 4
VOCAB_SIZE = 8
PREFERRED_ID = 7
MAX_STEPS = 6


class PreferSevenProposal(torch.nn.Module):  # pylint: disable=too-few-public-methods
    """Toy proposal that always ranks token 7 highest."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Returns proposal logits shaped ``[batch, seq_len, vocab]``."""
        logits = torch.full((tokens.size(0), tokens.size(1), VOCAB_SIZE), -4.0)
        logits[..., PREFERRED_ID] = 4.0
        logits[..., 5] = 1.0
        return logits * self.scale


class PreferManySevens(EnergyModel):  # pylint: disable=too-few-public-methods
    """Toy energy that rewards candidates containing more preferred tokens."""

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns one energy per candidate; lower means more preferred tokens."""
        del noisy_tokens, attention_mask
        counts = (candidate_tokens == PREFERRED_ID).float().sum(dim=1, keepdim=True)
        return -counts


def build_generator() -> QDiffusion:
    """Builds a deterministic, license-free generator."""
    spec = SequenceTokenSpec(
        mask_id=MASK_ID,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        x_id=X_ID,
        tokenizer=None,
    )
    config = QDiffusionConfig(
        num_diffusion_timesteps=8,
        num_candidates=1,
        proposal_temperature=0.0,
        resample_ratio=1.0,
    )
    return QDiffusion(PreferSevenProposal(), PreferManySevens(), spec, config)


def masked_positions(tokens: torch.Tensor) -> int:
    """Counts positions that still hold the mask token."""
    return int((tokens == MASK_ID).sum())


def main() -> None:
    """Runs the manual decode loop and compares it with ``generate``."""
    generator = build_generator()
    input_tokens = torch.tensor([[BOS_ID, X_ID, 5, 6, PREFERRED_ID, EOS_ID]])
    fixed = torch.tensor([[False, False, True, True, False, False]])

    state = generator.initialize_state(
        input_tokens, partial_masks=fixed, max_steps=MAX_STEPS
    )
    print("initial masked positions:", masked_positions(state["output_tokens"]))

    steps_used = 0
    for step_index in range(MAX_STEPS):
        state = generator.step(state)
        steps_used = step_index + 1
        print(
            f"step {steps_used}: masked={masked_positions(state['output_tokens'])} "
            f"history={len(state['history'])}"
        )
        if masked_positions(state["output_tokens"]) == 0:
            break

    print("final tokens:", state["output_tokens"].tolist())
    assert state["step"] == steps_used
    assert len(state["history"]) == steps_used + 1
    assert torch.equal(state["output_tokens"][fixed], input_tokens[fixed])
    assert masked_positions(state["output_tokens"]) == 0

    torch.manual_seed(0)
    manual = generator.initialize_state(
        input_tokens, partial_masks=fixed, max_steps=MAX_STEPS
    )
    for _ in range(MAX_STEPS):
        manual = generator.step(manual)

    torch.manual_seed(0)
    generated = generator.generate(
        input_tokens, partial_masks=fixed, max_steps=MAX_STEPS
    )
    same = torch.equal(manual["output_tokens"], generated)
    print("manual loop matches generate:", same)
    assert same
    print("manual decode OK")


if __name__ == "__main__":
    main()
