# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Constrained decoding with ``QDiffusion.partial_masks``.

``partial_masks`` fixes selected positions of the decoded sequence: every
position marked ``True`` keeps its input token, and only the remaining
non-special positions are proposed, reranked, and resampled.

This example uses a toy proposal model and a toy energy model, so it needs no
checkpoint, download, or Kaiwu license. The energy model rewards sequences that
contain more of the preferred content token, which makes the reranking step
visible in the printed output.
"""

from __future__ import annotations

import torch

from kaiwu.torch_plugin import EnergyModel, QDiffusion, QDiffusionConfig
from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec

BOS_ID, EOS_ID, X_ID, PAD_ID, MASK_ID = 0, 1, 2, 3, 4
VOCAB_SIZE = 8
PREFERRED_ID = 7


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


def build_generator(seed: int = 0) -> QDiffusion:
    """Builds a deterministic, license-free constrained generator."""
    torch.manual_seed(seed)
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
        num_candidates=4,
        proposal_temperature=1.0,
        energy_temperature=1.0,
        resample_ratio=1.0,
    )
    return QDiffusion(PreferSevenProposal(), PreferManySevens(), spec, config)


def main() -> None:
    """Runs one constrained generation and checks the decoding contract."""
    generator = build_generator()
    input_tokens = torch.tensor([[BOS_ID, X_ID, 5, 6, PREFERRED_ID, EOS_ID]])
    fixed = torch.tensor([[False, False, True, True, False, False]])
    output = generator.generate(input_tokens, max_steps=8, partial_masks=fixed)

    editable = (
        ~fixed
        & input_tokens.ne(PAD_ID)
        & input_tokens.ne(BOS_ID)
        & input_tokens.ne(EOS_ID)
    )

    print("input :", input_tokens.tolist())
    print("fixed :", fixed.tolist())
    print("output:", output.tolist())
    print("editable tokens:", output[editable].tolist())

    assert torch.equal(output[fixed], input_tokens[fixed]), "fixed positions changed"
    assert all(
        int(token) not in (BOS_ID, EOS_ID, PAD_ID, MASK_ID) for token in output[editable]
    ), "editable positions must not receive special or mask tokens"
    print("invariants OK: fixed span preserved, no special tokens generated")


if __name__ == "__main__":
    main()
