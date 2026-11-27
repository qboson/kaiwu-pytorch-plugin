"""A minimal Q-Diffusion generation example with a custom energy model.

``QDiffusion`` reranks proposal candidates with an **energy model**, which is a
public extension point (``kaiwu.torch_plugin.EnergyModel``). The bundled example
relies on a DPLM checkpoint and Hugging Face downloads, so this script keeps only
the pieces that are needed to see the mechanism:

* a tiny proposal model (an embedding table) that predicts token logits,
* a custom ``EnergyModel`` that scores a candidate by its token weights and by
  how much of the noisy context it keeps,
* ``generate()`` over a six-token vocabulary.

Everything is pure PyTorch: no checkpoint download, no Kaiwu SDK credentials and
no GPU are required, and the run is deterministic for a given ``--seed``.

Run:
    python example/qdiffusion_minimal/qdiffusion_minimal.py
    python example/qdiffusion_minimal/qdiffusion_minimal.py --steps 6 --seed 3
"""

from __future__ import annotations

# The example inserts src/ on sys.path before importing the plugin.
# pylint: disable=wrong-import-position,import-error,no-name-in-module,too-few-public-methods

import argparse
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from kaiwu.torch_plugin import EnergyModel, QDiffusion, QDiffusionConfig
from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec

# Vocabulary: 0 padding, 1 BOS, 2 EOS, 3..5 ordinary tokens, 6 = diffusion mask.
PAD_ID, BOS_ID, EOS_ID, MASK_ID = 0, 1, 2, 6
ORDINARY_TOKENS = (3, 4, 5)


class ToyProposal(nn.Module):
    """Proposal model: maps each token to a fixed logits row.

    Args:
        vocab_size: Number of token ids.
        seed: Seed for the embedding table.
    """

    def __init__(self, vocab_size: int, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        table = torch.randn(vocab_size, vocab_size, generator=generator) * 2.0
        self.table = nn.Parameter(table, requires_grad=False)

    def forward(self, noisy_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Returns logits of shape ``[batch, seq_len, vocab]``."""
        del kwargs
        return self.table[noisy_tokens]


class BagOfTokensEnergy(EnergyModel):
    """Custom energy model: low energy for heavy ordinary tokens.

    The score rewards ordinary tokens by a learnable weight and penalises
    candidates that drop the observed (non-masked) context, which is enough to
    change the reranking outcome.

    Args:
        vocab_size: Number of token ids.
        context_weight: Weight of the context-preservation term.
        seed: Seed for the token weights.
    """

    def __init__(self, vocab_size: int, context_weight: float = 1.5, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        weights = torch.zeros(vocab_size)
        weights[list(ORDINARY_TOKENS)] = torch.tensor([1.0, 2.0, 3.0])
        weights += 0.1 * torch.randn(vocab_size, generator=generator)
        self.token_weight = nn.Parameter(weights, requires_grad=False)
        self.context_weight = float(context_weight)
        self.calls = 0

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scores every candidate against its noisy context (``[batch, 1]``)."""
        self.calls += 1
        token_energy = self.token_weight[candidate_tokens].sum(dim=-1)
        observed = noisy_tokens.ne(MASK_ID) & attention_mask
        kept = (
            (candidate_tokens == noisy_tokens) & observed
        ).sum(dim=-1).float()
        return (token_energy - self.context_weight * kept).unsqueeze(-1)


def build_generator(seed: int = 0, context_weight: float = 1.5) -> QDiffusion:
    """Builds the toy generator.

    Args:
        seed: Seed for the proposal and energy tables.
        context_weight: Context-preservation weight of the energy model.

    Returns:
        QDiffusion: The assembled generator.
    """
    vocab_size = MASK_ID + 1
    proposal = ToyProposal(vocab_size, seed=seed)
    energy = BagOfTokensEnergy(vocab_size, context_weight=context_weight, seed=seed)
    spec = SequenceTokenSpec(
        mask_id=MASK_ID, pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID
    )
    config = QDiffusionConfig(
        num_diffusion_timesteps=10,
        num_candidates=4,
        proposal_temperature=1.0,
        energy_temperature=0.5,
        disable_resample=True,
    )
    return QDiffusion(proposal, energy, spec, config)


def generate_demo(
    steps: int = 4,
    seed: int = 0,
    context_weight: float = 1.5,
) -> dict[str, object]:
    """Runs one short generation and reports what the energy model did.

    Args:
        steps: Number of decode steps.
        seed: Seed for the models and the sampling.
        context_weight: Context-preservation weight of the energy model.

    Returns:
        dict[str, object]: ``tokens``, ``energy_calls`` and ``ordinary_fraction``.

    Raises:
        ValueError: If ``steps`` is not positive.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")

    torch.manual_seed(seed)
    generator = build_generator(seed=seed, context_weight=context_weight)
    tokens = torch.tensor([[BOS_ID, 3, MASK_ID, MASK_ID, 5, EOS_ID]])

    with torch.no_grad():
        output = generator.generate(tokens, max_steps=steps)

    row = output[0]
    ordinary = row[2:5]
    return {
        "tokens": row.tolist(),
        "energy_calls": int(generator.energy_model.calls),
        "ordinary_fraction": float(
            torch.isin(ordinary, torch.tensor(ORDINARY_TOKENS)).float().mean().item()
        ),
    }


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-weight", type=float, default=1.5)
    args = parser.parse_args()

    result = generate_demo(
        steps=args.steps, seed=args.seed, context_weight=args.context_weight
    )
    print(f"steps={args.steps} seed={args.seed} context_weight={args.context_weight}")
    print(f"generated tokens        : {result['tokens']}")
    print(f"energy-model calls      : {result['energy_calls']}")
    print(f"ordinary-token fraction : {result['ordinary_fraction']:.2f}")

    ok = result["energy_calls"] > 0 and result["ordinary_fraction"] >= 0.99
    print(
        "PASS: the energy model reranked every candidate and the mask was fully decoded"
        if ok
        else "FAIL: the energy model never scored candidates or a mask token survived"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
