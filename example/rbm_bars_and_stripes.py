"""Train a small RBM and measure its normalized visible distribution offline."""

import argparse
import json

import numpy as np
import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def _binary_states(width):
    indexes = torch.arange(1 << width)
    return ((indexes[:, None] >> torch.arange(width)) & 1).float()


def bars_and_stripes(side):
    """Return distinct binary square images with horizontal or vertical lines."""
    rows = _binary_states(side)
    horizontal = rows[:, :, None].expand(-1, -1, side)
    vertical = horizontal.transpose(1, 2)
    return torch.unique(torch.cat((horizontal, vertical)).reshape(-1, side * side), dim=0)


class EnumeratedBoltzmannSampler:
    """Draw from all Ising states at unit inverse temperature."""

    def __init__(self, width, reads, seed):
        self.reads = reads
        self.rng = np.random.default_rng(seed)
        bits = _binary_states(width).double().numpy()
        self.spins = np.column_stack((2 * bits - 1, np.ones(len(bits))))

    def solve(self, ising_matrix):
        """Return spin rows with the auxiliary spin fixed to +1."""
        scores = ((self.spins @ ising_matrix) * self.spins).sum(axis=1)
        weights = np.exp(scores - scores.max())
        chosen = self.rng.choice(len(weights), size=self.reads, p=weights / weights.sum())
        return self.spins[chosen]


@torch.no_grad()
def visible_metrics(model, patterns):
    """Compute target-pattern mass and mean negative log probability."""
    visible = _binary_states(model.num_visible)
    hidden = _binary_states(model.num_hidden)
    states = torch.cat(
        (visible.repeat_interleave(len(hidden), dim=0), hidden.repeat(len(visible), 1)),
        dim=1,
    )
    log_weights = -model(states).reshape(len(visible), len(hidden))
    visible_log_weights = log_weights.logsumexp(dim=1)
    log_probabilities = visible_log_weights - visible_log_weights.logsumexp(dim=0)
    target = (visible[:, None, :] == patterns[None, :, :]).all(dim=2).any(dim=1)
    return {
        "pattern_mass": log_probabilities[target].exp().sum().item(),
        "mean_nll": -log_probabilities[target].mean().item(),
    }


def run_benchmark(side=2, steps=300, reads=256, seed=42):
    """Use KPP sampler-backed negative phases and exact model evaluation."""
    if side not in (2, 3) or steps < 0 or reads <= 0:
        raise ValueError("Use side 2 or 3, nonnegative steps and positive reads")
    patterns = bars_and_stripes(side)
    visible, hidden = side * side, 4
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = RestrictedBoltzmannMachine(
        visible, hidden,
        quadratic_coef=0.3 * torch.randn(visible, hidden, generator=generator),
        device="cpu",
    )
    sampler = EnumeratedBoltzmannSampler(visible + hidden, reads, seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    initial = visible_metrics(model, patterns)
    for _ in range(steps):
        positive = model.get_hidden(patterns)
        negative = model.sample(sampler)
        optimizer.zero_grad()
        model.objective(positive, negative).backward()
        optimizer.step()

    return {
        "side": side, "patterns": len(patterns), "steps": steps, "reads_per_step": reads,
        "seed": seed, "initial": initial, "trained": visible_metrics(model, patterns),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, choices=(2, 3), default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--reads", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.side, args.steps, args.reads, args.seed), indent=2))


if __name__ == "__main__":
    main()
