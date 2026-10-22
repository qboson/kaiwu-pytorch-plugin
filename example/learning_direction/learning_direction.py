"""Does the documented training loop actually learn? Checked by exact enumeration.

The Quick Start loop is

    positive = rbm.get_hidden(batch, bernoulli=True)
    negative = rbm.sample(sampler)
    objective = rbm.objective(positive, negative)
    objective.backward(); optimizer.step()

``objective()`` is documented as a surrogate whose gradient matches the negative
log-likelihood gradient, so a step with a positive learning rate must make the
training patterns *more* probable under the model. For a three-visible /
two-hidden RBM the marginal probability is cheap to compute exactly, so this
example measures the claim instead of trusting it:

* the exact mean ``p(pattern)`` before and after a few dozen steps, and
* the mean energy gap between the patterns and a uniformly random state.

To stay free of Kaiwu SDK credentials the negative phase is provided by an
exhaustive ground-state search over the 2^5 model states (what a perfect sampler
would return); no optimizer object from the SDK is constructed.

Run:
    python example/learning_direction/learning_direction.py
    python example/learning_direction/learning_direction.py --steps 80
"""

from __future__ import annotations

# The example inserts src/ on sys.path before importing the plugin.
# pylint: disable=wrong-import-position,import-error,no-name-in-module

import argparse
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from kaiwu.torch_plugin import RestrictedBoltzmannMachine

PATTERNS = [
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
]


def _all_states(num_nodes: int) -> torch.Tensor:
    """Every binary state of ``num_nodes`` units as a ``(2**n, n)`` tensor."""
    return torch.tensor(
        list(itertools.product([0.0, 1.0], repeat=num_nodes)), dtype=torch.float32
    )


def exact_pattern_probability(model: RestrictedBoltzmannMachine, pattern: torch.Tensor) -> float:
    """Exact ``P(pattern)`` by enumerating all joint states.

    Args:
        model: The RBM to evaluate.
        pattern: Binary visible pattern of length ``model.num_visible``.

    Returns:
        float: Probability mass of the pattern under the model distribution.
    """
    states = _all_states(model.num_nodes)
    energies = -model(states).detach().double()
    weights = torch.exp(energies - energies.max())
    visible = states[:, : model.num_visible]
    matches = torch.all(visible == pattern.unsqueeze(0), dim=1)
    return float((weights[matches].sum() / weights.sum()).item())


def ground_state(model: RestrictedBoltzmannMachine) -> torch.Tensor:
    """Exhaustive minimal-energy state of the current model.

    Args:
        model: The RBM to inspect.

    Returns:
        torch.Tensor: State tensor of shape ``(1, num_nodes)``.
    """
    with torch.no_grad():
        states = _all_states(model.num_nodes)
        energies = model(states)
        return states[energies.argmin()].unsqueeze(0)


def mean_energy_gap(model: RestrictedBoltzmannMachine) -> float:
    """Mean energy of the patterns minus the exact mean energy of all states.

    The patterns are completed with their mean-field hidden activations, which is
    the same full-state convention ``objective()`` uses for the positive phase;
    the reference is the exhaustive average over the ``2**N`` model states, so the
    value is deterministic (no sampling noise).

    Args:
        model: The RBM to evaluate.

    Returns:
        float: Positive values mean the patterns sit below a typical state.
    """
    with torch.no_grad():
        patterns = torch.tensor(PATTERNS, dtype=torch.float32)
        positive = model.get_hidden(patterns)
        pattern_energy = float(model(positive).mean().item())
        all_energy = float(model(_all_states(model.num_nodes)).mean().item())
        return all_energy - pattern_energy


def train_steps(
    model: RestrictedBoltzmannMachine,
    *,
    steps: int,
    learning_rate: float,
    batch_size: int = 32,
    generator: torch.Generator | None = None,
    flip_sign: bool = False,
) -> None:
    """Runs ``steps`` iterations of the documented loop.

    Args:
        model: The RBM to train in place.
        steps: Number of optimisation steps.
        learning_rate: SGD learning rate.
        batch_size: Number of sampled training patterns per step.
        generator: Optional torch generator for reproducible batches.
        flip_sign: Negates the objective, used to prove the checks are sensitive
            to the direction of the update.

    Raises:
        ValueError: If ``steps`` is not positive.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    patterns = torch.tensor(PATTERNS, dtype=torch.float32)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        batch = patterns[
            torch.randint(0, patterns.size(0), (batch_size,), generator=generator)
        ]
        positive = model.get_hidden(batch, bernoulli=True)
        negative = ground_state(model).expand(positive.size(0), -1)
        optimizer.zero_grad()
        objective = model.objective(positive, negative)
        if flip_sign:
            objective = -objective
        objective.backward()
        optimizer.step()


def evaluate_learning(
    steps: int = 200,
    learning_rate: float = 0.4,
    seed: int = 0,
    flip_sign: bool = False,
) -> dict[str, float]:
    """Measures the exact data probability before and after training.

    Args:
        steps: Number of optimisation steps.
        learning_rate: SGD learning rate.
        seed: Seed for the model initialisation and batches.
        flip_sign: Negates the objective to prove the check is directional.

    Returns:
        dict[str, float]: ``p_before``, ``p_after`` and ``gap_before``/``gap_after``.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    patterns = torch.tensor(PATTERNS, dtype=torch.float32)

    p_before = float(
        np.mean([exact_pattern_probability(model, row) for row in patterns])
    )
    gap_before = mean_energy_gap(model)

    train_steps(
        model,
        steps=steps,
        learning_rate=learning_rate,
        generator=generator,
        flip_sign=flip_sign,
    )

    p_after = float(
        np.mean([exact_pattern_probability(model, row) for row in patterns])
    )
    gap_after = mean_energy_gap(model)
    return {
        "p_before": p_before,
        "p_after": p_after,
        "gap_before": gap_before,
        "gap_after": gap_after,
    }


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = evaluate_learning(steps=args.steps, seed=args.seed)
    print(f"steps={args.steps} seed={args.seed}")
    print(f"mean p(pattern) before: {result['p_before']:.6f}")
    print(f"mean p(pattern) after : {result['p_after']:.6f}")
    print(f"energy gap before     : {result['gap_before']:.4f}")
    print(f"energy gap after      : {result['gap_after']:.4f}")

    improved = result["p_after"] > result["p_before"]
    print(
        "PASS: the documented loop increased the probability of the training patterns"
        if improved
        else "FAIL: the documented loop did not increase the pattern probability"
    )
    return 0 if improved else 1


if __name__ == "__main__":
    raise SystemExit(main())
