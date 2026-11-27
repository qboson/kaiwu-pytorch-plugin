"""Bring your own sampler: the contract KPP expects, verified against brute force.

The documented training loop

    positive phase -> sampler.solve(ising_matrix) -> objective -> backward -> step

usually runs with a Kaiwu SDK optimizer, which needs platform credentials. This
example implements the *sampler contract* with plain PyTorch, so the contract can
be exercised and checked offline.

Sampler contract
----------------
``model.get_ising_matrix()`` returns an ``(N + 1) x (N + 1)`` auxiliary-spin
matrix ``M``. The model energy satisfies ``E(x) = c - y^T M y`` with
``y = [2x - 1, +1]``, so low-energy (high-probability) states are the
*maximizers* of ``s^T M s`` -- this is the direction the Kaiwu SDK solvers use.
``solve(M)`` returns ``num_reads`` spin vectors of length ``N + 1`` with entries
in ``{-1, +1}``; the plugin decodes them with ``spins[:-1] * spins[-1]``.

The example builds a small Boltzmann machine with a known ground state, asks the
custom sampler for samples through ``model.sample(sampler)``, and compares the
decoded result against exhaustive enumeration. It then runs a few steps of the
documented training loop as a smoke check.

Run:
    python example/local_sampler/local_sampler.py
    python example/local_sampler/local_sampler.py --steps 10 --reads 8
"""

from __future__ import annotations

# The example inserts src/ on sys.path before importing the plugin.
# pylint: disable=wrong-import-position,import-error,no-name-in-module,too-few-public-methods

import argparse
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine


class MetropolisSampler:
    """A minimal ``solve``-compatible sampler that maximises ``s^T M s``.

    Each sweep attempts one flip per spin in random order, using the exact
    incremental change ``delta = -4 s_i (M s)_i + 4 M_ii`` for the current field
    vector, and keeps the best state seen at any temperature.

    Args:
        num_reads: Number of spin vectors to return per call.
        num_sweeps: Metropolis sweeps per read.
        initial_temperature: Starting temperature of the geometric schedule.
        final_temperature: Final temperature of the schedule.
        seed: Seed for reproducible runs.

    Raises:
        ValueError: If ``num_reads``, ``num_sweeps`` or the temperatures are invalid.
    """

    def __init__(
        self,
        num_reads: int = 8,
        num_sweeps: int = 100,
        initial_temperature: float = 2.0,
        final_temperature: float = 0.2,
        seed: int = 0,
    ) -> None:
        if num_reads < 1:
            raise ValueError("num_reads must be at least 1")
        if num_sweeps < 1:
            raise ValueError("num_sweeps must be at least 1")
        if not 0 < final_temperature < initial_temperature:
            raise ValueError("temperatures must satisfy 0 < final < initial")
        self.num_reads = int(num_reads)
        self.num_sweeps = int(num_sweeps)
        self.initial_temperature = float(initial_temperature)
        self.final_temperature = float(final_temperature)
        self.generator = torch.Generator().manual_seed(int(seed))

    def solve(self, ising_matrix: np.ndarray) -> np.ndarray:
        """Return spin vectors that maximise ``s^T M s``.

        Args:
            ising_matrix: Square auxiliary-spin matrix of shape ``(N + 1, N + 1)``.

        Returns:
            numpy.ndarray: ``(num_reads, N + 1)`` array of ``-1``/``+1`` spins,
            ordered by decreasing objective value.

        Raises:
            ValueError: If the matrix is not a finite square array.
        """
        matrix = np.asarray(ising_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("ising_matrix must be a square matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("ising_matrix must contain only finite values")

        weights = torch.as_tensor(matrix, dtype=torch.float64)
        num_spins = weights.shape[0]
        diagonal = torch.diagonal(weights).clone()
        temperatures = np.geomspace(
            self.initial_temperature, self.final_temperature, self.num_sweeps
        )

        spins_all = torch.empty(self.num_reads, num_spins, dtype=torch.float64)
        values_all = torch.empty(self.num_reads, dtype=torch.float64)

        for read in range(self.num_reads):
            spins = (
                torch.randint(0, 2, (num_spins,), generator=self.generator, dtype=torch.float64)
                * 2
                - 1
            )
            field = weights @ spins
            value = float(spins @ field)
            best_spins, best_value = spins.clone(), value

            for temperature in temperatures:
                for index in torch.randperm(num_spins, generator=self.generator):
                    index = int(index.item())
                    delta = -4.0 * float(spins[index]) * float(field[index]) + 4.0 * float(
                        diagonal[index]
                    )
                    if delta >= 0 or float(
                        torch.rand((), generator=self.generator)
                    ) < float(np.exp(delta / temperature)):
                        spins[index] = -spins[index]
                        # field = M s after the flip: s_i changed sign
                        field = field + 2.0 * spins[index] * weights[:, index]
                        value += delta
                        if value > best_value:
                            best_value = value
                            best_spins = spins.clone()

            spins_all[read], values_all[read] = best_spins, best_value

        order = torch.argsort(values_all, descending=True)
        return spins_all[order].numpy().astype(int)


def build_demo_bm() -> BoltzmannMachine:
    """A four-node Boltzmann machine with four degenerate ground states.

    One anti-ferromagnetic pair (nodes 0/1) and one ferromagnetic pair (nodes
    2/3), so exhaustive enumeration finds the four combinations below instead of
    a single trivial all-ones optimum.
    """
    model = BoltzmannMachine(4)
    with torch.no_grad():
        coupling = torch.zeros(4, 4)
        coupling[0, 1] = coupling[1, 0] = -2.0
        coupling[2, 3] = coupling[3, 2] = 1.5
        model.quadratic_coef.copy_(coupling)
        model.linear_bias.zero_()
    return model


def enumerate_energies(model: BoltzmannMachine) -> dict[tuple[int, ...], float]:
    """Energy of every binary state, by brute force.

    Args:
        model: The model to enumerate.

    Returns:
        dict[tuple[int, ...], float]: Mapping from binary state to its energy.
    """
    energies = {}
    for bits in itertools.product([0.0, 1.0], repeat=model.num_nodes):
        state = torch.tensor(bits, dtype=torch.float32).unsqueeze(0)
        energies[tuple(int(bit) for bit in bits)] = float(model(state).item())
    return energies


def verify_sampler_against_enumeration(
    model: BoltzmannMachine,
    sampler: MetropolisSampler,
    num_draws: int = 4,
) -> dict[str, object]:
    """Check that ``model.sample(sampler)`` returns ground states.

    Args:
        model: The Boltzmann machine to sample from.
        sampler: The custom sampler under test.
        num_draws: Number of independent ``sample`` calls to verify.

    Returns:
        dict[str, object]: ``ground_state``, ``ground_energy``, the decoded
            samples and whether every draw hit a state with the minimal energy.
    """
    energies = enumerate_energies(model)
    ground_energy = min(energies.values())
    ground_states = sorted(state for state, value in energies.items() if abs(value - ground_energy) < 1e-9)

    sampled = []
    for _ in range(num_draws):
        states = model.sample(sampler)
        sampled.extend(tuple(int(value) for value in row) for row in states.tolist())

    hit = [state for state in sampled if abs(energies[state] - ground_energy) < 1e-9]
    return {
        "ground_states": ground_states,
        "ground_energy": ground_energy,
        "sampled": sampled,
        "all_draws_minimal": len(hit) == len(sampled),
        "num_minimal": len(hit),
        "num_sampled": len(sampled),
    }


def train_smoke_test(
    steps: int = 5,
    reads: int = 4,
    learning_rate: float = 0.1,
    seed: int = 0,
    sweeps: int = 30,
) -> dict[str, float]:
    """Run a few steps of the documented training loop and report the objective.

    Args:
        steps: Number of optimisation steps.
        reads: Number of samples per ``solve`` call.
        learning_rate: SGD learning rate.
        seed: Seed for the model and the sampler.
        sweeps: Metropolis sweeps per ``solve`` call.

    Returns:
        dict[str, float]: ``first_objective`` and ``last_objective``.

    Raises:
        ValueError: If ``steps`` is not positive.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")

    torch.manual_seed(seed)
    model = RestrictedBoltzmannMachine(num_visible=4, num_hidden=3)
    patterns = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
        ]
    )
    sampler = MetropolisSampler(num_reads=reads, num_sweeps=sweeps, seed=seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    objectives = []
    for _ in range(steps):
        batch = patterns[
            torch.randint(0, patterns.size(0), (16,), generator=sampler.generator)
        ]
        positive = model.get_hidden(batch, bernoulli=True)
        negative = model.sample(sampler)
        optimizer.zero_grad()
        objective = model.objective(positive, negative)
        objective.backward()
        optimizer.step()
        objectives.append(float(objective.detach()))

    return {"first_objective": objectives[0], "last_objective": objectives[-1]}


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--reads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = build_demo_bm()
    sampler = MetropolisSampler(num_reads=args.reads, seed=args.seed)
    report = verify_sampler_against_enumeration(model, sampler)

    print(f"ground states (exhaustive): {report['ground_states']}")
    print(f"ground energy             : {report['ground_energy']:.4f}")
    print(f"decoded samples           : {report['sampled']}")
    print(
        f"minimal-energy draws      : {report['num_minimal']}/{report['num_sampled']}"
        f" -> {'PASS' if report['all_draws_minimal'] else 'FAIL'}"
    )

    smoke = train_smoke_test(steps=args.steps, reads=args.reads, seed=args.seed)
    print(
        f"training smoke test       : objective {smoke['first_objective']:.4f} -> "
        f"{smoke['last_objective']:.4f} over {args.steps} step(s)"
    )
    return 0 if report["all_draws_minimal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
