"""Two-phase equilibrium-propagation training driven by the Kaiwu annealer.

Each training sample needs two solver calls: a free phase at ``beta = 0`` and a
nudged phase at ``beta > 0`` that warm-starts from the free solution. The
parameter update is the difference between the two phase energies, divided by
``beta``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ising_ep.kaiwu_solver import KaiwuSolver
from ising_ep.model import IsingEPModel

from .metrics import pair_dynamics


@dataclass(frozen=True)
class PhasePair:
    """Selected free and nudged spin states for one training sample.

    Attributes:
        free_states: One spin vector per layer from the free phase, output last.
        nudged_states: One spin vector per layer from the nudged phase, output last.
    """

    free_states: tuple[torch.Tensor, ...]
    nudged_states: tuple[torch.Tensor, ...]


def _solve_phase(
    solver: KaiwuSolver,
    qubo: torch.Tensor,
    pair_seed: int,
    init_solution: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one annealing phase and take the candidate Kaiwu ranked first.

    Args:
        solver: Annealer bound to the shared schedule.
        qubo: QUBO matrix for this phase.
        pair_seed: Seed shared by both phases of this sample.
        init_solution: Warm start in solver order, or ``None`` for a cold start.

    Returns:
        The selected physical spin vector and the solver row it came from. The
        row is what the next phase passes back as ``init_solution``. Kaiwu
        already ranks candidates and fixes the auxiliary spin at ``+1``, so the
        first row is the selection and its leading entries are the physical
        spins.
    """
    best = solver.solve(qubo, seed=pair_seed, init_solution=init_solution)[0]
    return best[:-1], best


def solve_phases(
    model: IsingEPModel,
    solver: KaiwuSolver,
    x: Any,
    target: Any,
    beta: float,
    pair_seed: int,
) -> PhasePair:
    """Solve the paired free and nudged phases for one sample.

    Both phases share ``pair_seed`` and the solver's annealing schedule, so the
    only intended differences between them are the output field and the warm
    start. The nudged phase is seeded with the free phase's complete solver row,
    which is this project's stand-in for the reverse annealing used on quantum
    hardware: it keeps the nudged state close to the free state so their
    difference reflects the nudge rather than solver noise.

    Args:
        model: Network whose fields define both QUBO problems.
        solver: Annealer bound to the shared schedule.
        x: Clamped input of shape ``(n_in,)``.
        target: Target spins of shape ``(n_output,)``.
        beta: Strictly positive nudging coefficient.
        pair_seed: Seed used by both phases of this sample.

    Returns:
        The selected free and nudged spin states, one vector per layer.

    Raises:
        ValueError: If ``beta`` is not strictly positive, which would make the
            EP gradient undefined.
    """
    if not beta > 0:
        raise ValueError("the nudged phase requires a strictly positive beta")

    # Fields depend on parameters but must not enter the autograd graph: the
    # solver is not differentiable, so gradients flow only through the energy
    # recomputed on the returned spins.
    with torch.no_grad():
        free_qubo, _ = model.qubo(x, target, 0.0)
        nudged_qubo, _ = model.qubo(x, target, beta)

    free_state, free_row = _solve_phase(solver, free_qubo, pair_seed, None)
    nudged_state, _ = _solve_phase(solver, nudged_qubo, pair_seed, free_row)

    free_states = tuple(model.split_state(free_state))
    nudged_states = tuple(model.split_state(nudged_state))
    return PhasePair(free_states=free_states, nudged_states=nudged_states)


def train_epoch(
    model: IsingEPModel,
    optimizer: torch.optim.Optimizer,
    solver: KaiwuSolver,
    images: Any,
    labels: Any,
    order: Any,
    *,
    beta: float,
    batch_size: int,
    pair_seed_fn: Callable[[int], int],
) -> dict[str, float]:
    """Train for one epoch, stepping once per batch.

    Every sample in a batch is solved against the *same* parameters, then a
    single optimizer step is applied to the mean loss. Stepping mid-batch would
    invalidate the phase pairs already collected, because their solver problems
    were built from the pre-step parameters.

    The loss is ``(E_nudged - E_free) / beta``. Differentiating it reproduces
    the two-phase EP estimator on every layer: since the spins are constants,
    the gradient of layer ``l`` is the free-minus-nudged correlation difference
    ``(s_{l-1}^free (x) s_l^free - s_{l-1}^nudged (x) s_l^nudged) / beta``.
    Autograd therefore replaces the hand-written gradient formula.

    Args:
        model: Network whose parameters are updated.
        optimizer: Torch optimizer, normally ``SGD``.
        solver: Annealer bound to the shared schedule.
        images: Training images indexed by ``order``.
        labels: Integer class labels aligned with ``images``.
        order: Sample indices visited this epoch.
        beta: Strictly positive nudging coefficient.
        batch_size: Samples sharing one parameter state.
        pair_seed_fn: Maps a sample index to its paired free/nudged seed.

    Returns:
        Epoch-averaged flip statistics plus ``n_samples`` and ``batch_updates``.
    """
    indices = np.asarray(order, dtype=np.int64).reshape(-1)
    images_array = np.asarray(images, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    totals = {"hidden_flip": 0.0, "output_flip": 0.0, "zero_update": 0.0, "hidden_zero_update": 0.0}
    updates = 0

    for start in range(0, indices.size, batch_size):
        batch = indices[start : start + batch_size]
        losses: list[torch.Tensor] = []
        for sample_index in batch.tolist():
            x = torch.as_tensor(
                images_array[sample_index].reshape(-1),
                dtype=model.dtype,
                device=model.device,
            )
            target = model.target_spins(int(labels_array[sample_index]))
            pair = solve_phases(model, solver, x, target, beta, pair_seed_fn(sample_index))
            losses.append(
                (
                    model.energy(x, pair.nudged_states)
                    - model.energy(x, pair.free_states)
                )
                / beta
            )
            for key, value in pair_dynamics(pair.free_states, pair.nudged_states).items():
                totals[key] += value

        optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        optimizer.step()
        updates += 1

    stats = {key: value / indices.size for key, value in totals.items()}
    stats["n_samples"] = float(indices.size)
    stats["batch_updates"] = float(updates)
    return stats
