"""Free-phase-only evaluation of a trained checkpoint.

Evaluation never uses the nudged phase: the nudge injects the label into the
output field, so running it at test time would leak the answer. One free solve
per test image is all that is needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ising_ep.kaiwu_solver import KaiwuSolver
from ising_ep.model import IsingEPModel

from .metrics import classification_metrics


def evaluate_free_only(
    model: IsingEPModel,
    solver: KaiwuSolver,
    images: Any,
    labels: Any,
    *,
    eval_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Classify every test image with a single free-phase solve.

    Args:
        model: Trained network to evaluate.
        solver: Annealer bound to the shared schedule.
        images: Test images, one row per sample.
        labels: Integer class labels aligned with ``images``.
        eval_seed: Base seed; sample ``i`` uses ``eval_seed + i`` so each image
            gets its own reproducible anneal.

    Returns:
        The classification metrics and one per-sample record holding the label,
        the prediction and whether the class scores tied.
    """
    images_array = np.asarray(images, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)

    # The free phase has no target, so the output field is b_o alone. Passing a
    # constant target with beta = 0 keeps the label out of the problem.
    unused_target = torch.ones(model.n_output, dtype=model.dtype, device=model.device)

    predictions: list[int] = []
    ties: list[bool] = []
    records: list[dict[str, Any]] = []

    for index, (image, label) in enumerate(zip(images_array, labels_array, strict=True)):
        x = torch.as_tensor(image.reshape(-1), dtype=model.dtype, device=model.device)
        with torch.no_grad():
            qubo, _ = model.qubo(x, unused_target, 0.0)

        best = solver.solve(qubo, seed=int(eval_seed) + index)[0]
        states = model.split_state(best[:-1])

        prediction, tied = model.predict(states)
        predictions.append(int(prediction))
        ties.append(bool(tied))
        records.append(
            {
                "index": index,
                "label": int(label),
                "prediction": int(prediction),
                "tied": bool(tied),
            }
        )

    metrics = classification_metrics(labels_array, predictions, ties, n_classes=model.n_classes)
    return metrics, records
