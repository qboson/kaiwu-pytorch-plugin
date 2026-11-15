"""Classification, two-phase dynamics and learning-evidence metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

#: A parameter norm must move further than this to count as having learned.
PARAM_CHANGE_THRESHOLD = 1e-3

#: At least one epoch must flip the hidden state on this fraction of samples.
HIDDEN_UPDATE_THRESHOLD = 0.05

#: Floor for the denominator when normalising a parameter change.
RELATIVE_EPS = 1e-8


def _array(value: Any, dtype: Any = np.float64) -> np.ndarray:
    """Convert a Torch tensor or array-like to a NumPy array.

    Args:
        value: Tensor or array-like input.
        dtype: Target NumPy dtype.

    Returns:
        A NumPy array view of ``value``.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    ties: Sequence[bool],
    *,
    n_classes: int = 10,
) -> dict[str, Any]:
    """Score predictions with accuracy, tie rate, confusion and per-class detail.

    The tie rate is reported separately because a tie means the output block
    sums were equal and the prediction came from the local-field fallback, which
    is a sign the solver did not commit to one class.

    Args:
        labels: True class indices.
        predictions: Predicted class indices, aligned with ``labels``.
        ties: Whether each prediction needed the tie-break, aligned with
            ``labels``.
        n_classes: Number of classes.

    Returns:
        Mapping with ``n``, ``correct``, ``accuracy``, ``tie_count``,
        ``tie_rate``, ``confusion`` and ``per_class``.
    """
    y = _array(labels, np.int64).reshape(-1)
    predicted = _array(predictions, np.int64).reshape(-1)
    tied = _array(ties, bool).reshape(-1)

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y, predicted), 1)
    totals = confusion.sum(axis=1)
    correct_per_class = np.diagonal(confusion)

    return {
        "n": int(y.size),
        "correct": int(correct_per_class.sum()),
        "accuracy": float(correct_per_class.sum() / y.size),
        "tie_count": int(tied.sum()),
        "tie_rate": float(tied.mean()),
        "confusion": confusion.tolist(),
        "per_class": [
            {
                "class": index,
                "correct": int(correct_per_class[index]),
                "total": int(totals[index]),
                "accuracy": float(correct_per_class[index] / totals[index])
                if totals[index]
                else 0.0,
            }
            for index in range(n_classes)
        ],
    }


def pair_dynamics(free_states: Sequence[Any], nudged_states: Sequence[Any]) -> dict[str, float]:
    """Measure how far the nudge moved the solver state for one sample.

    These rates diagnose a silent failure mode: if the nudged phase returns the
    same state as the free phase, the EP gradient is exactly zero and the sample
    contributes nothing to learning. Every layer but the last is hidden.

    Args:
        free_states: Free-phase spin vectors, output last.
        nudged_states: Nudged-phase spin vectors, output last.

    Returns:
        Mapping with ``hidden_flip``, ``output_flip``, ``zero_update`` and
        ``hidden_zero_update``.
    """
    free_layers = [_array(state).reshape(-1) for state in free_states]
    nudged_layers = [_array(state).reshape(-1) for state in nudged_states]
    free_hidden = np.concatenate(free_layers[:-1]) if len(free_layers) > 1 else np.zeros(0)
    nudged_hidden = (
        np.concatenate(nudged_layers[:-1]) if len(nudged_layers) > 1 else np.zeros(0)
    )
    free_output = free_layers[-1]
    nudged_output = nudged_layers[-1]
    hidden_equal = bool(np.array_equal(free_hidden, nudged_hidden))

    return {
        "hidden_flip": float(np.mean(free_hidden != nudged_hidden)) if free_hidden.size else 0.0,
        "output_flip": float(np.mean(free_output != nudged_output)),
        "zero_update": float(hidden_equal and np.array_equal(free_output, nudged_output)),
        "hidden_zero_update": float(hidden_equal),
    }


def l2_distance(left: Any, right: Any) -> float:
    """Return the Euclidean distance between two flattened arrays.

    Args:
        left: First array or tensor.
        right: Second array or tensor.

    Returns:
        The scalar distance ``||left - right||``.
    """
    return float(np.linalg.norm(_array(left).reshape(-1) - _array(right).reshape(-1)))


def relative_change(initial: Any, final: Any, *, epsilon: float = RELATIVE_EPS) -> dict[str, float]:
    """Report how far a parameter moved, absolutely and relative to its scale.

    Args:
        initial: Parameter before training.
        final: Parameter after training.
        epsilon: Floor for the denominator, guarding an all-zero initial value.

    Returns:
        Mapping with ``absolute``, ``relative`` and the ``epsilon`` used.
    """
    absolute = l2_distance(initial, final)
    denominator = max(float(np.linalg.norm(_array(initial))), float(epsilon))
    return {"absolute": absolute, "relative": absolute / denominator, "epsilon": float(epsilon)}


def learning_evidence(
    *,
    weight_changes: Sequence[float],
    hidden_nonzero_rates: Sequence[float],
) -> dict[str, Any]:
    """Check that training actually changed the model.

    A run can finish without error and still have learned nothing, for instance
    if every nudged solve returned the free state. This records the signals that
    rule that out: every weight matrix moved, and at least one epoch flipped
    hidden spins often enough to carry gradient.

    Args:
        weight_changes: Distance moved by each weight matrix over training,
            input layer first.
        hidden_nonzero_rates: Per-epoch fraction of samples whose hidden state
            differed between the two phases.

    Returns:
        Mapping with the measured values, the thresholds applied and ``passed``.
    """
    changes = [float(value) for value in weight_changes]
    rates = [float(value) for value in hidden_nonzero_rates]
    return {
        "weight_changes": changes,
        "param_threshold": PARAM_CHANGE_THRESHOLD,
        "hidden_nonzero_rates": rates,
        "hidden_update_threshold": HIDDEN_UPDATE_THRESHOLD,
        "passed": bool(
            bool(changes)
            and all(change > PARAM_CHANGE_THRESHOLD for change in changes)
            and any(rate >= HIDDEN_UPDATE_THRESHOLD for rate in rates)
        ),
    }


def population_stats(values: Sequence[float]) -> dict[str, float]:
    """Summarise scalar values with the population standard deviation.

    Args:
        values: One scalar per seed. A single-seed run is allowed and reports a
            standard deviation of zero.

    Returns:
        Mapping with ``mean``, ``std``, ``min`` and ``max``.

    Raises:
        ValueError: If ``values`` is empty.
    """
    array = _array(list(values)).reshape(-1)
    if array.size == 0:
        raise ValueError("cannot aggregate an empty sequence")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }
