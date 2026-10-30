"""Bridge Kaiwu SDK 1.3.1 and 1.4.x interface changes."""

import kaiwu as kw

try:
    from kaiwu.preprocess import PrecisionReducer
except ImportError:
    from kaiwu.cim import PrecisionReducer

try:
    from kaiwu.core import qubo_matrix_to_ising_matrix
except ImportError:
    from kaiwu.conversion import qubo_matrix_to_ising_matrix

__all__ = ["PrecisionReducer", "normalize_task_mode", "qubo_matrix_to_ising_matrix"]


def normalize_task_mode(task_mode: str | None) -> str | None:
    """Translate old and new CIM mode names for the installed Kaiwu SDK.

    None preserves the SDK's default mode. Unknown modes are rejected before
    creating a cloud task, so a typo cannot select a different computation.
    """
    if task_mode is None:
        return None
    modes = {
        "quota": "OPTIMIZATION", "optimization": "OPTIMIZATION",
        "sample": "SAMPLING", "sampling": "SAMPLING",
    }
    try:
        name = modes[task_mode.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported Kaiwu task mode: {task_mode}") from error
    task_modes = getattr(kw.cim, "TaskMode", None)
    if task_modes is not None:
        return getattr(task_modes, name)
    return {"OPTIMIZATION": "quota", "SAMPLING": "sample"}[name]
