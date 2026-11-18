"""Private validation helpers for parameter clip windows."""

from __future__ import annotations

import math


def validate_clip_range(name, bounds):
    """Validate one clamp window.

    Args:
        name: Parameter name used in the error message.
        bounds: One or two finite bounds, ordered as ``(min, max)`` when
            two are given.

    Returns:
        tuple: The bounds converted to floats.

    Raises:
        TypeError: If ``bounds`` is not a sequence of one or two numbers.
        ValueError: If a bound is non-finite or the window is inverted.
    """
    try:
        values = tuple(float(bound) for bound in bounds)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number or a (min, max) pair") from exc
    if len(values) not in (1, 2):
        raise TypeError(f"{name} must be a number or a (min, max) pair")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")
    if len(values) == 2 and values[0] > values[1]:
        raise ValueError(f"{name} must be ordered (min <= max)")
    return values
