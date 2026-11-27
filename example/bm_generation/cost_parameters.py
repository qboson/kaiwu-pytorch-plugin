# SPDX-License-Identifier: Apache-2.0
"""Validate BM cost weights independently of model construction."""

import math


def _cost_weight(value, name):
    message = f"{name} must be a finite non-negative number"
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(message)
    return value


def validate_cost_parameters(alpha, beta):
    """Return a new float-valued mapping after validating both cost weights.

    Values follow Python's float conversion semantics. Zero is allowed, and
    there is no upper bound other than finiteness. Invalid conversion, negative
    values and non-finite values raise a ValueError naming the parameter.
    """
    return {
        "alpha": _cost_weight(alpha, "alpha"),
        "beta": _cost_weight(beta, "beta"),
    }
