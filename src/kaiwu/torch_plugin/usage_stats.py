# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Implicit usage statistics for kpp.

Statistics are only triggered within the kpp call path
(``AbstractBoltzmannMachine.sample()``); direct calls to kaiwu SDK
optimizers are never tracked, preventing false attribution.

How it works
------------
kaiwu SDK's ``@track_data`` decorator inspects ``_caller_context``
(``threading.local``); when a *source* attribute is present, the
``alg_name`` is automatically prefixed (e.g. ``sa`` -> ``kpp_sa``).
``CIMOptimizer._create_task()`` reads the same variable so that
task payloads carry the ``task_source_detail`` field.

Global switch
-------------
Controlled by the environment variable ``KPP_STATS_ENABLED``
(default ``"true"``).  The switch can also be toggled at runtime via
``enable_usage_stats()`` / ``disable_usage_stats()``.
"""

import logging
import os
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache: shared _caller_context from kaiwu SDK
# ---------------------------------------------------------------------------
try:
    from kaiwu.license._track_data import _caller_context as _kpp_caller_context
except ImportError:
    _kpp_caller_context = None

# ---------------------------------------------------------------------------
# Global switch
# ---------------------------------------------------------------------------
# Env var: KPP_STATS_ENABLED; values "true"/"1" enable, anything else disables.
# Enabled by default.  The code-level _stats_enabled takes precedence over
# the environment variable.
# ---------------------------------------------------------------------------
_stats_enabled: Optional[bool] = None  # None means follow env var
# Cached env-var parse result to avoid os.environ.get() on every call
_env_stats_cache: Optional[bool] = None


def _is_stats_enabled() -> bool:
    """Return whether the statistics feature is enabled.

    Priority: code-level setting > env var > default (True).
    """
    global _env_stats_cache  # pylint: disable=global-statement
    if _stats_enabled is not None:
        return _stats_enabled
    if _env_stats_cache is not None:
        return _env_stats_cache
    env_val = os.environ.get("KPP_STATS_ENABLED", "true").lower()
    _env_stats_cache = env_val in ("true", "1", "yes", "on")
    return _env_stats_cache


def enable_usage_stats() -> None:
    """Enable kpp usage statistics (enabled by default; usually not needed)."""
    global _stats_enabled, _env_stats_cache  # pylint: disable=global-statement
    _stats_enabled = True
    _env_stats_cache = None  # reset cache so code setting takes precedence
    logger.info("kpp usage stats enabled")


def disable_usage_stats() -> None:
    """Disable kpp usage statistics. No data will be reported after disabling."""
    global _stats_enabled  # pylint: disable=global-statement
    _stats_enabled = False
    logger.info("kpp usage stats disabled")


def is_usage_stats_enabled() -> bool:
    """Return whether usage statistics are currently enabled."""
    return _is_stats_enabled()


# ---------------------------------------------------------------------------
# Caller context management
# ---------------------------------------------------------------------------
# kpp sets ``_caller_context.source = "kpp"`` inside sample();
# kaiwu SDK's @track_data decorator and CIMOptimizer._create_task() read this
# value to automatically prefix alg_name (``sa`` -> ``kpp_sa``) and include
# ``task_source_detail`` in CIM task payloads.
# A save/restore pattern supports nested sample() calls within the same thread.
# ---------------------------------------------------------------------------

@contextmanager
def kpp_caller_context(source: str = "kpp"):
    """Context manager that sets the caller identity and restores on exit.

    kpp wraps ``sampler.solve()`` with ``with kpp_caller_context():`` inside
    ``sample()``.  kaiwu SDK's ``@track_data`` decorator and
    ``CIMOptimizer._create_task()`` read this value to prefix ``alg_name``
    (``sa`` -> ``kpp_sa``) and carry ``task_source_detail`` in CIM tasks.
    Supports nested ``sample()`` calls within the same thread.
    """
    if not _is_stats_enabled() or _kpp_caller_context is None:
        yield
        return

    old_value = getattr(_kpp_caller_context, "source", None)
    _kpp_caller_context.source = source
    try:
        yield
    finally:
        _kpp_caller_context.source = old_value
