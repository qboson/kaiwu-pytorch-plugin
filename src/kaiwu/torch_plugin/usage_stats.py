# -*- coding: utf-8 -*-
# pylint: disable=mixed-line-endings
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
Some kaiwu builds expose a ``_caller_context`` (``threading.local``) object in
``kaiwu.license._track_data``; when a *source* attribute is present on it, the
SDK's ``@track_data`` decorator prefixes ``alg_name`` (e.g. ``sa`` -> ``kpp_sa``)
and CIM task payloads carry the ``task_source_detail`` field.

That hook is optional. The PyPI build pinned in ``requirements/requirements.txt``
(``kaiwu==1.3.1``) does not provide it, so on that build the context manager
degrades to a no-op: sampling still runs, but the SDK never labels the call as
coming from kpp. ``caller_context_available()`` reports which case applies to the
installed build.

Global switch
-------------
Controlled by the environment variable ``KPP_STATS_ENABLED``
(default ``"true"``).  The switch can also be toggled at runtime via
``enable_usage_stats()`` / ``disable_usage_stats()``.

Kaiwu 1.4.3 does not expose ``_caller_context``. In that release this
module leaves SDK calls unchanged; its switch does not control SDK telemetry.
"""

import logging
import os
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache: shared _caller_context from kaiwu SDK
# ---------------------------------------------------------------------------
# Optional hook: only kaiwu builds that ship ``_caller_context`` can attribute a
# sampler call to kpp. When the import fails, kpp_caller_context() is a no-op and
# caller_context_available() returns False.
try:
    from kaiwu.license._track_data import _caller_context as _kpp_caller_context
except ImportError:
    _kpp_caller_context = None


def caller_context_available() -> bool:
    """Return whether the installed kaiwu build exposes the attribution hook.

    When this is ``False``, ``kpp_caller_context()`` cannot label sampler calls as
    originating from kpp (the SDK offers no hook to do so); sampling itself is
    unaffected.

    Returns:
        bool: ``True`` when ``kaiwu.license._track_data._caller_context`` could be
        imported from the installed kaiwu build.
    """
    return _kpp_caller_context is not None

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
    """Disable KPP caller attribution without changing SDK telemetry settings."""
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
