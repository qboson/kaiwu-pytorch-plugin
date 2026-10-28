# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Inspect and toggle KPP usage statistics.

KPP reports anonymous usage statistics only from its own sampler call path. The
global switch can be read with ``is_usage_stats_enabled()``, changed at runtime
with ``enable_usage_stats()`` / ``disable_usage_stats()``, or set before the
process starts through ``KPP_STATS_ENABLED``. A code-level setting takes
precedence over the environment variable.

This example prints the current state, toggles it, and shows the environment
variable in child processes; no sampler, dataset, or license is needed.
"""

from __future__ import annotations

import os
import subprocess
import sys

from kaiwu.torch_plugin import (
    disable_usage_stats,
    enable_usage_stats,
    is_usage_stats_enabled,
)

CHILD_PROBE = (
    "from kaiwu.torch_plugin import is_usage_stats_enabled as f; "
    "print('enabled' if f() else 'disabled')"
)


def state() -> str:
    """Returns the current switch state as a short string."""
    return "enabled" if is_usage_stats_enabled() else "disabled"


def child_state(value: str) -> str:
    """Runs a child interpreter with ``KPP_STATS_ENABLED`` set to ``value``."""
    env = dict(os.environ, KPP_STATS_ENABLED=value)
    completed = subprocess.run(
        [sys.executable, "-c", CHILD_PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def main() -> None:
    """Prints and checks the runtime and environment-variable switches."""
    print("current:", state())

    disable_usage_stats()
    print("after disable:", state())
    assert state() == "disabled"

    enable_usage_stats()
    print("after enable:", state())
    assert state() == "enabled"

    enabled_child = child_state("true")
    disabled_child = child_state("false")
    print("child with KPP_STATS_ENABLED=true:", enabled_child)
    print("child with KPP_STATS_ENABLED=false:", disabled_child)
    assert enabled_child == "enabled"
    assert disabled_child == "disabled"
    print("usage-statistics switch OK")


if __name__ == "__main__":
    main()
