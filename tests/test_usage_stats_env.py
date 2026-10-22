# -*- coding: utf-8 -*-
"""Environment-variable reflection tests for kpp usage statistics.

These tests load ``usage_stats.py`` standalone (via importlib) so they run
without the heavier ``kaiwu.torch_plugin`` package dependencies (torch/numpy).
This isolates the pure-Python switch logic.
"""

import importlib.util
import os
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(
    _HERE, "..", "src", "kaiwu", "torch_plugin", "usage_stats.py"
)


def _load_usage_stats():
    """Load the pure-Python usage_stats module directly from source."""
    spec = importlib.util.spec_from_file_location(
        "kaiwu_torch_plugin_usage_stats_standalone", _SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEnvVarReflection(unittest.TestCase):
    """Verify KPP_STATS_ENABLED is honored even after the first read."""

    def setUp(self):
        self.us = _load_usage_stats()
        self.us._stats_enabled = None
        self.us._env_stats_cache = None
        os.environ.pop("KPP_STATS_ENABLED", None)

    def tearDown(self):
        os.environ.pop("KPP_STATS_ENABLED", None)
        self.us._stats_enabled = None
        self.us._env_stats_cache = None

    def test_default_enabled(self):
        """Stats are enabled by default when the env var is unset."""
        self.assertTrue(self.us.is_usage_stats_enabled())

    def test_env_var_disables(self):
        """An explicit "false" env var disables stats."""
        os.environ["KPP_STATS_ENABLED"] = "false"
        self.assertFalse(self.us.is_usage_stats_enabled())

    def test_runtime_env_change_is_reflected(self):
        """A runtime flip of KPP_STATS_ENABLED must take effect.

        Regression test: the switch used to cache the first parsed env-var
        value, so later runtime changes were silently ignored.
        """
        # Default (env unset) -> enabled, which populates the internal state.
        self.assertTrue(self.us.is_usage_stats_enabled())
        # Operator flips the switch off at runtime.
        os.environ["KPP_STATS_ENABLED"] = "false"
        self.assertFalse(self.us.is_usage_stats_enabled())

    def test_runtime_env_change_back_to_true(self):
        """A runtime flip back to "true" is also honored."""
        os.environ["KPP_STATS_ENABLED"] = "false"
        self.assertFalse(self.us.is_usage_stats_enabled())
        os.environ["KPP_STATS_ENABLED"] = "true"
        self.assertTrue(self.us.is_usage_stats_enabled())

    def test_code_override_still_wins(self):
        """The code-level switch keeps precedence over the env var."""
        os.environ["KPP_STATS_ENABLED"] = "false"
        self.us.enable_usage_stats()
        self.assertTrue(self.us.is_usage_stats_enabled())
        self.us.disable_usage_stats()
        self.assertFalse(self.us.is_usage_stats_enabled())


if __name__ == "__main__":
    unittest.main()
