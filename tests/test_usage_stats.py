# -*- coding: utf-8 -*-
"""
Unit tests for kpp usage statistics module.

Test focus:
1. kpp_caller_context context manager (set/restore, nesting, exception safety)
2. Global switch (env var / code API / priority / disabled skip)
"""
# pylint: disable=protected-access,import-outside-toplevel

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _get_us():
    """Always import and return the live usage_stats module."""
    import kaiwu.torch_plugin.usage_stats
    return sys.modules['kaiwu.torch_plugin.usage_stats']


class TestKppCallerContext(unittest.TestCase):
    """Test the kpp_caller_context context manager."""

    def test_context_manager_does_not_raise(self):
        """Context manager executes without error (even if kaiwu SDK unavailable)."""
        us = _get_us()
        with us.kpp_caller_context():
            pass  # Should not raise

    def test_context_manager_restores_on_exception(self):
        """Context manager restores state even when block raises."""
        us = _get_us()
        ctx = us._kpp_caller_context
        initial = getattr(ctx, "source", None) if ctx else None

        with self.assertRaises(RuntimeError):
            with us.kpp_caller_context():
                raise RuntimeError("test")

        final = getattr(ctx, "source", None) if ctx else None
        self.assertEqual(initial, final)

    def test_nested_context_does_not_corrupt(self):
        """Nested calls preserve outer context."""
        us = _get_us()
        ctx = us._kpp_caller_context
        if ctx is None:
            self.skipTest("kaiwu SDK not available")

        ctx.source = None
        with us.kpp_caller_context("outer"):
            self.assertEqual(ctx.source, "outer")
            with us.kpp_caller_context("inner"):
                self.assertEqual(ctx.source, "inner")
            self.assertEqual(ctx.source, "outer")
        self.assertIsNone(ctx.source)

    def test_custom_source_applied(self):
        """Custom source value is set and restored."""
        us = _get_us()
        ctx = us._kpp_caller_context
        if ctx is None:
            self.skipTest("kaiwu SDK not available")

        ctx.source = None
        with us.kpp_caller_context("myframework"):
            self.assertEqual(ctx.source, "myframework")
        self.assertIsNone(ctx.source)

    def test_noop_when_disabled(self):
        """When disabled, context manager does not modify source."""
        us = _get_us()
        ctx = us._kpp_caller_context

        us.disable_usage_stats()
        try:
            if ctx is not None:
                ctx.source = None
            with us.kpp_caller_context():
                if ctx is not None:
                    self.assertIsNone(ctx.source)
        finally:
            us.enable_usage_stats()


class TestStatsSwitch(unittest.TestCase):
    """Test the global statistics switch."""

    def setUp(self):
        """Reset switch state before each test."""
        us = _get_us()
        us._stats_enabled = None
        us._env_stats_cache = None

    def tearDown(self):
        """Reset after each test to avoid pollution."""
        us = _get_us()
        us._stats_enabled = None
        us._env_stats_cache = None
        os.environ.pop("KPP_STATS_ENABLED", None)

    def test_default_enabled(self):
        """Stats are enabled by default."""
        us = _get_us()
        self.assertTrue(us.is_usage_stats_enabled())

    def test_env_var_disable(self):
        """Stats can be disabled via KPP_STATS_ENABLED env var."""
        us = _get_us()
        os.environ["KPP_STATS_ENABLED"] = "false"
        self.assertFalse(us.is_usage_stats_enabled())

    def test_code_disable(self):
        """Stats can be disabled via code API."""
        us = _get_us()
        us.disable_usage_stats()
        self.assertFalse(us.is_usage_stats_enabled())

    def test_code_enable(self):
        """Stats can be re-enabled via code API."""
        us = _get_us()
        us.disable_usage_stats()
        self.assertFalse(us.is_usage_stats_enabled())
        us.enable_usage_stats()
        self.assertTrue(us.is_usage_stats_enabled())

    def test_code_overrides_env(self):
        """Code-level setting takes precedence over env var."""
        us = _get_us()
        os.environ["KPP_STATS_ENABLED"] = "false"
        us.enable_usage_stats()
        self.assertTrue(us.is_usage_stats_enabled())

    def test_env_var_true_values(self):
        """Various truthy env var values are accepted."""
        for val in ("true", "1", "yes", "on"):
            os.environ["KPP_STATS_ENABLED"] = val
            us = _get_us()
            us._env_stats_cache = None
            self.assertTrue(us.is_usage_stats_enabled(), f"Failed for value: {val}")

    def test_env_var_false_values(self):
        """Non-truthy env var values disable stats."""
        for val in ("false", "0", "no", "off", "anything"):
            os.environ["KPP_STATS_ENABLED"] = val
            us = _get_us()
            us._env_stats_cache = None
            self.assertFalse(us.is_usage_stats_enabled(), f"Failed for value: {val}")


if __name__ == "__main__":
    unittest.main()
