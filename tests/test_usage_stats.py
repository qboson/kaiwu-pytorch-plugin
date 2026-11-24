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
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _get_us():
    """Always import and return the live usage_stats module."""
    import kaiwu.torch_plugin.usage_stats
    return sys.modules['kaiwu.torch_plugin.usage_stats']


def _apply_patch(test, patcher):
    """Apply a test patch and restore it even if setup or the test fails."""
    patcher.start()
    test.addCleanup(patcher.stop)


class TestKppCallerContext(unittest.TestCase):
    """Test plugin behavior with a fresh context independent of the SDK."""

    def setUp(self):
        self.us = _get_us()
        self.context = threading.local()
        _apply_patch(self, patch.multiple(
            self.us, _kpp_caller_context=self.context, _stats_enabled=True,
            _env_stats_cache=None,
        ))

    def test_custom_source_applied(self):
        self.context.source = "existing"
        with self.us.kpp_caller_context("myframework"):
            self.assertEqual(self.context.source, "myframework")
        self.assertEqual(self.context.source, "existing")

    def test_nested_context_does_not_corrupt(self):
        self.context.source = "existing"
        with self.us.kpp_caller_context("outer"):
            with self.us.kpp_caller_context("inner"):
                self.assertEqual(self.context.source, "inner")
            self.assertEqual(self.context.source, "outer")
        self.assertEqual(self.context.source, "existing")

    def test_context_manager_restores_on_exception(self):
        self.context.source = "existing"
        with self.assertRaisesRegex(RuntimeError, "test"):
            with self.us.kpp_caller_context():
                self.assertEqual(self.context.source, "kpp")
                raise RuntimeError("test")
        self.assertEqual(self.context.source, "existing")

    def test_noop_when_disabled(self):
        self.context.source = "existing"
        self.us.disable_usage_stats()
        with self.us.kpp_caller_context():
            self.assertEqual(self.context.source, "existing")

    def test_noop_without_sdk_context(self):
        with patch.object(self.us, "_kpp_caller_context", None):
            with self.us.kpp_caller_context():
                self.assertIsNone(self.us._kpp_caller_context)

    def test_context_isolated_between_threads(self):
        def worker():
            initial = getattr(self.context, "source", None)
            with self.us.kpp_caller_context("worker"):
                active = self.context.source
            return initial, active, self.context.source

        with self.us.kpp_caller_context("main"):
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(pool.submit(worker).result(timeout=5), (None, "worker", None))
            self.assertEqual(self.context.source, "main")


@pytest.mark.integration
class TestNativeCallerContext(unittest.TestCase):
    """Test the shared SDK object without submitting jobs or telemetry."""

    def setUp(self):
        self.us = _get_us()
        self.context = self.us._kpp_caller_context
        if self.context is None:
            self.skipTest("installed Kaiwu SDK does not expose the statistics caller context")
        native = import_module("kaiwu.license._track_data")
        self.assertIs(self.context, native._caller_context)
        _apply_patch(self, patch.object(self.us, "_stats_enabled", True))
        _apply_patch(self, patch.object(self.context, "source", "existing", create=True))

    def test_custom_source_applied(self):
        with self.us.kpp_caller_context("myframework"):
            self.assertEqual(self.context.source, "myframework")
            with ThreadPoolExecutor(max_workers=1) as pool:
                other = pool.submit(getattr, self.context, "source", None).result(timeout=5)
            self.assertIsNone(other, "SDK caller context must be thread local")
        self.assertEqual(self.context.source, "existing")

    def test_nested_context_does_not_corrupt(self):
        with self.us.kpp_caller_context("outer"):
            self.assertEqual(self.context.source, "outer")
            with self.us.kpp_caller_context("inner"):
                self.assertEqual(self.context.source, "inner")
            self.assertEqual(self.context.source, "outer")
            with self.assertRaisesRegex(RuntimeError, "inner exception"):
                with self.us.kpp_caller_context("inner"):
                    self.assertEqual(self.context.source, "inner")
                    raise RuntimeError("inner exception")
            self.assertEqual(self.context.source, "outer")
        self.assertEqual(self.context.source, "existing")


class TestStatsSwitch(unittest.TestCase):
    """Test the global statistics switch."""

    def setUp(self):
        us = _get_us()
        _apply_patch(self, patch.multiple(us, _stats_enabled=None, _env_stats_cache=None))
        _apply_patch(self, patch.dict(os.environ))
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
