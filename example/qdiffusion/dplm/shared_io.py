"""Compatibility wrapper for example DPLM I/O helpers."""

try:
    from .utils.io import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - direct script-path compatibility
    from utils.io import *  # noqa: F401,F403
