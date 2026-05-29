"""Compatibility wrapper for example DPLM runtime helpers."""

try:
    from .utils.runtime import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - direct script-path compatibility
    from utils.runtime import *  # noqa: F401,F403
