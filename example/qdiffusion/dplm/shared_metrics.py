"""Compatibility wrapper for example DPLM metrics helpers."""

try:
    from .utils.metrics import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - direct script-path compatibility
    from utils.metrics import *  # noqa: F401,F403
