"""Packaging metadata must match the documented and installable Python range.

The docs state that only Python 3.10 is supported, and the pinned
``torch==2.7.0`` itself requires Python >= 3.9, so ``requires-python`` cannot
advertise 3.8.
"""

import os
import re
from importlib import metadata

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def declared_requires_python():
    """Reads the ``requires-python`` value from pyproject.toml."""
    path = os.path.join(REPO_ROOT, "pyproject.toml")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r'^requires-python\s*=\s*"([^"]+)"', text, re.M)
    assert match is not None, "requires-python is not declared"
    return match.group(1)


def lower_bound(specifier):
    """Extracts the numeric lower bound from a ``>=X.Y`` specifier."""
    match = re.match(r">=\s*(\d+)\.(\d+)", specifier)
    assert match is not None, f"unsupported specifier: {specifier}"
    return (int(match.group(1)), int(match.group(2)))


def test_declares_the_documented_python_version():
    """The metadata advertises 3.10, matching the documentation."""
    assert declared_requires_python() == ">=3.10"


def test_declared_floor_covers_the_pinned_torch():
    """The declared floor is not lower than torch's own requires-python."""
    torch_floor = lower_bound(metadata.metadata("torch").get("Requires-Python"))
    assert lower_bound(declared_requires_python()) >= torch_floor
