"""Tests for the license-aware behavior of the feature-selection example.

The `sa` and `kaiwu_cim` solvers both route through the Kaiwu SDK and need
a license; without one the SDK drops into an interactive credential prompt
that crashes non-interactive runs. The example now skips those solvers
(linear_regression_solvers.py) or exits with instructions
(neural_network_kaiwu_cim.py) when the environment has no credentials.
"""

import os
import sys

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../example/feature_selection")
    ),
)

import kaiwu

# Extend namespace package __path__ so that local src is preferred
kaiwu.__path__ = list(kaiwu.__path__) + [os.path.join(src_root, "kaiwu")]

for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith(
        "kaiwu.torch_plugin."
    ):
        del sys.modules[module_name]
if hasattr(kaiwu, "torch_plugin"):
    delattr(kaiwu, "torch_plugin")

import pytest

import kaiwu_license


@pytest.mark.parametrize(
    "user_id,sdk_code,expected",
    [
        (None, None, False),
        ("user-1", None, False),
        (None, "code-1", False),
        ("", "code-1", False),
        ("user-1", "code-1", True),
    ],
)
def test_has_license_env(user_id, sdk_code, expected, monkeypatch):
    """The check requires both credentials to be non-empty."""
    monkeypatch.setenv("LICENSE_USER_ID", user_id or "")
    monkeypatch.setenv("LICENSE_SDK_CODE", sdk_code or "")
    if user_id is None:
        monkeypatch.delenv("LICENSE_USER_ID", raising=False)
    if sdk_code is None:
        monkeypatch.delenv("LICENSE_SDK_CODE", raising=False)
    assert kaiwu_license.has_license_env() is expected


def test_init_without_credentials_is_a_no_op(monkeypatch):
    """_init_kaiwu_license_from_env returns silently without credentials."""
    monkeypatch.delenv("LICENSE_USER_ID", raising=False)
    monkeypatch.delenv("LICENSE_SDK_CODE", raising=False)
    assert kaiwu_license._init_kaiwu_license_from_env() is None


def test_importing_example_modules_has_no_side_effects():
    """The solver scripts import cleanly without touching the license."""
    import linear_regression_solvers  # noqa: F401
    import neural_network_kaiwu_cim  # noqa: F401

    assert callable(linear_regression_solvers.main)
    assert callable(neural_network_kaiwu_cim.main)
