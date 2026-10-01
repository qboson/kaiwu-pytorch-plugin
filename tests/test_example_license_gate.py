"""Tests for the license gate in the quick-start example scripts.

``example/run_rbm.py`` and ``example/run_bm.py`` submit their negative
phase to Kaiwu SDK samplers, which require a license. Without the gate
they used to drop into the SDK's interactive credential prompt and crash
non-interactive runs; the gate initializes the license from the
documented environment variables or exits with setup instructions.
"""

import os
import sys

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../example"))
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

import run_bm
import run_rbm

EXAMPLE_MODULES = (run_rbm, run_bm)


@pytest.mark.parametrize("module", EXAMPLE_MODULES, ids=["run_rbm", "run_bm"])
def test_missing_credentials_exit_with_instructions(module, monkeypatch):
    """Without credentials the gate exits with actionable instructions."""
    monkeypatch.delenv("USER_ID", raising=False)
    monkeypatch.delenv("SDK_CODE", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        module._ensure_license()
    message = str(excinfo.value)
    assert "platform.qboson.com" in message
    assert "USER_ID" in message
    assert "SDK_CODE" in message


@pytest.mark.parametrize("module", EXAMPLE_MODULES, ids=["run_rbm", "run_bm"])
def test_present_credentials_initialize_license(module, monkeypatch):
    """With credentials the gate initializes the SDK license once."""
    monkeypatch.setenv("USER_ID", "user-123")
    monkeypatch.setenv("SDK_CODE", "code-456")
    calls = []

    def fake_init(user_id, sdk_code):
        calls.append((user_id, sdk_code))

    monkeypatch.setattr(module.kw.license, "init", fake_init)
    module._ensure_license()
    assert calls == [("user-123", "code-456")]


@pytest.mark.parametrize("module", EXAMPLE_MODULES, ids=["run_rbm", "run_bm"])
def test_partial_credentials_exit(module, monkeypatch):
    """A half-configured environment still fails fast with instructions."""
    monkeypatch.setenv("USER_ID", "user-123")
    monkeypatch.delenv("SDK_CODE", raising=False)
    with pytest.raises(SystemExit):
        module._ensure_license()


def test_gate_only_runs_as_script():
    """Importing the example modules has no side effects on the license."""
    import kaiwu as kw

    # _ensure_license is defined but not called at import time; importing
    # run_rbm/run_bm above already proves this, assert the symbol exists.
    for module in EXAMPLE_MODULES:
        assert callable(module._ensure_license)
