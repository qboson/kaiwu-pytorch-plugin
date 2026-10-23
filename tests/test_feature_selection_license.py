"""License-initialisation contract for the feature-selection example."""

import importlib.util
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_PATH = os.path.join(REPO_ROOT, "example", "feature_selection", "kaiwu_license.py")

_spec = importlib.util.spec_from_file_location("feature_selection_kaiwu_license", MODULE_PATH)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)

import kaiwu.license as license_manager  # noqa: E402


@pytest.fixture()
def recorder(monkeypatch):
    """Replaces the SDK initialiser and records its arguments."""
    calls = []

    def fake_init(*args):
        calls.append(args)

    monkeypatch.setattr(license_manager, "init", fake_init)
    return calls


def test_present_environment_variables_initialise_the_sdk(monkeypatch, recorder):
    """Both variables are forwarded to the SDK initialiser."""
    monkeypatch.setenv("LICENSE_USER_ID", "user-1")
    monkeypatch.setenv("LICENSE_SDK_CODE", "code-1")
    module._init_kaiwu_license_from_env()
    assert recorder == [("user-1", "code-1")]


def test_sdk_failure_is_wrapped_with_a_helpful_message(monkeypatch):
    """Any SDK exception becomes a RuntimeError that keeps the original cause."""

    def failing_init(*_args):
        raise KeyError("boom")

    monkeypatch.setattr(license_manager, "init", failing_init)
    monkeypatch.setenv("LICENSE_USER_ID", "user-1")
    monkeypatch.setenv("LICENSE_SDK_CODE", "code-1")
    with pytest.raises(RuntimeError, match="Kaiwu license initialization failed") as info:
        module._init_kaiwu_license_from_env()
    assert isinstance(info.value.__cause__, KeyError)
