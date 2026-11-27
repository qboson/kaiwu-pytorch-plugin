"""Non-interactive license setup for explicitly marked real SDK tests."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from unittest.mock import patch

import pytest


def pytest_configure(config):
    """Register the marker even when the repository's legacy pytest.ini is used."""
    config.addinivalue_line(
        "markers", "kaiwu_license: runs the real SDK solver with a valid Kaiwu license"
    )


def initialize_kaiwu_license(environment, initializer=None):
    """Initialize once from credentials, or use an existing local license.

    CI requires explicit credentials. Never expose SDK output or exception text,
    which may contain credential values, when activation fails.
    """
    user_id = environment.get("KAIWU_USER_ID", "").strip()
    sdk_code = environment.get("KAIWU_SDK_CODE", "").strip()
    required = environment.get("KAIWU_REQUIRE_LICENSE") == "1"
    if not user_id or not sdk_code:
        if required or user_id or sdk_code:
            raise RuntimeError(
                "Real Kaiwu tests require both KAIWU_USER_ID and KAIWU_SDK_CODE."
            )
        return  # A locally activated SDK remains supported.
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            if initializer is None:
                from kaiwu.license import init  # pylint: disable=import-outside-toplevel
                initializer = init
            initializer(user_id, sdk_code)
    except Exception:
        raise RuntimeError(
            "Kaiwu license initialization failed; check credentials and license-server access."
        ) from None


def reject_license_input(*_args, **_kwargs):
    """Fail immediately if the SDK falls back to interactive activation."""
    raise RuntimeError(
        "Interactive Kaiwu license input is disabled; initialize a valid license first."
    )


@pytest.fixture(scope="session")
def activated_kaiwu_license():
    """Activate at most once, only when a marked test is actually selected."""
    try:
        initialize_kaiwu_license(os.environ)
    except RuntimeError as exc:
        # Long pytest tracebacks may include function arguments (os.environ).
        # Keep credentials out of both the console and the JUnit artifact.
        pytest.fail(str(exc), pytrace=False)


@pytest.fixture(autouse=True)
def noninteractive_kaiwu_tests(request):
    """Protect both activation and real solves without altering unmarked tests."""
    if request.node.get_closest_marker("kaiwu_license") is None:
        yield
        return
    with patch("builtins.input", reject_license_input):
        request.getfixturevalue("activated_kaiwu_license")
        yield
