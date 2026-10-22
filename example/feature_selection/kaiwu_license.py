from __future__ import annotations

import os


def has_license_env() -> bool:
    """Return whether Kaiwu license credentials are configured.

    Both the ``sa`` and ``kaiwu_cim`` solvers route through the Kaiwu SDK
    and need an initialized license; without one the SDK drops into an
    interactive credential prompt that crashes non-interactive runs.

    Returns:
        bool: True when both ``LICENSE_USER_ID`` and ``LICENSE_SDK_CODE``
        are set in the environment.
    """
    user_id = os.environ.get("LICENSE_USER_ID")
    sdk_code = os.environ.get("LICENSE_SDK_CODE")
    return bool(user_id and sdk_code)


def _init_kaiwu_license_from_env() -> None:
    """Initialize the Kaiwu license from environment variables.

    Raises:
        RuntimeError: If Kaiwu license initialization fails.
    """
    if not has_license_env():
        return

    import kaiwu.license as license_manager

    user_id = os.environ.get("LICENSE_USER_ID")
    sdk_code = os.environ.get("LICENSE_SDK_CODE")

    try:
        license_manager.init(user_id, sdk_code)
    except Exception as exc:
        raise RuntimeError(
            "Kaiwu license initialization failed. Check whether LICENSE_USER_ID "
            "and LICENSE_SDK_CODE are correct, and whether the machine can reach "
            "the Kaiwu license server."
        ) from exc
