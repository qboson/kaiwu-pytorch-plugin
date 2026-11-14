"""JSON and NPZ writing for experiment runs, plus optional Kaiwu credentials.

JSON uses sorted keys and no insignificant whitespace, so two runs of the same
configuration produce byte-identical files.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string with second precision.

    Returns:
        A timestamp such as ``2026-09-21T12:20:17Z``.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as compact JSON with sorted keys.

    Args:
        path: Destination file. Parent directories are created as needed.
        payload: A structure of dicts, lists and JSON primitives.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    """Write named parameter tensors to an ``.npz`` file.

    Args:
        path: Destination file, including the ``.npz`` suffix. Parent
            directories are created as needed.
        arrays: Mapping of array name to a Torch tensor.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric = {
        key: np.ascontiguousarray(value.detach().cpu().numpy()) for key, value in arrays.items()
    }
    np.savez(path, **numeric)


def load_dotenv(path: Path | None = None) -> tuple[str, str] | None:
    """Read Kaiwu credentials from the environment, falling back to ``.env``.

    Existing environment variables win, so a shell export overrides the file.
    Credentials are returned to the caller and never written to an artifact.

    Without an explicit ``path`` the project root is searched first, then its
    parent directory (the workspace root); the first ``.env`` found is used.

    Args:
        path: Explicit ``.env`` location. Skips the default search.

    Returns:
        ``(user_id, sdk_code)`` when both are present, otherwise ``None``.
    """
    if path:
        candidates = [Path(path)]
    else:
        project = Path(__file__).resolve().parents[2]
        candidates = [project / ".env", project.parent / ".env"]
    env_path = next((item for item in candidates if item.is_file()), None)
    if env_path is not None:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    user_id = os.environ.get("KAIWU_USER_ID", "").strip()
    sdk_code = os.environ.get("KAIWU_SDK_CODE", "").strip()
    return (user_id, sdk_code) if user_id and sdk_code else None
