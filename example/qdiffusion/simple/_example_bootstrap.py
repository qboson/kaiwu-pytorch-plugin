"""Development-time path bootstrap for simple qdiffusion examples."""

from __future__ import annotations

from pathlib import Path
import sys


# Path bootstrap helper.

def ensure_repo_src_on_path() -> None:
    """Adds local source paths needed by the simple examples.

    The simple examples depend on both the repository ``src`` tree and the
    shared ``example/qdiffusion`` example package root.
    """
    example_root = Path(__file__).resolve().parents[1]
    repo_root = example_root.parents[1]
    src_path = repo_root / "src"
    example_pkg_path = example_root

    for path in (src_path, example_pkg_path):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
