"""Documented-path contract for the getting-started guide and example index."""

import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOKEN = re.compile(r"`([^`]+)`")


def paths_in(text, base):
    """Returns existing-relevant path tokens declared in backticks."""
    found = []
    for token in TOKEN.findall(text):
        token = token.strip()
        if token.startswith("http") or " " in token:
            continue
        if not (token.endswith(".py") or token.endswith(".ipynb") or token.endswith("/")):
            continue
        if token.endswith("/") and token.count("/") == 1:
            continue
        root = REPO_ROOT if token.startswith("example/") else base
        found.append(os.path.join(root, token))
    return found


def test_example_index_paths_exist():
    """Scripts and notebooks named in ``example/README.md`` exist."""
    example_dir = os.path.join(REPO_ROOT, "example")
    with open(os.path.join(example_dir, "README.md"), encoding="utf-8") as handle:
        text = handle.read()
    candidates = paths_in(text, example_dir)
    assert candidates, "no script/notebook paths found in example/README.md"
    for path in candidates:
        assert os.path.exists(path), path
