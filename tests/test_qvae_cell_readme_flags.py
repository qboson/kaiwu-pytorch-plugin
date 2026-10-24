"""README/CLI consistency contract for the single-cell example."""

import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLE = os.path.join(REPO_ROOT, "example", "qvae_cell")
FLAG = re.compile(r"--[a-z][a-z0-9-]*")


def cli_flags(filename):
    """Returns every ``add_argument`` flag declared by one script."""
    with open(os.path.join(EXAMPLE, filename), encoding="utf-8") as handle:
        return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', handle.read()))


def readme_flags():
    """Returns every flag mentioned inside README code fences."""
    flags = set()
    with open(os.path.join(EXAMPLE, "README.md"), encoding="utf-8") as handle:
        in_fence = False
        for line in handle:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                flags.update(FLAG.findall(line))
    return flags


def test_readme_examples_only_use_declared_flags():
    """Every flag shown in the README is accepted by one of the two CLIs."""
    declared = cli_flags("train_qvae_cell.py") | cli_flags("evaluate_benchmark.py")
    mentioned = readme_flags()
    assert mentioned, "no flags found in the README code fences"
    assert mentioned <= declared, sorted(mentioned - declared)
