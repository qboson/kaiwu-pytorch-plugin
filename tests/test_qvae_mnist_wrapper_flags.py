"""Shell-wrapper/CLI consistency contract for the MNIST pipeline."""

import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLE = os.path.join(REPO_ROOT, "example", "qvae_mnist")
FLAG = re.compile(r"--[a-z][a-z0-9_-]*")


def declared_flags():
    """Returns every ``add_argument`` flag of ``run_pipeline.py``."""
    with open(os.path.join(EXAMPLE, "run_pipeline.py"), encoding="utf-8") as handle:
        return set(re.findall(r'add_argument\(\s*"(--[a-z0-9_-]+)"', handle.read()))


def wrapper_flags():
    """Returns every flag used by ``run_pipeline.sh`` (comments stripped)."""
    flags = set()
    with open(os.path.join(EXAMPLE, "run_pipeline.sh"), encoding="utf-8") as handle:
        for line in handle:
            code = line.split("#", 1)[0]
            flags.update(FLAG.findall(code))
    return flags


def test_wrapper_only_passes_declared_flags():
    """The shell wrapper and the Python parser agree on option names."""
    used = wrapper_flags()
    assert used, "no flags found in run_pipeline.sh"
    assert used <= declared_flags(), sorted(used - declared_flags())
