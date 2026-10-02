"""Regression tests for the dplm workflow compatibility entrypoints.

Running ``train_workflow.py`` / ``eval_esm2_distances.py`` as scripts makes the
package-relative imports fail, so the workflow modules fall back to importing
their siblings by bare module name.  Those sibling modules live in the
``workflows`` directory, which is not on ``sys.path`` when the entrypoint is
executed directly, so the fallback must add its own directory before importing.
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPLM_DIR = os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm")

ENTRYPOINTS = [
    os.path.join(DPLM_DIR, "train_workflow.py"),
    os.path.join(DPLM_DIR, "eval_esm2_distances.py"),
]


def _run_entrypoint(script):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_dplm_entrypoints_resolve_workflow_siblings():
    for script in ENTRYPOINTS:
        result = _run_entrypoint(script)
        combined = result.stdout + result.stderr
        assert "ModuleNotFoundError" not in combined, (
            f"{os.path.basename(script)} failed to import its workflow helpers:\n"
            f"{combined}"
        )
        assert (
            "No module named 'workflow_helpers'" not in combined
        ), f"{os.path.basename(script)} cannot resolve workflow_helpers"
        assert (
            "No module named 'esm2_eval_helpers'" not in combined
        ), f"{os.path.basename(script)} cannot resolve esm2_eval_helpers"
