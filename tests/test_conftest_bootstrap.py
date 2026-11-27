"""Tests for the import bootstrap in ``tests/conftest.py``.

Every case runs in a fresh interpreter: the bootstrap mutates ``kaiwu.__path__`` and
``sys.modules`` at import time, so asserting on it in-process would either depend on
whatever state the parent session happens to be in or corrupt it for the rest of the
run.  The subprocess imports ``tests/conftest.py`` by path and reports what the
import resolved to; the assertions stay here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).resolve().with_name("conftest.py")
SRC_PLUGIN = Path(__file__).resolve().parents[1] / "src" / "kaiwu" / "torch_plugin" / "__init__.py"
OPT_OUT_ENV = "KAIWU_TESTS_KEEP_INSTALLED"

_PROBE = r"""
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

conftest_path, case = Path(sys.argv[1]), sys.argv[2]
spec = importlib.util.spec_from_file_location("kaiwu_tests_conftest", conftest_path)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)  # the module level call applies it once

src_plugin = conftest_path.parents[1] / "src" / "kaiwu" / "torch_plugin" / "__init__.py"


def resolved():
    # Report where ``kaiwu.torch_plugin`` is imported from right now.
    module = importlib.import_module("kaiwu.torch_plugin")
    path = module.__file__
    try:
        is_src = os.path.samefile(path, src_plugin)
    except OSError:
        is_src = False
    return {"path": path, "is_src": is_src}


import kaiwu

report = {"first": resolved()}
report["duplicates"] = len(kaiwu.__path__) - len(
    {os.path.normcase(os.path.realpath(entry)) for entry in kaiwu.__path__}
)
report["second_return"] = bootstrap._prefer_working_tree()
report["second"] = resolved()

if case == "optout":
    del os.environ["KAIWU_TESTS_KEEP_INSTALLED"]
    report["reverted_return"] = bootstrap._prefer_working_tree()
    report["reverted"] = resolved()
else:
    # The bootstrap must leave the eagerly imported siblings of kaiwu.torch_plugin
    # alone: the installed wheel does not rebuild them once dropped.
    for name in ("classical", "core"):
        importlib.import_module("kaiwu." + name)
        report[name + "_attribute_kept"] = hasattr(kaiwu, name)

sys.stdout.write("<<<" + json.dumps(report) + ">>>")
"""


_REPORTS: dict = {}


def _run(case: str) -> dict:
    """Run the probe in a subprocess with a clean CI/opt-out environment.

    Each case costs a fresh interpreter plus a ``torch`` import, so the report is
    cached: several assertions read different fields of the same run.
    """
    if case in _REPORTS:
        return _REPORTS[case]
    if not SRC_PLUGIN.is_file():
        pytest.skip("this checkout has no src/kaiwu/torch_plugin to import")
    env = {key: value for key, value in os.environ.items() if key not in ("CI", OPT_OUT_ENV)}
    if case == "ci":
        env["CI"] = "true"
    elif case == "optout":
        env[OPT_OUT_ENV] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, str(CONFTEST), case],
        capture_output=True,
        text=True,
        cwd=str(CONFTEST.parents[1]),
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = completed.stdout.split("<<<")[-1].split(">>>")[0]
    _REPORTS[case] = json.loads(payload)
    return _REPORTS[case]


def test_local_run_prefers_the_working_tree() -> None:
    """Outside CI the checkout wins, and the helper is a no-op on the second call."""
    report = _run("default")
    assert report["first"]["is_src"] is True, report["first"]
    assert report["second_return"] is False
    assert report["second"]["is_src"] is True
    assert report["duplicates"] == 0, "the checkout entry must not be appended twice"


def test_ci_keeps_validating_the_installed_artifact() -> None:
    """CI runs the suite against what ``pip install .`` produced, not the checkout."""
    report = _run("ci")
    assert report["first"]["is_src"] is False, report["first"]
    assert report["second_return"] is False
    assert report["second"]["is_src"] is False


def test_purge_leaves_installed_siblings_alone() -> None:
    """Only ``kaiwu.torch_plugin`` is dropped; sibling subpackages keep working."""
    report = _run("default")
    assert report["classical_attribute_kept"] is True
    assert report["core_attribute_kept"] is True


def test_opt_out_is_respected_and_not_sticky() -> None:
    """``KAIWU_TESTS_KEEP_INSTALLED`` selects the installed copy, one process at a time."""
    report = _run("optout")
    assert report["first"]["is_src"] is False, report["first"]
    assert report["second_return"] is False
    assert report["reverted_return"] is True
    assert report["reverted"]["is_src"] is True
