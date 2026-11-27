"""Exercise DPLM script and package entrypoints without starting a workflow."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("transformers")
pytest.importorskip("omegaconf")
pytest.importorskip("tqdm")
pytest.importorskip("esm")

REPO_ROOT = Path(
    os.environ.get("DPLM_ENTRYPOINT_REPO", Path(__file__).resolve().parents[1])
).resolve()
DPLM_ROOT = REPO_ROOT / "example" / "qdiffusion" / "dplm"
WORKFLOWS = [
    ("train_workflow.py", "train.py"),
    ("eval_esm2_distances.py", "esm2_eval.py"),
]


def _environment(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(
        HF_HOME=str(tmp_path / "hf"),
        TORCH_HOME=str(tmp_path / "torch"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        MPLCONFIGDIR=str(tmp_path / "mpl"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        KPP_STATS_ENABLED="false",
    )
    return env


@pytest.mark.parametrize("entrypoint, implementation", WORKFLOWS)
@pytest.mark.parametrize("compatibility_entry", [True, False])
def test_script_reaches_its_actual_main_without_running_it(
    tmp_path, entrypoint, implementation, compatibility_entry
):
    """Use genuine imports and stop before any training or model loading."""
    expected_main = DPLM_ROOT / "workflows" / implementation
    script = DPLM_ROOT / entrypoint if compatibility_entry else expected_main
    guard_dir = tmp_path / "startup_guard"
    guard_dir.mkdir()
    marker = "DPLM_ENTRYPOINT_REACHED_EXPECTED_MAIN"
    # A function-call trace runs before main's first statement. Imports and
    # argument handling are untouched; no model/solver implementation is stubbed.
    (guard_dir / "sitecustomize.py").write_text(
        "import sys\n"
        "def _stop_at_main(frame, event, arg):\n"
        f"    if event == 'call' and frame.f_code.co_name == 'main' and frame.f_code.co_filename == {str(expected_main)!r}:\n"
        "        sys.settrace(None)\n"
        f"        print({marker!r})\n"
        "        raise SystemExit(0)\n"
        "    return None\n"
        "sys.settrace(_stop_at_main)\n",
        encoding="utf-8",
    )
    env = _environment(tmp_path)
    env["PYTHONPATH"] = str(guard_dir)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert marker in completed.stdout, completed.stdout + completed.stderr


@pytest.mark.parametrize("entrypoint, implementation", WORKFLOWS)
def test_package_entrypoint_keeps_its_canonical_main(
    tmp_path, entrypoint, implementation
):
    """Package consumers retain the same implementation and source location."""
    module_name = "example.qdiffusion.dplm." + Path(entrypoint).stem
    expected_main = DPLM_ROOT / "workflows" / implementation
    code = (
        "import importlib, inspect; "
        f"entry = importlib.import_module({module_name!r}); "
        "print(entry.main.__module__); "
        "print(inspect.getsourcefile(entry.main))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.splitlines() == [
        "example.qdiffusion.dplm.workflows." + Path(implementation).stem,
        str(expected_main),
    ]
