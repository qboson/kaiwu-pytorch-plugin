"""Check experiment dispatch without starting MNIST training."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def runner():
    """Load the standalone dispatcher without importing the SDK."""
    path = Path(__file__).resolve().parents[1] / "example/ep_mnist/closeout/run_experiments.py"
    spec = importlib.util.spec_from_file_location("closeout_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_from_another_directory(tmp_path, monkeypatch, capsys):
    """Dry-run resolves paths without creating data or output directories."""
    module = runner()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("started training"))
    assert module.main(["--data-dir", "data", "--output-dir", "new", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["jobs"]) == 18
    assert not (tmp_path / "new").exists()
    assert not (tmp_path / "data").exists()
    for job in payload["jobs"]:
        assert Path(job["command"][3]).is_file()


def test_sweep_is_validation_only(tmp_path):
    """Every historical combination occurs once, with no test evaluation."""
    jobs = runner().job_list(tmp_path, "sweep")
    actual = set()
    for _, options in jobs:
        get = lambda flag: options[options.index(flag) + 1]
        assert "--validation-only" in options
        assert (get("--limit-train"), get("--limit-eval"), get("--epochs")) == ("2000", "1000", "3")
        actual.add((int(get("--beta")), int(get("--sa-iterations")), int(get("--seed"))))
    assert len(jobs) == 15
    assert actual == {(beta, budget, seed) for beta, budget in
                      ((2, 50), (1, 50), (4, 50), (2, 20), (2, 100)) for seed in range(3)}


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    """Reject existing output before launching a child process."""
    module = runner()
    marker = tmp_path / "keep.txt"
    marker.write_text("original", encoding="utf-8")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("started training"))
    with pytest.raises(SystemExit):
        module.main(["--data-dir", str(tmp_path), "--output-dir", str(tmp_path)])
    assert marker.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("codes,expected", [([0, 0], 0), ([3, 0], 1), ([0, 4], 1)])
def test_dispatch_exit_status(tmp_path, monkeypatch, codes, expected):
    """Propagate child and summarizer failures, retaining their exit codes."""
    module = runner()
    pending = iter(codes)
    def fake_run(*_args, **kwargs):
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["MKL_NUM_THREADS"] == "1"
        return SimpleNamespace(returncode=next(pending))
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output = tmp_path / "run"
    assert module.run_jobs(module.job_list(tmp_path, "bp"), output) == expected
    status = json.loads((output / "status.json").read_text())["bp"]
    assert status["returncode"] == codes[0]
    assert status["summary_returncode"] == codes[1]
    assert status["state"] == ("passed" if codes[0] == 0 else "failed")
