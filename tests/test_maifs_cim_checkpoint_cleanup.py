"""Exercise CIM checkpoint ownership offline through public MAIFS calls."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from torch import nn

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
import kaiwu

kaiwu.__path__.insert(0, str(SOURCE / "kaiwu"))
from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper
from kaiwu.torch_plugin.maifs import qubo

assert Path(qubo.__file__).resolve() == (
    SOURCE / "kaiwu" / "torch_plugin" / "maifs" / "qubo.py"
).resolve()


@pytest.fixture
def offline_cim(monkeypatch, tmp_path):
    """Stub hardware and precision splitting; guard every recursive test deletion."""
    state = {"jobs": [], "failure": None, "one_dimensional": False}
    real_rmtree = qubo.shutil.rmtree
    disposable_root = tmp_path.resolve()

    def guarded_rmtree(path, *args, **kwargs):
        target = Path(path).resolve()
        assert target != disposable_root
        assert target.is_relative_to(disposable_root)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(qubo.shutil, "rmtree", guarded_rmtree)
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(qubo.tempfile, "gettempdir", lambda: str(fallback))
    # Job-directory uniqueness must not depend on this task-name timestamp.
    monkeypatch.setattr(qubo, "time_ns", lambda: 123)

    class OfflineExplorer:
        """Leave splitting out of a checkpoint-ownership regression."""

        def __init__(self, **kwargs):
            pass

        def search(self, matrix):
            return SimpleNamespace(split_matrix=matrix)

        def restore_solution(self, solution):
            if state["failure"] == "restore":
                raise RuntimeError("offline restoration failed")
            return solution

    class OfflineCIM:
        """Write representative SDK records into the configured checkpoint directory."""

        def __init__(self, **kwargs):
            checkpoint_dir = Path(kaiwu.common.CheckpointManager.save_dir).resolve()
            assert checkpoint_dir.is_relative_to(disposable_root)
            record_dir = checkpoint_dir / kwargs["task_name"]
            record_dir.mkdir(parents=True, exist_ok=True)
            record = record_dir / "result.bin"
            record.write_bytes(b"current CIM result")
            (checkpoint_dir / "job-metadata.txt").write_bytes(b"current CIM metadata")
            state["jobs"].append({"directory": checkpoint_dir, "record": record})
            if state["failure"] == "construct":
                raise RuntimeError("offline constructor failed")

        def solve(self, matrix):
            if state["failure"] == "solve":
                raise RuntimeError("offline solve failed")
            if state["failure"] == "none":
                return None
            if state["failure"] == "empty":
                return np.empty((0, 3))
            if state["one_dimensional"]:
                return np.array([-1, 1, -1])  # Negative gauge decodes to [1, 0].
            return np.array([[1, -1, 1], [-1, 1, 1]])

    monkeypatch.setattr(qubo, "PrecisionSplitExplorer", OfflineExplorer)
    monkeypatch.setattr(kaiwu.cim, "CIMOptimizer", OfflineCIM)
    return state


def prepare_directories(tmp_path, monkeypatch, mode):
    """Seed only new disposable directories with unrelated user and prior-job records."""
    existing_sdk_dir = tmp_path / "sdk-existing"
    existing_sdk_dir.mkdir()
    (existing_sdk_dir / "sdk-user.txt").write_bytes(b"SDK user's independent record")
    previous_setting = None if mode == "default" else str(existing_sdk_dir)
    monkeypatch.setattr(kaiwu.common.CheckpointManager, "save_dir", previous_setting)
    if mode == "explicit":
        checkpoint_base = tmp_path / "requested"
    elif mode == "existing":
        checkpoint_base = existing_sdk_dir
    else:
        checkpoint_base = tmp_path / "fallback" / "feature_selection_kaiwu_cim"
    checkpoint_base.mkdir(parents=True, exist_ok=True)
    user_file = checkpoint_base / "user-notes.txt"
    user_file.write_bytes(b"Unrelated user notes")
    prior_job = checkpoint_base / "prior-job"
    prior_job.mkdir()
    prior_record = prior_job / "checkpoint.bin"
    prior_record.write_bytes(b"Unrelated earlier job")
    options = {"save_dir": checkpoint_base} if mode == "explicit" else {}
    return checkpoint_base, previous_setting, user_file, prior_record, options


def public_solve(options):
    """Use the public solver and its real conversion and auxiliary-spin decoding."""
    return qubo.solve_qubo(
        np.array([[1.0, 0.25], [0.25, 2.0]]),
        np.array([-2.0, 0.5]),
        np.ones(2, dtype=int),
        solver="kaiwu_cim",
        **options,
    )


def wrapper_update(options):
    """Use real model derivatives, QUBO construction, public solving and mask storage."""
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.8, -0.2]]))
    selector = FeatureSelectionWrapper(
        model,
        feature_dim=2,
        min_selected_features=0,
        solver="kaiwu_cim",
        solver_kwargs=options,
    )
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = torch.tensor([[1.0], [0.0], [1.0]])
    selected = selector.update_mask([(inputs, targets)], nn.MSELoss())
    assert selector.selected_indices().tolist() == [0]
    assert selector.get_support().tolist() == [True, False]
    return selected


@pytest.mark.parametrize("caller", ["solver", "wrapper"])
@pytest.mark.parametrize("cleanup", [True, False])
@pytest.mark.parametrize("mode", ["explicit", "existing", "default"])
def test_success_owns_only_its_job_records(
    tmp_path, monkeypatch, offline_cim, caller, cleanup, mode
):
    """Public solver and wrapper preserve all unrelated records for every base choice."""
    checkpoint_base, previous, user_file, prior_record, options = prepare_directories(
        tmp_path, monkeypatch, mode
    )
    options["cleanup_records"] = cleanup
    offline_cim["one_dimensional"] = caller == "wrapper"
    selected = public_solve(options) if caller == "solver" else wrapper_update(options)

    assert selected.tolist() == [1, 0]
    assert user_file.read_bytes() == b"Unrelated user notes"
    assert prior_record.read_bytes() == b"Unrelated earlier job"
    assert (tmp_path / "sdk-existing" / "sdk-user.txt").read_bytes() == b"SDK user's independent record"
    assert kaiwu.common.CheckpointManager.save_dir == previous
    job = offline_cim["jobs"][0]
    assert job["directory"].parent == checkpoint_base.resolve()
    assert job["directory"] != checkpoint_base.resolve()
    if cleanup:
        assert not job["directory"].exists()
    else:
        assert job["record"].read_bytes() == b"current CIM result"
        assert (job["directory"] / "job-metadata.txt").read_bytes() == b"current CIM metadata"


@pytest.mark.parametrize("cleanup", [True, False])
@pytest.mark.parametrize("failure", ["construct", "solve", "none", "empty", "restore"])
def test_failed_jobs_restore_sdk_setting_and_keep_diagnostic_records(
    tmp_path, monkeypatch, offline_cim, cleanup, failure
):
    """Failures preserve their owned records for inspection regardless of cleanup mode."""
    checkpoint_base, previous, user_file, prior_record, options = prepare_directories(
        tmp_path, monkeypatch, "explicit"
    )
    options["cleanup_records"] = cleanup
    offline_cim["failure"] = failure
    with pytest.raises(RuntimeError):
        public_solve(options)

    assert kaiwu.common.CheckpointManager.save_dir == previous
    assert user_file.read_bytes() == b"Unrelated user notes"
    assert prior_record.read_bytes() == b"Unrelated earlier job"
    job = offline_cim["jobs"][0]
    assert job["directory"].parent == checkpoint_base.resolve()
    assert job["record"].read_bytes() == b"current CIM result"


@pytest.mark.parametrize("second_cleanup", [True, False])
def test_consecutive_jobs_do_not_remove_each_others_records(
    tmp_path, monkeypatch, offline_cim, second_cleanup
):
    """A later job cannot remove a retained earlier job, even with identical task names."""
    checkpoint_base, previous, user_file, prior_record, options = prepare_directories(
        tmp_path, monkeypatch, "explicit"
    )
    first = public_solve({**options, "cleanup_records": False})
    first_job = offline_cim["jobs"][0]
    assert kaiwu.common.CheckpointManager.save_dir == previous
    second = public_solve({**options, "cleanup_records": second_cleanup})
    second_job = offline_cim["jobs"][1]

    assert first.tolist() == second.tolist() == [1, 0]
    assert first_job["directory"] != second_job["directory"]
    assert first_job["directory"].parent == second_job["directory"].parent == checkpoint_base.resolve()
    assert first_job["record"].read_bytes() == b"current CIM result"
    assert second_job["directory"].exists() == (not second_cleanup)
    assert user_file.read_bytes() == b"Unrelated user notes"
    assert prior_record.read_bytes() == b"Unrelated earlier job"
    assert kaiwu.common.CheckpointManager.save_dir == previous
