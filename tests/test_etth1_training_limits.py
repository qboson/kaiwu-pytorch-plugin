"""Regression tests for the ETTh1 training command's limits."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("flag", ["--epochs", "--max-train-batches"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_training_limits_stop_before_loading_data(
    flag: str,
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = (
        Path(__file__).parents[1] / "example/qdeq/simple/simple_train_example.py"
    )
    checkpoint = tmp_path / "model.pt"
    existing_checkpoint = b"an existing checkpoint"
    checkpoint.write_bytes(existing_checkpoint)

    def reject_setup(*_args: object) -> None:
        pytest.fail("An invalid training limit reached the trainer")

    # The CLI validation precedes any data loading or solver setup.
    # Keep this test independent of optional dependencies of the example.
    example = SimpleNamespace(
        DEFAULT_CHECKPOINT=checkpoint,
        DEFAULT_DATA_PATH=tmp_path / "ETTh1.csv",
        DEQForecaster=reject_setup,
        build_config=reject_setup,
        build_loaders=reject_setup,
        build_sampler=reject_setup,
        evaluate=reject_setup,
    )
    monkeypatch.setitem(sys.modules, "etth1_common", example)
    monkeypatch.setattr(sys, "argv", [str(script), flag, value])

    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")

    assert error.value.code == 2
    assert f"{flag} must be a positive integer" in capsys.readouterr().err
    assert checkpoint.read_bytes() == existing_checkpoint
