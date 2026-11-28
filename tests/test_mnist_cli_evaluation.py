"""Exercise the MNIST command-line data boundary without downloading images."""
from pathlib import Path
import runpy
import sys
from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def pipeline_main(monkeypatch):
    """Load the example's real entry point with its optional dependencies."""
    for name in ("gif", "imageio", "torchvision", "torchmetrics", "pandas", "seaborn", "sklearn"):
        pytest.importorskip(name)
    example = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
    monkeypatch.syspath_prepend(str(example))
    with patch.dict(sys.modules):
        for name in list(sys.modules):
            if name.split(".", 1)[0] in {"model", "trainer", "downstream", "utils"}:
                del sys.modules[name]
        yield runpy.run_path(str(example / "run_pipeline.py"))["main"]


@pytest.mark.parametrize("option", ["--num-train-samples", "--num-test-samples"])
@pytest.mark.parametrize("value", ["0", "-2"])
def test_invalid_sample_counts_exit_before_loading(pipeline_main, monkeypatch, capsys, option, value):
    loader = Mock(side_effect=AssertionError("The command must reject the argument before loading"))
    monkeypatch.setitem(pipeline_main.__globals__, "loadMNIST", loader)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", option, value])
    with pytest.raises(SystemExit) as error:
        pipeline_main()
    assert error.value.code == 2
    assert f"{option}: must be a positive integer" in capsys.readouterr().err
    loader.assert_not_called()


def test_empty_evaluation_exits_before_building_pipeline(pipeline_main, monkeypatch, capsys):
    train = DataLoader(TensorDataset(torch.ones(2, 4), torch.ones(2, dtype=torch.long)))
    evaluation = DataLoader(TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)))
    monkeypatch.setitem(pipeline_main.__globals__, "loadMNIST", Mock(return_value=(train, evaluation)))
    builder = Mock(side_effect=AssertionError("The command must reject empty data before training"))
    monkeypatch.setitem(pipeline_main.__globals__, "get_full_pipeline", builder)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--num-test-samples", "1"])
    with pytest.raises(SystemExit) as error:
        pipeline_main()
    assert error.value.code == 2
    assert "evaluation data must contain at least one test sample" in capsys.readouterr().err
    builder.assert_not_called()


def test_positive_evaluation_reaches_pipeline_scoring(pipeline_main, monkeypatch, capsys):
    train = DataLoader(TensorDataset(torch.ones(2, 4), torch.tensor([0, 1])))
    evaluation = DataLoader(TensorDataset(torch.zeros(1, 4), torch.tensor([1])))
    loader = Mock(return_value=(train, evaluation))
    pipeline = Mock()
    pipeline.score.return_value = 1.0
    monkeypatch.setitem(pipeline_main.__globals__, "loadMNIST", loader)
    monkeypatch.setitem(pipeline_main.__globals__, "get_full_pipeline", Mock(return_value=pipeline))
    monkeypatch.setattr(sys, "argv", [
        "run_pipeline.py", "--num-train-samples", "2", "--num-test-samples", "1",
    ])

    pipeline_main()

    assert loader.call_args.kwargs["num_evts_train"] == 2
    assert loader.call_args.kwargs["num_evts_test"] == 1
    pipeline.fit.assert_called_once()
    pipeline.score.assert_called_once()
    features, labels = pipeline.score.call_args.args
    assert features.shape == (1, 4)
    assert labels.tolist() == [1]
    assert "Test accuracy: 100.00%" in capsys.readouterr().out
