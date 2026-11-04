"""Regression tests for output-path containment in the qvae_mnist example.

The example layer writes training artifacts (model checkpoints, results
files, text dumps) under operator-supplied directories.  These tests pin
the containment behaviour: normal usage is unchanged, while labels or
paths that would escape the intended output directory are rejected.
"""

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "example/qvae_mnist"


@pytest.fixture()
def model_tuner(monkeypatch):
    """Import trainer.model_tuner (torch/numpy only) with the example dir on sys.path."""
    if str(EXAMPLE_ROOT) not in sys.path:
        monkeypatch.syspath_prepend(str(EXAMPLE_ROOT))
    import trainer.model_tuner as model_tuner

    return model_tuner


def _import_helpers():
    """Import utils.helpers after guarding its plotting/video dependencies."""
    for module in ("gif", "imageio", "matplotlib", "PIL", "sklearn", "torchvision", "torchmetrics"):
        pytest.importorskip(module)
    if str(EXAMPLE_ROOT) not in sys.path:
        sys.path.insert(0, str(EXAMPLE_ROOT))
    return importlib.import_module("utils.helpers")


def test_safe_outfile_keeps_plain_names_inside_outpath(model_tuner, tmp_path):
    target = model_tuner._safe_outfile(str(tmp_path), "model_final_QVAE.pt")
    assert target.parent == tmp_path.resolve()
    assert target.name == "model_final_QVAE.pt"


def test_safe_outfile_rejects_names_that_escape_outpath(model_tuner, tmp_path):
    for name in (
        "../escape.pt",
        "model_../escape.pt",
        "sub/dir/model_x.pt",
        str(Path.home() / "model_x.pt"),
    ):
        with pytest.raises(ValueError):
            model_tuner._safe_outfile(str(tmp_path), name)
    assert not (tmp_path.parent / "escape.pt").exists()
    assert not (tmp_path / "sub").exists()


def test_save_model_writes_inside_outpath_and_refuses_traversal(
    model_tuner, tmp_path
):
    torch = pytest.importorskip("torch")

    stub = type(
        "StubModel",
        (),
        {
            "state_dict": lambda self: {"w": torch.zeros(1)},
            "prior": torch.zeros(1),
        },
    )()
    tuner = model_tuner.ModelTuner()
    tuner.outpath = str(tmp_path)
    tuner._model = stub

    tuner.save_model(config_string="final_QVAE")
    assert (tmp_path / "model_final_QVAE.pt").is_file()
    tuner.save_rbm(config_string="final_QVAE")
    assert (tmp_path / "rbm_final_QVAE.pt").is_file()

    with pytest.raises(ValueError):
        tuner.save_model(config_string="../escape")
    assert not (tmp_path.parent / "model_escape.pt").exists()


def test_save_list_to_txt_writes_within_requested_directory(tmp_path):
    helpers = _import_helpers()

    target = tmp_path / "deep" / "nested" / "values.txt"
    helpers.save_list_to_txt(str(target), [0.123456789, -2.5])
    text = target.read_text()
    assert "0.123457" in text
    assert "-2.500000" in text


def test_save_list_to_txt_stays_inside_requested_directory(tmp_path):
    helpers = _import_helpers()

    helpers.save_list_to_txt(str(tmp_path / "values.txt"), [1.0])
    assert not (tmp_path.parent / "values.txt").exists()


def test_evaluate_qvae_fid_writes_result_inside_save_dir(tmp_path, monkeypatch):
    helpers = _import_helpers()

    monkeypatch.setattr(helpers, "compute_fid_in_batches", lambda *a, **k: 0.42)
    fake_trainer = type(
        "FakeTrainer",
        (),
        {
            "config": type("Config", (), {"num_epochs": 3})(),
            "model": type("Model", (), {"_latent_dimensions": 8})(),
        },
    )()

    score = helpers.evaluate_qvae_fid(
        fake_trainer, None, None, "cpu", save_path=str(tmp_path)
    )
    assert score == 0.42
    assert "FID" in (tmp_path / "fid_results.txt").read_text()


def test_trainer_output_dir_is_resolved_and_created(tmp_path, monkeypatch):
    for module in ("pandas", "seaborn", "gif", "imageio", "matplotlib", "PIL", "sklearn", "torchvision", "torchmetrics"):
        pytest.importorskip(module)
    if str(EXAMPLE_ROOT) not in sys.path:
        monkeypatch.syspath_prepend(str(EXAMPLE_ROOT))
    import trainer as trainer_module
    from model import Config  # noqa: E402  (needs EXAMPLE_ROOT on sys.path)

    monkeypatch.chdir(tmp_path)
    config = Config(model_type="QVAE", output_dir=str(tmp_path / "run_out"))
    instance = trainer_module.Trainer(config)
    assert Path(instance.output_dir) == (tmp_path / "run_out").resolve()
    assert (tmp_path / "run_out").is_dir()
