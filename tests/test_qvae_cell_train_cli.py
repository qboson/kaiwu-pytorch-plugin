"""CLI-defaults contract for the single-cell training script."""

import importlib.util
import os
import sys
import types

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_cell"))
for name in ("scanpy", "anndata"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
for _name in [m for m in list(sys.modules) if m == "models" or m.startswith("models.")]:
    del sys.modules[_name]
import kaiwu.preprocess as _preprocess  # noqa: E402

if not hasattr(_preprocess, "PrecisionReducer"):
    _preprocess.PrecisionReducer = type(
        "PrecisionReducer", (), {"__init__": lambda self, *args, **kwargs: None}
    )

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_train_script",
    os.path.join(REPO_ROOT, "example", "qvae_cell", "train_qvae_cell.py"),
)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


def parse(argv):
    """Runs the script's parser on a controlled argv."""
    original = sys.argv
    sys.argv = ["train_qvae_cell.py", *argv]
    try:
        return script.parse_args()
    finally:
        sys.argv = original


def test_training_defaults():
    """The documented defaults are exposed by the parser."""
    args = parse([])
    assert args.dataset == "immune"
    assert args.output_dir == "./outputs_immune_sa"
    assert args.device == "cuda:0"
    assert args.seed == 42
    assert args.batch_size == 512
    assert args.epochs == 100
    assert args.early_stopping_patience == 10
    assert args.checkpoint_every == 10
    assert args.latent_dim == 256
    assert (args.num_visible, args.num_hidden) == (128, 128)
    assert args.loss_type == "mse"
    assert args.representation == "q"
    assert args.sampler_type == "sa"
    assert (args.sa_alpha, args.sa_size_limit) == (0.5, 10)
    assert args.tmp_dir == "./tmp"


def test_invalid_choice_is_rejected():
    """argparse rejects unsupported enumerations."""
    with pytest.raises(SystemExit):
        parse(["--loss-type", "poisson"])
