"""t-SNE animation helper contract for the QVAE-MNIST example."""

import importlib.util
import os
import sys

from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

_spec = importlib.util.spec_from_file_location(
    "qvae_mnist_helpers_animation",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "helpers.py"),
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def make_frames(tmp_path):
    """Writes two differently sized PNG frames."""
    frames = []
    for index, size in enumerate((10, 20), start=1):
        path = tmp_path / f"frame{index}.png"
        Image.new("RGB", (size, size), (index * 100, 0, 0)).save(path)
        frames.append(str(path))
    return frames


def test_animation_is_written_with_uniform_frame_size(tmp_path):
    """All frames are resized to the first frame and encoded as one GIF."""
    frames = make_frames(tmp_path)
    helpers.create_tsne_animation(frames, str(tmp_path), duration=50)
    output = tmp_path / "qvae_training_evolution.gif"
    assert output.is_file()
    with Image.open(output) as animation:
        assert animation.size == (10, 10)
        assert animation.n_frames == 2


def test_empty_frame_list_is_a_no_op(tmp_path):
    """Without frames no animation file is created."""
    helpers.create_tsne_animation([], str(tmp_path))
    assert not (tmp_path / "qvae_training_evolution.gif").exists()
