"""The t-SNE animation helper accepts PNG frames with different color modes."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(name="helpers")
def fixture_helpers():
    """Import the real helper and installed image encoder dependencies."""
    for dependency in ("matplotlib", "pandas", "sklearn", "torchvision",
                       "torchmetrics", "gif", "imageio"):
        pytest.importorskip(dependency)
    path = Path(__file__).resolve().parents[1] / "example/qvae_mnist/utils/helpers.py"
    spec = importlib.util.spec_from_file_location("mnist_animation_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("modes", [("RGB", "RGBA", "L"), ("RGBA", "RGBA", "RGBA")])
def test_animation_encodes_mixed_color_modes_and_sizes(helpers, tmp_path, modes):
    """Inspect actual GIF frames, their colors, size, count and duration."""
    paths = []
    for index, mode in enumerate(modes):
        path = tmp_path / f"frame_{index}.png"
        color = (255, 0, 0) if index == 0 else (0, 255, 0) if index == 1 else (80, 80, 80)
        if mode == "RGBA":
            color = (*color, 255)
        elif mode == "L":
            color = 80
        with Image.new(mode, (16 + index * 4, 12 + index * 3), color) as frame:
            frame.save(path)
        paths.append(str(path))

    helpers.create_tsne_animation(paths, str(tmp_path), duration=200)

    with Image.open(tmp_path / "qvae_training_evolution.gif") as animation:
        assert animation.format == "GIF" and animation.n_frames == 3
        assert animation.size == (16, 12)
        for index, color in enumerate(((255, 0, 0), (0, 255, 0), (80, 80, 80))):
            animation.seek(index)
            assert animation.info["duration"] == 200
            np.testing.assert_array_equal(np.asarray(animation.convert("RGB"))[6, 8], color)


def test_animation_without_existing_frames_does_not_create_output(helpers, tmp_path):
    """The existing no-frame behavior remains a harmless no-op."""
    helpers.create_tsne_animation([str(tmp_path / "missing.png")], str(tmp_path))
    assert not (tmp_path / "qvae_training_evolution.gif").exists()
