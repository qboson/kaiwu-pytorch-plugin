"""Deterministic MNIST/100 subset loading.

MNIST/100 is the reduced split used by the Ising EP literature: the first 100
training images and first 10 test images of each digit, taken in the original
IDX file order. Selecting by position rather than at random means the split is
reproducible from the source files alone, with no seed to record.

Pixels are scaled to ``[0, 1]`` and flattened in C order. There is no
augmentation: the solver problem size is fixed by the network, so the input
pipeline stays as simple as possible.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import struct
import urllib.request

import numpy as np

#: Mirror hosting the four IDX archives.
MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"

#: Archive filenames, in the order MNIST publishes them.
MNIST_ARCHIVES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

#: Archive filename to download URL.
MNIST_URLS = {name: f"{MNIST_BASE_URL}/{name}" for name in MNIST_ARCHIVES}

TRAIN_PER_CLASS = 100
TEST_PER_CLASS = 10
N_CLASSES = 10

_IMAGE_MAGIC = 2051
_LABEL_MAGIC = 2049


@dataclass(frozen=True)
class Mnist100Split:
    """One loaded split of the MNIST/100 subset.

    Attributes:
        images: Flattened ``float64`` pixels in ``[0, 1]``, one row per sample.
        labels: Integer class labels aligned with ``images``.
        indices: Positions of these samples in the original IDX file.
    """

    images: np.ndarray
    labels: np.ndarray
    indices: list[int]


def mnist_raw_dir(root: str | Path) -> Path:
    """Return the directory holding the raw IDX archives.

    Args:
        root: Data root, for example ``data``.

    Returns:
        The ``<root>/MNIST/raw`` path.
    """
    return Path(root) / "MNIST" / "raw"


def locate_mnist_files(
    root: str | Path,
    *,
    download_if_missing: bool = False,
) -> dict[str, Path]:
    """Find the four IDX archives, optionally downloading them.

    Downloading is opt-in so a training run never reaches the network
    unexpectedly.

    Args:
        root: Data root containing ``MNIST/raw``.
        download_if_missing: Fetch absent archives from :data:`MNIST_URLS`.

    Returns:
        Mapping of archive filename to its path on disk.

    Raises:
        FileNotFoundError: If archives are missing and downloading is disabled.
    """
    raw = mnist_raw_dir(root)
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for name, url in MNIST_URLS.items():
        destination = raw / name
        if not destination.is_file() or destination.stat().st_size == 0:
            if not download_if_missing:
                missing.append(name)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(destination.suffix + ".part")
            urllib.request.urlretrieve(url, partial)
            partial.replace(destination)
        paths[name] = destination
    if missing:
        raise FileNotFoundError(
            f"missing MNIST archives in {raw}: {missing}. "
            "Pass download_if_missing=True or place the IDX files there."
        )
    return paths


def _read_idx_images(path: Path) -> np.ndarray:
    """Read an IDX image archive as raw ``uint8`` pixels.

    Args:
        path: Gzipped IDX image archive.

    Returns:
        A ``(count, 28, 28)`` ``uint8`` array.

    Raises:
        ValueError: If the header magic, image shape or payload length is wrong,
            which means the file is not the expected MNIST archive.
    """
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != _IMAGE_MAGIC:
            raise ValueError(f"{path.name}: expected image magic {_IMAGE_MAGIC}, got {magic}")
        if (rows, cols) != (28, 28):
            raise ValueError(f"{path.name}: expected 28x28 images, got {rows}x{cols}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    if data.size != count * rows * cols:
        raise ValueError(f"{path.name}: payload does not match the header count")
    return data.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    """Read an IDX label archive.

    Args:
        path: Gzipped IDX label archive.

    Returns:
        A one-dimensional ``int64`` label array.

    Raises:
        ValueError: If the header magic or payload length is wrong.
    """
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != _LABEL_MAGIC:
            raise ValueError(f"{path.name}: expected label magic {_LABEL_MAGIC}, got {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    if data.size != count:
        raise ValueError(f"{path.name}: payload does not match the header count")
    return data.astype(np.int64)


def normalize_pixels(images: np.ndarray) -> np.ndarray:
    """Flatten ``uint8`` images in C order and scale them to ``[0, 1]``.

    Args:
        images: A ``(count, 28, 28)`` ``uint8`` array.

    Returns:
        A ``(count, 784)`` ``float64`` array with values in ``[0, 1]``.
    """
    return np.ascontiguousarray(images.reshape(images.shape[0], -1), dtype=np.float64) / 255.0


def first_per_class_indices(labels: np.ndarray, per_class: int) -> list[int]:
    """Select the first ``per_class`` positions of every class, in file order.

    Args:
        labels: Label array to scan.
        per_class: Samples to take from each class.

    Returns:
        Ascending positions into ``labels``.

    Raises:
        ValueError: If a label is outside ``[0, 10)``, or some class has fewer
            than ``per_class`` samples.
    """
    values = np.asarray(labels, dtype=np.int64)
    if values.size and (values.min() < 0 or values.max() >= N_CLASSES):
        raise ValueError(f"labels must lie in [0, {N_CLASSES})")
    chosen: list[int] = []
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for position, label in enumerate(values.tolist()):
        if counts[label] < per_class:
            chosen.append(position)
            counts[label] += 1
            if int(counts.sum()) == N_CLASSES * per_class:
                break
    if int(counts.min()) < per_class:
        raise ValueError(f"not every class has {per_class} samples available")
    return chosen


def load_mnist100(
    root: str | Path,
    *,
    download_if_missing: bool = False,
    train_per_class: int = TRAIN_PER_CLASS,
    test_per_class: int = TEST_PER_CLASS,
) -> tuple[Mnist100Split, Mnist100Split]:
    """Load the deterministic MNIST/100 train and test splits.

    Args:
        root: Data root containing ``MNIST/raw``.
        download_if_missing: Fetch absent archives from :data:`MNIST_URLS`.
        train_per_class: Training images per class.
        test_per_class: Test images per class.

    Returns:
        The training split and the test split.

    Raises:
        FileNotFoundError: If archives are missing and downloading is disabled.
    """
    paths = locate_mnist_files(root, download_if_missing=download_if_missing)
    splits = []
    for image_key, label_key, per_class in (
        ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", train_per_class),
        ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", test_per_class),
    ):
        labels = _read_idx_labels(paths[label_key])
        indices = first_per_class_indices(labels, per_class)
        images = normalize_pixels(_read_idx_images(paths[image_key])[indices])
        splits.append(Mnist100Split(images=images, labels=labels[indices], indices=indices))
    return splits[0], splits[1]


def load_validation_split(
    root: str | Path,
    validation_per_class: int,
    *,
    download_if_missing: bool = False,
    train_per_class: int = TRAIN_PER_CLASS,
) -> Mnist100Split:
    """Load a validation split disjoint from the MNIST/100 training split.

    The split holds, for each class, the training-file images that follow the
    first ``train_per_class``: positions ``train_per_class`` to
    ``train_per_class + validation_per_class - 1`` of that class, in file order.
    It involves no randomness and never touches the test archive.

    Args:
        root: Data root containing ``MNIST/raw``.
        validation_per_class: Validation images per class.
        download_if_missing: Fetch absent archives from :data:`MNIST_URLS`.
        train_per_class: Training images per class that must be skipped.

    Returns:
        The validation split.

    Raises:
        FileNotFoundError: If archives are missing and downloading is disabled.
    """
    paths = locate_mnist_files(root, download_if_missing=download_if_missing)
    labels = _read_idx_labels(paths["train-labels-idx1-ubyte.gz"])
    skipped = set(first_per_class_indices(labels, train_per_class))
    upto = first_per_class_indices(labels, train_per_class + validation_per_class)
    indices = [position for position in upto if position not in skipped]
    images = normalize_pixels(_read_idx_images(paths["train-images-idx3-ubyte.gz"])[indices])
    return Mnist100Split(images=images, labels=labels[indices], indices=indices)
