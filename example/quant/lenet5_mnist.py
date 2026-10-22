"""LeNet-5 and MNIST helpers used by the QUBO PTQ examples."""

# pylint: disable=too-few-public-methods,too-many-instance-attributes,too-many-locals

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader, TensorDataset


class LeNet5(nn.Module):
    """Define the LeNet-5 model used by the original notebooks.

    Args:
        num_classes (int, optional): Number of output classes.
    """

    def __init__(self, num_classes: int = 10):
        """Create LeNet-5 layers.

        Args:
            num_classes (int, optional): Number of output classes.
        """

        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.subsampel1 = nn.MaxPool2d(2, 2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )
        self.subsampel2 = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(12, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run LeNet-5 inference.

        Args:
            input_tensor (torch.Tensor): Input image batch.

        Returns:
            torch.Tensor: Class logits.
        """

        feature_tensor = self.subsampel1(self.layer1(input_tensor))
        feature_tensor = self.subsampel2(self.layer2(feature_tensor))
        feature_tensor = self.gap(feature_tensor).view(feature_tensor.size(0), -1)
        return self.fc2(self.relu(self.fc1(feature_tensor)))


def read_mnist(root: Path, train: bool, limit: int | None = None) -> TensorDataset:
    """Load local MNIST IDX files and apply notebook preprocessing.

    Args:
        root (Path): Dataset root containing ``MNIST/raw``.
        train (bool): Whether to load the train split.
        limit (int | None, optional): Maximum number of samples.

    Returns:
        TensorDataset: Preprocessed images and labels.

    Raises:
        ValueError: If the IDX magic number is invalid.
    """

    raw_directory = root / "MNIST" / "raw"
    prefix = "train" if train else "t10k"
    image_path = _idx_path(raw_directory / f"{prefix}-images-idx3-ubyte")
    label_path = _idx_path(raw_directory / f"{prefix}-labels-idx1-ubyte")

    image_bytes = _read_bytes(image_path)
    magic, count, rows, columns = struct.unpack(">IIII", image_bytes[:16])
    if magic != 2051:
        raise ValueError(f"invalid MNIST image file: {image_path}")
    count = min(count, limit or count)
    images = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
        offset=16,
        count=count * rows * columns,
    ).copy()
    images = torch.from_numpy(images).float().view(count, 1, rows, columns) / 255.0
    images = torch_functional.interpolate(
        images,
        size=(32, 32),
        mode="bilinear",
        align_corners=False,
    )
    image_mean, image_std = (0.1307, 0.3081) if train else (0.1325, 0.3105)
    images = (images - image_mean) / image_std

    label_bytes = _read_bytes(label_path)
    magic, label_count = struct.unpack(">II", label_bytes[:8])
    if magic != 2049:
        raise ValueError(f"invalid MNIST label file: {label_path}")
    labels = torch.from_numpy(
        np.frombuffer(
            label_bytes,
            dtype=np.uint8,
            offset=8,
            count=min(label_count, count),
        ).copy()
    ).long()
    return TensorDataset(images, labels)


def make_loader(
    root: Path,
    train: bool,
    batch_size: int,
    limit: int | None = None,
) -> DataLoader:
    """Create a DataLoader for local MNIST data.

    Args:
        root (Path): Dataset root containing ``MNIST/raw``.
        train (bool): Whether to load the train split.
        batch_size (int): Batch size.
        limit (int | None, optional): Maximum number of samples.

    Returns:
        DataLoader: MNIST data loader.
    """

    return DataLoader(
        read_mnist(root, train=train, limit=limit),
        batch_size=batch_size,
        shuffle=False,
    )


def load_model(path: Path) -> nn.Module:
    """Load the trained LeNet-5 checkpoint.

    Args:
        path (Path): Checkpoint path.

    Returns:
        nn.Module: Evaluation-mode LeNet-5 model.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        return checkpoint.eval()
    model = LeNet5().eval()
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    return model


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """Evaluate average cross-entropy loss and accuracy.

    Args:
        model (nn.Module): Model to evaluate.
        loader (DataLoader): Evaluation data loader.

    Returns:
        tuple[float, float]: Average loss and accuracy.
    """

    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            loss_sum += torch_functional.cross_entropy(
                logits,
                labels,
                reduction="sum",
            ).item()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return loss_sum / total, correct / total


def _idx_path(path: Path) -> Path:
    """Return plain or gzip-compressed IDX path.

    Args:
        path (Path): Plain IDX path.

    Returns:
        Path: Existing plain or gzip IDX path.
    """

    return path if path.exists() else path.with_suffix(path.suffix + ".gz")


def _read_bytes(path: Path) -> bytes:
    """Read plain or gzip-compressed bytes.

    Args:
        path (Path): File path.

    Returns:
        bytes: File contents.
    """

    with (
        gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
    ) as file_handle:
        return file_handle.read()
