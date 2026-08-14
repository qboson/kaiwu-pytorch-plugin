"""LeNet-5 Hessian-method QUBO PTQ example on Kaiwu CIM."""

# pylint: disable=wrong-import-position,too-few-public-methods,duplicate-code
# pylint: disable=too-many-instance-attributes,too-many-locals

from __future__ import annotations

import gzip
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kaiwu.torch_plugin import quantization


class LeNet5(nn.Module):
    """LeNet-5 model used by the original notebooks."""

    def __init__(self, num_classes: int = 10):
        """Create the LeNet-5 layers.

        Args:
            num_classes (int, optional): Number of output classes.
        """

        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 6, 3, padding=1), nn.BatchNorm2d(6), nn.ReLU())
        self.subsampel1 = nn.MaxPool2d(2, 2)
        self.layer2 = nn.Sequential(nn.Conv2d(6, 12, 3, padding=1), nn.BatchNorm2d(12), nn.ReLU())
        self.subsampel2 = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(12, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run LeNET-5 inference.

        Args:
            input_tensor (torch.Tensor): Input image batch.

        Returns:
            torch.Tensor: Class logits.
        """

        feature_tensor = self.subsampel1(self.layer1(input_tensor))
        feature_tensor = self.subsampel2(self.layer2(feature_tensor))
        feature_tensor = self.gap(feature_tensor).view(feature_tensor.size(0), -1)
        return self.fc2(self.relu(self.fc1(feature_tensor)))


def read_mnist(root: Path, train: bool) -> TensorDataset:
    """Load MNIST raw IDX files and apply the notebook preprocessing.

    Args:
        root (Path): Dataset root containing ``MNIST/raw``.
        train (bool): Whether to load the train split.

    Returns:
        TensorDataset: Images and labels.

    Raises:
        ValueError: If the IDX magic number is invalid.
    """

    raw_directory = root / "MNIST" / "raw"
    prefix = "train" if train else "t10k"
    image_path = raw_directory / f"{prefix}-images-idx3-ubyte"
    label_path = raw_directory / f"{prefix}-labels-idx1-ubyte"
    if not image_path.exists():
        image_path = image_path.with_suffix(image_path.suffix + ".gz")
    if not label_path.exists():
        label_path = label_path.with_suffix(label_path.suffix + ".gz")

    with (
        gzip.open(image_path, "rb") if image_path.suffix == ".gz" else image_path.open("rb")
    ) as file_handle:
        image_bytes = file_handle.read()
    magic, count, rows, cols = struct.unpack(">IIII", image_bytes[:16])
    if magic != 2051:
        raise ValueError(f"invalid MNIST image file: {image_path}")
    images = np.frombuffer(image_bytes, dtype=np.uint8, offset=16).copy()
    images = torch.from_numpy(images).float().view(count, 1, rows, cols) / 255.0
    images = torch_functional.interpolate(
        images,
        size=(32, 32),
        mode="bilinear",
        align_corners=False,
    )
    image_mean, image_std = (0.1307, 0.3081) if train else (0.1325, 0.3105)
    images = (images - image_mean) / image_std

    with (
        gzip.open(label_path, "rb") if label_path.suffix == ".gz" else label_path.open("rb")
    ) as file_handle:
        label_bytes = file_handle.read()
    magic, label_count = struct.unpack(">II", label_bytes[:8])
    if magic != 2049:
        raise ValueError(f"invalid MNIST label file: {label_path}")
    labels = torch.from_numpy(
        np.frombuffer(label_bytes, dtype=np.uint8, offset=8, count=label_count).copy()
    ).long()
    return TensorDataset(images, labels)


def load_model(path: Path) -> nn.Module:
    """Load the trained LeNet-5 checkpoint used by the notebooks.

    Args:
        path (Path): Checkpoint path.

    Returns:
        nn.Module: Evaluation-mode LeNet-5 model.
    """

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, nn.Module):
        return loaded.eval()
    loaded_model = LeNet5().eval()
    state_dict = (
        loaded["state_dict"]
        if isinstance(loaded, dict) and "state_dict" in loaded
        else loaded
    )
    loaded_model.load_state_dict(state_dict)
    return loaded_model


def evaluate(eval_model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """Return average cross-entropy loss and accuracy.

    Args:
        eval_model (nn.Module): Model to evaluate.
        loader (DataLoader): Evaluation data loader.

    Returns:
        tuple[float, float]: Average loss and accuracy.
    """

    eval_model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            logits = eval_model(images)
            loss_sum += torch_functional.cross_entropy(logits, labels, reduction="sum").item()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return loss_sum / total, correct / total


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MODEL_PATH = ROOT / "base_model.pth"
HESSIAN_PATH = ROOT / "base_hessian_matrix_module_layerwise_params.pt"
CIM_SAVE_DIR = ROOT / "kaiwu_cim_results"
KAIWU_PROJECT_NAME = "RDSA260204015"
KAIWU_PROJECT_NO = "26071422"
BITS = 4
LAYERS = None
BATCH_SIZE = 64
TASK_TAG = time.strftime("%Y%m%d_%H%M%S")
SOLVERS = [{
    "name": "cim",
    "params": {
        "project_no": KAIWU_PROJECT_NO,
        "task_name": f"qubo_ptq_lenet5_hessian_{TASK_TAG}",
        "task_mode": "sample",
        "sample_number": 100,
        "wait": True,
        "interval": 1,
        "bit_width": 14,
        "save_dir": str(CIM_SAVE_DIR),
    },
}]

test_loader = DataLoader(read_mnist(DATA_ROOT, train=False), batch_size=BATCH_SIZE, shuffle=False)
model = load_model(MODEL_PATH)
hessians = torch.load(HESSIAN_PATH, map_location="cpu", weights_only=False)

before_loss, before_accuracy = evaluate(model, test_loader)
print(f"before quantization: loss={before_loss:.6f}, accuracy={before_accuracy:.4%}")

quantized_model, report = quantization(
    model,
    bits=BITS,
    solvers=SOLVERS,
    layers=LAYERS,
).hessian(hessians)

after_loss, after_accuracy = evaluate(quantized_model, test_loader)
print(f"after quantization:  loss={after_loss:.6f}, accuracy={after_accuracy:.4%}")
print(
    f"method={report['method']}, "
    f"bits={report['bits']}, "
    f"layers={[layer['name'] for layer in report['layers']]}"
)
