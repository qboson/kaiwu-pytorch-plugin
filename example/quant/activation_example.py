"""LeNet-5 activation-method QUBO PTQ example on Kaiwu CIM."""

from __future__ import annotations

import gzip
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kaiwu.torch_plugin import quantization


class LeNet5(nn.Module):
    """LeNet-5 model used by the original notebooks."""

    def __init__(self, num_classes: int = 10):
        """Create the LeNet-5 layers."""

        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 6, 3, padding=1), nn.BatchNorm2d(6), nn.ReLU())
        self.subsampel1 = nn.MaxPool2d(2, 2)
        self.layer2 = nn.Sequential(nn.Conv2d(6, 12, 3, padding=1), nn.BatchNorm2d(12), nn.ReLU())
        self.subsampel2 = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(12, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run LeNet-5 inference."""

        x = self.subsampel1(self.layer1(x))
        x = self.subsampel2(self.layer2(x))
        x = self.gap(x).view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))


def read_mnist(root: Path, train: bool, limit: int | None = None) -> TensorDataset:
    """Load MNIST raw IDX files and apply the notebook preprocessing."""

    raw = root / "MNIST" / "raw"
    prefix = "train" if train else "t10k"
    image_path = raw / f"{prefix}-images-idx3-ubyte"
    label_path = raw / f"{prefix}-labels-idx1-ubyte"
    if not image_path.exists():
        image_path = image_path.with_suffix(image_path.suffix + ".gz")
    if not label_path.exists():
        label_path = label_path.with_suffix(label_path.suffix + ".gz")

    with (gzip.open(image_path, "rb") if image_path.suffix == ".gz" else image_path.open("rb")) as f:
        data = f.read()
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"invalid MNIST image file: {image_path}")
    count = min(count, limit or count)
    images = np.frombuffer(data, dtype=np.uint8, offset=16, count=count * rows * cols).copy()
    images = torch.from_numpy(images).float().view(count, 1, rows, cols) / 255.0
    images = F.interpolate(images, size=(32, 32), mode="bilinear", align_corners=False)
    mean, std = (0.1307, 0.3081) if train else (0.1325, 0.3105)
    images = (images - mean) / std

    with (gzip.open(label_path, "rb") if label_path.suffix == ".gz" else label_path.open("rb")) as f:
        data = f.read()
    magic, label_count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"invalid MNIST label file: {label_path}")
    labels = torch.from_numpy(np.frombuffer(data, dtype=np.uint8, offset=8, count=min(label_count, count)).copy()).long()
    return TensorDataset(images, labels)


def load_model(path: Path) -> nn.Module:
    """Load the trained LeNet-5 checkpoint used by the notebooks."""

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, nn.Module):
        return loaded.eval()
    model = LeNet5().eval()
    state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
    model.load_state_dict(state_dict)
    return model


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """Return average cross-entropy loss and accuracy."""

    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return loss_sum / total, correct / total


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MODEL_PATH = ROOT / "base_model.pth"
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
        "task_name": f"qubo_ptq_lenet5_activation_{TASK_TAG}",
        "task_mode": "sample",
        "sample_number": 100,
        "wait": True,
        "interval": 1,
        "bit_width": 14,
        "save_dir": str(CIM_SAVE_DIR),
    },
}]

torch.manual_seed(123)
calibration_loader = DataLoader(read_mnist(DATA_ROOT, train=True, limit=BATCH_SIZE), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(read_mnist(DATA_ROOT, train=False), batch_size=BATCH_SIZE, shuffle=False)
calibration_data, _ = next(iter(calibration_loader))
model = load_model(MODEL_PATH)

before_loss, before_acc = evaluate(model, test_loader)
print(f"before quantization: loss={before_loss:.6f}, accuracy={before_acc:.4%}")

quantized_model, report = quantization(model, bits=BITS, solvers=SOLVERS, layers=LAYERS).activation(calibration_data)

after_loss, after_acc = evaluate(quantized_model, test_loader)
print(f"after quantization:  loss={after_loss:.6f}, accuracy={after_acc:.4%}")
print(f"method={report['method']}, bits={report['bits']}, layers={[layer['name'] for layer in report['layers']]}")
