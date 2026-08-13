from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from kaiwu.cim._optimizer_adapter import TaskMode
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kaiwu.torch_plugin import FeatureSelectionWrapper

KAIWU_PROJECT_NO = "Your Project No"


def _init_kaiwu_license_from_env() -> None:
    """Initialize the Kaiwu license from environment variables.

    Raises:
        RuntimeError: If Kaiwu license initialization fails.
    """
    user_id = os.environ.get("LICENSE_USER_ID")
    sdk_code = os.environ.get("LICENSE_SDK_CODE")
    if not user_id or not sdk_code:
        return

    import kaiwu.license as license_manager

    try:
        license_manager.init(user_id, sdk_code)
    except Exception as exc:
        raise RuntimeError(
            "Kaiwu license initialization failed. Check whether LICENSE_USER_ID "
            "and LICENSE_SDK_CODE are correct, and whether the machine can reach "
            "the Kaiwu license server."
        ) from exc


class TinyCNN(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * self.image_size * self.image_size, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        images = inputs.view(-1, 1, self.image_size, self.image_size)
        return self.net(images)


class SimpleRNN(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.rnn = nn.RNN(feature_dim, hidden_size=8, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _sequence, hidden = self.rnn(inputs)
        return self.head(hidden[-1])


class SimpleLSTM(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_size=8, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _sequence, (hidden, _cell) = self.lstm(inputs)
        return self.head(hidden[-1])


def build_cnn_dataset() -> tuple[DataLoader, torch.Tensor, torch.Tensor, list[int], int]:
    sample_count = 20
    image_size = 8
    feature_dim = image_size * image_size
    signal_features = [10, 45]

    inputs = torch.randn(sample_count, feature_dim)
    score = inputs[:, signal_features[0]] - inputs[:, signal_features[1]]
    targets = (score > 0).long()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=10, shuffle=False)
    return loader, inputs, targets, signal_features, feature_dim


def build_sequence_dataset() -> tuple[
    DataLoader,
    torch.Tensor,
    torch.Tensor,
    list[int],
    int,
]:
    sample_count = 20
    sequence_length = 6
    feature_dim = 8
    signal_features = [1, 6]

    inputs = torch.randn(sample_count, sequence_length, feature_dim)
    score = (
        inputs[:, :, signal_features[0]].mean(dim=1)
        - inputs[:, :, signal_features[1]].mean(dim=1)
    )
    targets = (score > 0).long()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=10, shuffle=False)
    return loader, inputs, targets, signal_features, feature_dim


def make_kaiwu_solver_kwargs() -> dict[str, object]:
    return {
        "target_precision": 8,
        "max_bits": 1000,
        "max_precision": 8,
        "precision_step": 4,
        "sample_number": 10,
        "task_mode": TaskMode.SAMPLE,
        "project_no": KAIWU_PROJECT_NO,
    }


def run_feature_selection(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    signal_features: list[int],
    feature_dim: int,
) -> dict[str, object]:
    selector = FeatureSelectionWrapper(
        model,
        feature_dim=feature_dim,
        cardinality_k=len(signal_features),
        solver="kaiwu_cim",
        solver_kwargs=make_kaiwu_solver_kwargs(),
        mask_update_epochs=5,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(selector.model.parameters(), lr=0.01)
    selector.fit_weights(loader, loss_fn, optimizer, train_epochs=20)

    logits = selector(inputs)
    accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return {
        "model": model_name,
        "loss": round(float(loss_fn(logits, targets).detach()), 6),
        "accuracy": round(float(accuracy), 6),
        "signal_features": signal_features,
        "selected_features": selector.selected_indices().tolist(),
    }


def main() -> None:
    _init_kaiwu_license_from_env()
    cnn_loader, cnn_inputs, cnn_targets, cnn_signal_features, cnn_feature_dim = (
        build_cnn_dataset()
    )
    sequence_loader, sequence_inputs, sequence_targets, signal_features, feature_dim = (
        build_sequence_dataset()
    )
    results = [
        run_feature_selection(
            "cnn",
            TinyCNN(image_size=8),
            cnn_loader,
            cnn_inputs,
            cnn_targets,
            cnn_signal_features,
            cnn_feature_dim,
        ),
        run_feature_selection(
            "rnn",
            SimpleRNN(feature_dim),
            sequence_loader,
            sequence_inputs,
            sequence_targets,
            signal_features,
            feature_dim,
        ),
        run_feature_selection(
            "lstm",
            SimpleLSTM(feature_dim),
            sequence_loader,
            sequence_inputs,
            sequence_targets,
            signal_features,
            feature_dim,
        ),
    ]
    print({"solver": "kaiwu_cim", "results": results})


if __name__ == "__main__":
    main()
