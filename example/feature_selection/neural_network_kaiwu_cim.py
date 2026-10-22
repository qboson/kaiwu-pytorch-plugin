from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
LOCAL_KAIWU_PACKAGE = str(SRC_ROOT / "kaiwu")
sys.path.insert(0, str(SRC_ROOT))

import kaiwu

kaiwu_search_path = list(getattr(kaiwu, "__path__", []))
if LOCAL_KAIWU_PACKAGE not in kaiwu_search_path:
    kaiwu_search_path.append(LOCAL_KAIWU_PACKAGE)
    kaiwu.__path__ = kaiwu_search_path

from feature_selection_datasets import (
    build_cnn_dataset,
    build_sequence_dataset,
)
from feature_selection_models import SimpleLSTM, SimpleRNN, TinyCNN
from kaiwu.torch_plugin import FeatureSelectionWrapper
from kaiwu_license import _init_kaiwu_license_from_env, has_license_env

KAIWU_PROJECT_NO = "Your PROJECT NO"


def main() -> None:
    if not has_license_env():
        raise SystemExit(
            "This example uses the kaiwu_cim solver, which routes through"
            " the Kaiwu SDK and needs a license. Set LICENSE_USER_ID and"
            " LICENSE_SDK_CODE (see this directory README) first."
        )
    _init_kaiwu_license_from_env()
    cnn_dataset = build_cnn_dataset()
    sequence_dataset = build_sequence_dataset()
    loss_fn = nn.CrossEntropyLoss()

    kaiwu_cim_solver_kwargs = {
        "target_precision": 8,
        "max_bits": 1000,
        "max_precision": 8,
        "precision_step": 4,
        "sample_number": 1,
        "task_mode": "quota",
        "interval": 1,
        "project_no": KAIWU_PROJECT_NO,
    }

    cnn_selector = FeatureSelectionWrapper(
        TinyCNN(image_size=8),
        feature_dim=cnn_dataset.feature_dim,
        cardinality_k=len(cnn_dataset.signal_features),
        solver="kaiwu_cim",
        solver_kwargs=kaiwu_cim_solver_kwargs,
        mask_update_epochs=5,
    )
    cnn_optimizer = torch.optim.Adam(cnn_selector.model.parameters(), lr=0.01)
    cnn_selector.fit_weights(cnn_dataset.loader, loss_fn, cnn_optimizer, train_epochs=20)
    cnn_logits = cnn_selector(cnn_dataset.inputs)
    cnn_accuracy = (
        (cnn_logits.argmax(dim=1) == cnn_dataset.targets).float().mean().item()
    )
    print(
        {
            "model": "cnn",
            "loss": round(float(loss_fn(cnn_logits, cnn_dataset.targets).detach()), 6),
            "accuracy": round(float(cnn_accuracy), 6),
            "signal_features": cnn_dataset.signal_features,
            "selected_features": cnn_selector.selected_indices().tolist(),
        }
    )

    rnn_selector = FeatureSelectionWrapper(
        SimpleRNN(sequence_dataset.feature_dim),
        feature_dim=sequence_dataset.feature_dim,
        cardinality_k=len(sequence_dataset.signal_features),
        solver="kaiwu_cim",
        solver_kwargs=kaiwu_cim_solver_kwargs,
        mask_update_epochs=5,
    )
    rnn_optimizer = torch.optim.Adam(rnn_selector.model.parameters(), lr=0.01)
    rnn_selector.fit_weights(
        sequence_dataset.loader,
        loss_fn,
        rnn_optimizer,
        train_epochs=20,
    )
    rnn_logits = rnn_selector(sequence_dataset.inputs)
    rnn_accuracy = (
        (rnn_logits.argmax(dim=1) == sequence_dataset.targets).float().mean().item()
    )
    print(
        {
            "model": "rnn",
            "loss": round(
                float(loss_fn(rnn_logits, sequence_dataset.targets).detach()),
                6,
            ),
            "accuracy": round(float(rnn_accuracy), 6),
            "signal_features": sequence_dataset.signal_features,
            "selected_features": rnn_selector.selected_indices().tolist(),
        }
    )

    lstm_selector = FeatureSelectionWrapper(
        SimpleLSTM(sequence_dataset.feature_dim),
        feature_dim=sequence_dataset.feature_dim,
        cardinality_k=len(sequence_dataset.signal_features),
        solver="kaiwu_cim",
        solver_kwargs=kaiwu_cim_solver_kwargs,
        mask_update_epochs=5,
    )
    lstm_optimizer = torch.optim.Adam(lstm_selector.model.parameters(), lr=0.01)
    lstm_selector.fit_weights(
        sequence_dataset.loader,
        loss_fn,
        lstm_optimizer,
        train_epochs=20,
    )
    lstm_logits = lstm_selector(sequence_dataset.inputs)
    lstm_accuracy = (
        (lstm_logits.argmax(dim=1) == sequence_dataset.targets).float().mean().item()
    )
    print(
        {
            "model": "lstm",
            "loss": round(
                float(loss_fn(lstm_logits, sequence_dataset.targets).detach()),
                6,
            ),
            "accuracy": round(float(lstm_accuracy), 6),
            "signal_features": sequence_dataset.signal_features,
            "selected_features": lstm_selector.selected_indices().tolist(),
        }
    )


if __name__ == "__main__":
    main()
