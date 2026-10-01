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
    build_linear_regression_dataset,
)
from kaiwu.torch_plugin import FeatureSelectionWrapper
from kaiwu_license import _init_kaiwu_license_from_env, has_license_env

KAIWU_PROJECT_NO = "Your PROJECT NO"


def main() -> None:
    _init_kaiwu_license_from_env()
    dataset = build_linear_regression_dataset()
    loss_fn = nn.MSELoss()

    local_search_solver_kwargs = {"max_iter": 1000}
    local_search_selector = FeatureSelectionWrapper(
        nn.Linear(dataset.feature_dim, 1, bias=False),
        feature_dim=dataset.feature_dim,
        cardinality_k=len(dataset.signal_features),
        solver="local_search",
        solver_kwargs=local_search_solver_kwargs,
        mask_update_epochs=20,
    )
    local_search_optimizer = torch.optim.SGD(
        local_search_selector.model.parameters(),
        lr=0.05,
    )
    local_search_selector.fit_weights(
        dataset.loader,
        loss_fn,
        local_search_optimizer,
        train_epochs=20,
    )
    local_search_loss = float(
        loss_fn(local_search_selector(dataset.inputs), dataset.targets).detach()
    )
    print(
        {
            "model": "linear_regression",
            "solver": "local_search",
            "loss": round(local_search_loss, 6),
            "signal_features": dataset.signal_features,
            "selected_features": local_search_selector.selected_indices().tolist(),
        }
    )

    if not has_license_env():
        print(
            "Skipping the sa and kaiwu_cim solvers: both route through the"
            " Kaiwu SDK and need a license. Set LICENSE_USER_ID and"
            " LICENSE_SDK_CODE (see this directory README) to run them."
        )
        return

    sa_solver_kwargs = {
        "alpha": 0.99,
        "size_limit": 1,
        "rand_seed": 0,
    }
    sa_selector = FeatureSelectionWrapper(
        nn.Linear(dataset.feature_dim, 1, bias=False),
        feature_dim=dataset.feature_dim,
        cardinality_k=len(dataset.signal_features),
        solver="sa",
        solver_kwargs=sa_solver_kwargs,
        mask_update_epochs=20,
    )
    sa_optimizer = torch.optim.SGD(sa_selector.model.parameters(), lr=0.05)
    sa_selector.fit_weights(
        dataset.loader,
        loss_fn,
        sa_optimizer,
        train_epochs=20,
    )
    sa_loss = float(loss_fn(sa_selector(dataset.inputs), dataset.targets).detach())
    print(
        {
            "model": "linear_regression",
            "solver": "sa",
            "loss": round(sa_loss, 6),
            "signal_features": dataset.signal_features,
            "selected_features": sa_selector.selected_indices().tolist(),
        }
    )

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
    kaiwu_cim_selector = FeatureSelectionWrapper(
        nn.Linear(dataset.feature_dim, 1, bias=False),
        feature_dim=dataset.feature_dim,
        cardinality_k=len(dataset.signal_features),
        solver="kaiwu_cim",
        solver_kwargs=kaiwu_cim_solver_kwargs,
        mask_update_epochs=5,
    )
    kaiwu_cim_optimizer = torch.optim.SGD(
        kaiwu_cim_selector.model.parameters(),
        lr=0.05,
    )
    kaiwu_cim_selector.fit_weights(
        dataset.loader,
        loss_fn,
        kaiwu_cim_optimizer,
        train_epochs=10,
    )
    kaiwu_cim_loss = float(
        loss_fn(kaiwu_cim_selector(dataset.inputs), dataset.targets).detach()
    )
    print(
        {
            "model": "linear_regression",
            "solver": "kaiwu_cim",
            "loss": round(kaiwu_cim_loss, 6),
            "signal_features": dataset.signal_features,
            "selected_features": kaiwu_cim_selector.selected_indices().tolist(),
        }
    )


if __name__ == "__main__":
    main()
