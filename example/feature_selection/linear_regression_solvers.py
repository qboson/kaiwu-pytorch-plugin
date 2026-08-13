from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
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


def build_dataset() -> tuple[
    DataLoader,
    torch.Tensor,
    torch.Tensor,
    list[int],
    int,
]:
    sample_count = 1024
    feature_dim = 100
    signal_count = 5

    rng = np.random.default_rng()
    signal_indices = rng.choice(feature_dim, size=signal_count, replace=False)
    signal_weights = rng.uniform(5.0, 10.0, size=signal_count)
    signal_weights *= rng.choice([-1.0, 1.0], size=signal_count)

    inputs = torch.randn(sample_count, feature_dim)
    true_weight = torch.zeros(feature_dim, 1)
    true_weight[torch.tensor(signal_indices, dtype=torch.long)] = torch.tensor(
        signal_weights,
        dtype=true_weight.dtype,
    ).view(-1, 1)
    targets = inputs @ true_weight + 0.001 * torch.randn(sample_count, 1)

    loader = DataLoader(TensorDataset(inputs, targets), batch_size=128, shuffle=False)
    return loader, inputs, targets, sorted(signal_indices.tolist()), feature_dim


def make_solver_kwargs(solver: str) -> dict[str, object]:
    if solver == "local_search":
        return {"max_iter": 1000}
    if solver == "sa":
        return {"max_iter": 50000}
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
    solver: str,
    loader: DataLoader,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    signal_indices: list[int],
    feature_dim: int,
) -> dict[str, object]:
    model = nn.Linear(feature_dim, 1, bias=False)
    selector = FeatureSelectionWrapper(
        model,
        feature_dim=feature_dim,
        cardinality_k=len(signal_indices),
        solver=solver,
        solver_kwargs=make_solver_kwargs(solver),
        mask_update_epochs=20 if solver != "kaiwu_cim" else 5,
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(selector.model.parameters(), lr=0.05)
    selector.fit_weights(
        loader,
        loss_fn,
        optimizer,
        train_epochs=20 if solver != "kaiwu_cim" else 10,
    )

    loss = float(loss_fn(selector(inputs), targets).detach())
    return {
        "solver": solver,
        "loss": round(loss, 6),
        "signal_indices": signal_indices,
        "selected_indices": selector.selected_indices().tolist(),
    }


def main() -> None:
    _init_kaiwu_license_from_env()
    loader, inputs, targets, signal_indices, feature_dim = build_dataset()
    results = [
        run_feature_selection(
            solver,
            loader,
            inputs,
            targets,
            signal_indices,
            feature_dim,
        )
        for solver in ("local_search", "sa", "kaiwu_cim")
    ]
    print({"model": "linear_regression", "results": results})


if __name__ == "__main__":
    main()
