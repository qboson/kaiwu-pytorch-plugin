"""Classifier evaluation can test generalization to distinct donors."""

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")
pytest.importorskip("scanpy")

from example.qvae_cell.evaluate_benchmark import evaluate_classifier, make_long_summary


@pytest.fixture
def donor_data():
    # The features identify donor-specific cell states without any class
    # signal shared across donors.
    cells_per_state = 8
    states = np.repeat(np.arange(8), cells_per_state)
    features = np.eye(8)[states]
    obs = pd.DataFrame({
        "label": np.where(states % 2 == 0, "early", "late"),
        "donor": (states // 2).astype(str),
    }, index=np.arange(len(states)).astype(str))
    data = anndata.AnnData(np.zeros((len(states), 1)), obs=obs)
    data.obsm["X_qvae"] = features
    return data


def test_group_holdout_prevents_donor_specific_features_leaking(donor_data):
    args = (donor_data, "X_qvae", "label", 0.25, 42, 100)
    random_cells = evaluate_classifier(*args)
    held_out = evaluate_classifier(*args, group_key="donor")

    assert random_cells.at[0, "accuracy"] == pytest.approx(1.0)
    assert held_out.at[0, "accuracy"] == pytest.approx(0.5)
    assert held_out.at[0, "group_key"] == "donor"
    assert held_out.at[0, "n_train_groups"] == 3
    assert held_out.at[0, "n_test_groups"] == 1
    assert "group_key" not in make_long_summary([random_cells], "label")
    assert make_long_summary([held_out], "label")["group_key"].unique().tolist() == ["donor"]


def test_group_holdout_reports_missing_group_labels(donor_data):
    donor_data.obs.iloc[0, donor_data.obs.columns.get_loc("donor")] = None
    with pytest.raises(ValueError, match="donor.*no missing values"):
        evaluate_classifier(donor_data, "X_qvae", "label", 0.25, 42, 100, group_key="donor")
