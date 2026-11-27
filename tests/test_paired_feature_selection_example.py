"""The solver comparison visits the same first mask objective for all backends."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import kaiwu
import torch
from torch.utils.data import DataLoader, TensorDataset


def test_linear_example_uses_paired_weights_and_the_same_mask_schedule(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "example" / "feature_selection"
    with monkeypatch.context() as local:
        local.syspath_prepend(str(directory))
        local.setattr(kaiwu, "__path__", list(kaiwu.__path__))
        for name in ("linear_regression_solvers", "feature_selection_datasets", "kaiwu_license"):
            local.delitem(sys.modules, name, raising=False)
        script = importlib.import_module("linear_regression_solvers")
        for name in ("linear_regression_solvers", "feature_selection_datasets", "kaiwu_license"):
            sys.modules.pop(name, None)

    observed = {}

    class RecordingSelector(script.FeatureSelectionWrapper):
        def __init__(self, model, **kwargs):
            super().__init__(model, **kwargs)
            observed[self.solver] = {"initial": model.weight.detach().clone(), "updates": []}

        def update_mask(self, *_args, **_kwargs):
            observed[self.solver]["updates"].append(self.model.weight.detach().clone())
            self.mask.copy_(torch.tensor([1., 0.]))
            return self.get_support()

    inputs = torch.tensor([[1., 2.], [-1., 1.], [0.5, 0.4], [0., -1.]])
    targets = 2 * inputs[:, :1]
    dataset = SimpleNamespace(
        inputs=inputs, targets=targets, feature_dim=2, signal_features=[0],
        loader=DataLoader(TensorDataset(inputs, targets), batch_size=2),
    )
    monkeypatch.setattr(script, "_init_kaiwu_license_from_env", lambda: None)
    monkeypatch.setattr(script, "build_linear_regression_dataset", lambda: dataset)
    monkeypatch.setattr(script, "FeatureSelectionWrapper", RecordingSelector)

    script.main()

    assert set(observed) == {"local_search", "sa", "kaiwu_cim"}
    expected = observed["local_search"]
    for current in observed.values():
        assert len(current["updates"]) == 1
        torch.testing.assert_close(current["initial"], expected["initial"])
        torch.testing.assert_close(current["updates"][0], expected["updates"][0])
        assert not torch.equal(current["initial"], current["updates"][0])
