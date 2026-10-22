"""FeatureSelectionWrapper must validate the wrapped model."""

import pytest
import torch

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper


@pytest.mark.parametrize("model", [None, "linear", lambda x: x])
def test_non_module_model_is_rejected(model) -> None:
    with pytest.raises(TypeError, match="model must be a torch.nn.Module"):
        FeatureSelectionWrapper(model, feature_dim=4)


def test_module_model_is_accepted() -> None:
    selector = FeatureSelectionWrapper(torch.nn.Linear(4, 1), feature_dim=4)
    assert isinstance(selector.model, torch.nn.Module)
