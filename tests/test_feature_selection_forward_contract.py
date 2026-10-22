"""FeatureSelectionWrapper.forward must apply the current mask."""

import torch
from torch import nn

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper


def _selector() -> FeatureSelectionWrapper:
    model = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    selector = FeatureSelectionWrapper(model, feature_dim=4)
    with torch.no_grad():
        selector.mask.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0]))
    return selector


def test_forward_equals_the_model_on_the_masked_input() -> None:
    selector = _selector()
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    output = selector(inputs)

    expected = selector.model(inputs * selector.mask)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(output, torch.tensor([[4.0], [12.0]]))
