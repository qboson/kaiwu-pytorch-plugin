"""The fixed-point map must preserve the state tensor's shape."""

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class BroadcastableLayer(nn.Module):
    """Return one feature where the equilibrium state has several."""

    def forward(self, state, inputs):
        return (state + inputs).mean(dim=-1, keepdim=True)


class TupleLayer(nn.Module):
    """Return auxiliary data alongside a state, as some attention layers do."""

    def forward(self, state, inputs):
        return state + inputs, None


@pytest.mark.parametrize(
    ("layer", "error", "message"),
    [
        (BroadcastableLayer(), ValueError, r"shape \(2, 3\).*\(2, 1\)"),
        (TupleLayer(), TypeError, "must return a tensor"),
    ],
)
def test_invalid_fixed_point_map_is_rejected_at_the_layer_boundary(layer, error, message):
    model = QDEQ(layer, object(), QDEQConfig(f_thres=0)).eval()
    with pytest.raises(error, match=message):
        model(torch.ones(2, 3))


def test_initial_state_can_have_a_different_shape_from_conditioning_input():
    class ConditionedLayer(nn.Module):
        def forward(self, state, inputs):
            return state + inputs[:, :1]

    model = QDEQ(ConditionedLayer(), object(), QDEQConfig(f_thres=0)).eval()
    state = torch.zeros(2, 3)
    output = model(torch.ones(2, 4), z0=state)
    torch.testing.assert_close(output, torch.ones_like(state))
