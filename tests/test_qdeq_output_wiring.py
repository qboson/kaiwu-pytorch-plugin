"""QDEQ must return the layer output at the solved point, not the point itself."""

import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class ScaledInjection(torch.nn.Module):
    """``layer(z, x) = 2 z + x``; its output differs from its input at every z."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, z, x):
        """Records the arguments and returns ``2 z + x``."""
        self.calls.append((z.detach().clone(), x.detach().clone()))
        return 2.0 * z + x


class FixedPointQDEQ(QDEQ):
    """Returns a fixed point supplied by the test instead of iterating."""

    def __init__(self, layer, z_star):
        super().__init__(layer, None, QDEQConfig())
        self.z_star = z_star

    def solve_fixed_point(self, func, z0):
        """Returns the stored point without calling ``func``."""
        del func, z0
        return self.z_star.clone(), {"nstep": 0}


def test_output_is_the_layer_value_at_the_solved_point():
    layer = ScaledInjection()
    z_star = torch.full((2, 3), 1.0)
    x = torch.full((2, 3), 3.0)
    deq = FixedPointQDEQ(layer, z_star).eval()

    out = deq(x)

    torch.testing.assert_close(out, 2.0 * z_star + x)
    assert not torch.equal(out, z_star)


def test_the_final_call_receives_the_solved_point_and_the_input():
    layer = ScaledInjection()
    z_star = torch.full((2, 3), 1.0)
    x = torch.full((2, 3), 3.0)
    deq = FixedPointQDEQ(layer, z_star).eval()

    deq(x)

    z_arg, x_arg = layer.calls[-1]
    torch.testing.assert_close(z_arg, z_star)
    torch.testing.assert_close(x_arg, x)
