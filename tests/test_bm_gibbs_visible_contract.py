"""gibbs_sample must keep the clamped visible units unchanged."""

import torch

from kaiwu.torch_plugin import BoltzmannMachine


def test_gibbs_sample_preserves_the_visible_prefix() -> None:
    torch.manual_seed(0)
    model = BoltzmannMachine(num_nodes=4)
    visible = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    samples = model.gibbs_sample(3, s_visible=visible)

    assert samples.shape == (2, 4)
    torch.testing.assert_close(samples[:, :3], visible)
    assert torch.all((samples == 0) | (samples == 1))
