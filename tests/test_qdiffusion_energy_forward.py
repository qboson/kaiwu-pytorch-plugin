"""EnergyModel.forward must delegate with the same arguments."""

import torch

from kaiwu.torch_plugin.qdiffusion import EnergyModel


class Recorder(EnergyModel):
    """Records the delegated arguments and returns constant scores."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Stores the arguments and returns constant energies."""
        self.calls.append((noisy_tokens, candidate_tokens, attention_mask))
        return torch.full((candidate_tokens.size(0), 1), 3.0)


def test_forward_passes_each_argument_to_score_conditioned():
    energy = Recorder()
    noisy = torch.tensor([[5, 6]])
    candidate = torch.tensor([[5, 7]])
    mask = torch.tensor([[True, False]])

    scores = energy(noisy, candidate, mask)

    got_noisy, got_candidate, got_mask = energy.calls[-1]
    torch.testing.assert_close(got_noisy, noisy)
    torch.testing.assert_close(got_candidate, candidate)
    torch.testing.assert_close(got_mask, mask)
    torch.testing.assert_close(scores, torch.full((1, 1), 3.0))
