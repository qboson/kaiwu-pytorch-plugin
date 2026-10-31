"""Replay a QDEQ computation with an owned RNG and a tiny offline Ising solver.

Run with an installed kaiwu-torch-plugin. No dataset, license, or cloud task
is needed; exhaustive enumeration is suitable only for this tiny example.
"""

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


@dataclass
class ExactSampler:
    """Maximize a tiny Ising spin objective by complete enumeration."""

    max_spins: int = 10

    def solve(self, matrix):
        """Return one maximizing row, including the auxiliary spin."""
        if matrix.shape[0] > self.max_spins:
            raise ValueError(f"This demo enumerates at most {self.max_spins} spins")
        spins = np.array(list(product((-1., 1.), repeat=matrix.shape[0])))
        scores = np.einsum("bi,ij,bj->b", spins, matrix, spins)
        return spins[[int(scores.argmax())]]


class AffineLayer(torch.nn.Module):
    """Contractive map with an analytic equilibrium x / (1 - scale)."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))

    def forward(self, state, inputs):
        """Return the next equilibrium iterate."""
        return self.scale * state + inputs


def make_model(generator):
    """Use a tiny QUBO and two Jacobian probes to exercise both RNG paths."""
    config = QDEQConfig(num_basis=2, coeff_bits=2, f_thres=12, jac_loss_vecs=2)
    return QDEQ(AffineLayer(), ExactSampler(), config, generator=generator)


def main():
    """Save a continuation point, perturb global randomness, and replay it."""
    model = make_model(torch.Generator(device="cpu").manual_seed(17))
    inputs = torch.tensor([[0.2, -0.4]], dtype=torch.float64)
    model(inputs)
    with tempfile.TemporaryDirectory(prefix="qdeq-rng-") as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "rng": model.generator.get_state()}, path)
        global_state = torch.get_rng_state()
        expected = model(inputs)
        penalty = model.last_jac_loss.detach().clone()
        isolated = torch.equal(global_state, torch.get_rng_state())

        checkpoint = torch.load(path, weights_only=True)
        restored = make_model(torch.Generator(device="cpu"))
        restored.load_state_dict(checkpoint["model"])
        restored.generator.set_state(checkpoint["rng"])
        torch.rand(100)  # Unrelated application randomness must not affect QDEQ.
        actual = restored(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(restored.last_jac_loss, penalty, rtol=0, atol=0)
        (actual.square().sum() + restored.last_jac_loss).backward()
        assert isolated
        assert torch.isfinite(restored.layer.scale.grad)
        print(json.dumps({"output_replayed": True, "jacobian_probes_replayed": True,
                          "global_rng_preserved": isolated, "finite_gradient": True}))


if __name__ == "__main__":
    main()
