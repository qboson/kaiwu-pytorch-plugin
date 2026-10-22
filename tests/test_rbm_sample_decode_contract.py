"""Pin the spin-to-binary decoding performed by sampler-backed sample()."""

import numpy as np

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


class _SpinSampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        del ising_matrix
        return np.array(
            [
                [1, -1, 1, -1, 1, 1],
                [-1, 1, -1, 1, -1, -1],
            ],
            dtype=np.float32,
        )


def test_sample_decodes_spins_with_the_auxiliary_gauge() -> None:
    model = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)

    binary = model.sample(_SpinSampler())

    expected = [
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
    ]
    assert binary.tolist() == expected
