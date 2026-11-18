"""Fill missing binary pixels with conditional Boltzmann Machine samples.

Run from the repository root after installing the plugin with ``pip install .``::

    python example/bm_conditional_imputation.py
"""

from itertools import product

import torch

from kaiwu.torch_plugin import BoltzmannMachine


@torch.no_grad()
def main() -> None:
    """Compare Gibbs estimates with exact conditional probabilities on CPU."""
    torch.manual_seed(42)

    # The first row of each 2x2 grid is observed. Each missing pixel tends
    # to agree with the pixel above it, and the missing pixels also interact.
    observed = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    couplings = torch.zeros(4, 4)
    couplings[0, 2] = couplings[1, 3] = 4.0
    couplings[2, 3] = 0.75
    model = BoltzmannMachine(
        num_nodes=4,
        quadratic_coef=couplings,
        linear_bias=torch.tensor([0.0, 0.0, -2.0, -2.0]),
        device=torch.device("cpu"),
    )

    # Gibbs sampling holds a prefix of the units fixed in each chain.
    # Start several independent chains for each observed row.
    num_chains = 2048
    clamped = observed.repeat_interleave(num_chains, dim=0)
    samples = model.gibbs_sample(num_steps=30, s_visible=clamped)
    missing_samples = samples[:, 2:].reshape(len(observed), num_chains, 2)
    estimated = missing_samples.mean(dim=1)

    # With only two missing pixels we can enumerate their four states.
    # Conditional probabilities are proportional to exp(-energy).
    completions = torch.tensor(list(product((0.0, 1.0), repeat=2)))
    print("Observed row | Gibbs P(pixel=1) | Exact P(pixel=1) | Imputed row")
    for row, probabilities in zip(observed, estimated):
        states = torch.cat((row.expand(len(completions), -1), completions), dim=1)
        exact = torch.softmax(-model(states), dim=0) @ completions
        imputed = (probabilities >= 0.5).int().tolist()
        print(
            f"{row.int().tolist()} | "
            f"{[round(p, 3) for p in probabilities.tolist()]} | "
            f"{[round(p, 3) for p in exact.tolist()]} | {imputed}"
        )


if __name__ == "__main__":
    main()
