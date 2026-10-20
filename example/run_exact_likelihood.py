"""Check a two-node BM's likelihood gradient by exact CPU enumeration.

Run from the repository root: python example/run_exact_likelihood.py
Requires the normal Kaiwu-PyTorch-Plugin installation; no extra dependencies.
"""
from itertools import product

import torch

from kaiwu.torch_plugin import BoltzmannMachine


def main():
    """Compare NLL and objective gradients, then take one gradient-descent step."""
    model = BoltzmannMachine(
        2,
        quadratic_coef=torch.zeros(2, 2, dtype=torch.float64),
        linear_bias=torch.zeros(2, dtype=torch.float64),
        device="cpu",
    )
    configurations = torch.tensor(
        list(product((0.0, 1.0), repeat=2)), dtype=torch.float64
    )
    observed = torch.ones(1, 2, dtype=torch.float64)

    # All four configurations contribute to the exact partition function.
    energies = model(configurations)
    log_partition = torch.logsumexp(-energies, dim=0)
    nll = model(observed).mean() + log_partition
    exact_gradient = torch.autograd.grad(nll, model.quadratic_coef)[0]
    probability_before = torch.exp(-nll.detach())

    # At this zero-parameter starting point the model is uniform. Including
    # each configuration once therefore gives its exact negative-phase mean.
    # For a nonuniform model, enumeration must use model-probability weights.
    objective = model.objective(observed, configurations)
    objective_gradient = torch.autograd.grad(objective, model.quadratic_coef)[0]
    torch.testing.assert_close(exact_gradient, objective_gradient)
    torch.testing.assert_close(
        exact_gradient[0, 1], torch.tensor(-0.75, dtype=torch.float64)
    )

    # d(NLL)/dw = model correlation - data correlation: 0.25 - 1 = -0.75.
    # Gradient descent increases the edge weight, while the biases stay fixed.
    learning_rate = 0.1
    with torch.no_grad():
        model.quadratic_coef.sub_(learning_rate * exact_gradient)
        new_log_partition = torch.logsumexp(-model(configurations), dim=0)
        new_nll = model(observed).mean() + new_log_partition
        probability_after = torch.exp(-new_nll)
    assert probability_after > probability_before

    print(f"Exact NLL edge gradient: {exact_gradient[0, 1].item():.6f}")
    print(f"Existing objective edge gradient: {objective_gradient[0, 1].item():.6f}")
    print(
        "Observed-state probability after one edge-weight gradient step: "
        f"{probability_before.item():.6f} -> {probability_after.item():.6f}"
    )


if __name__ == "__main__":
    main()
