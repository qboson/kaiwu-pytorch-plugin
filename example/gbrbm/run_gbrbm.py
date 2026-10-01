"""Train a Gaussian-Bernoulli RBM on continuous data.

This example trains a ``GaussianBernoulliRestrictedBoltzmannMachine`` on
synthetic correlated continuous data with contrastive divergence:

* the positive phase completes the data with sampled Bernoulli units
  (``infer_from_gaussian``);
* the negative phase draws model states with local Gibbs sampling
  (``gibbs_sample``), so the script runs without a Kaiwu SDK license;
* the inherited objective ``bm.objective(s_positive, s_negative)`` yields
  the same gradient as the negative log-likelihood of the model.

After training, the script generates fresh continuous samples from the
model (``gibbs_sample`` with a burn-in) and compares their per-dimension
statistics with the data, then scores both with ``marginal_energy``.

Set ``USE_SOLVER = True`` to draw the negative phase from the Kaiwu SDK
simulated-annealing sampler instead of local Gibbs. That path requires a
Kaiwu license; see the README for instructions.
"""

import torch

# Deep import keeps this example independent of the pending package-root
# export (PR #197); the root path `from kaiwu.torch_plugin import ...`
# works as well once that change lands.
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine

# Set to True to draw the negative phase from the Kaiwu SDK sampler
# (requires a license); False keeps the whole example local and
# license-free.
USE_SOLVER = False


def make_dataset(n_sample, num_visible=4, seed=0):
    """Draw a correlated continuous dataset.

    The Gaussian energy of the GBRBM is diagonal in the visible units, so
    the model can match per-dimension means and variances; the off-diagonal
    mixing below makes the data non-trivial without changing that target.

    Args:
        n_sample (int): Number of samples to draw.
        num_visible (int, optional): Dimensionality of the data.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.Tensor: Data tensor of shape (n_sample, num_visible).
    """
    generator = torch.Generator().manual_seed(seed)
    mean = torch.tensor([1.0, -0.5, 0.25, -1.25])
    mix = torch.tensor(
        [
            [1.0, 0.4, 0.0, 0.0],
            [0.4, 1.0, 0.3, 0.0],
            [0.0, 0.3, 1.0, 0.2],
            [0.0, 0.0, 0.2, 1.0],
        ]
    )
    noise = torch.randn(n_sample, num_visible, generator=generator)
    return noise @ mix + mean


def train(bm, data, epochs=300, lr=0.05, cd_steps=1, log_every=100):
    """Train the GBRBM with contrastive divergence.

    Each epoch completes the data batch with sampled Bernoulli units
    (positive phase), draws one model sweep from the same batch with
    ``gibbs_sample`` (negative phase), and takes one SGD step on the
    contrastive objective.

    Args:
        bm (GaussianBernoulliRestrictedBoltzmannMachine): Model to train.
        data (torch.Tensor): Continuous data of shape (n_sample, num_gaussian).
        epochs (int, optional): Number of parameter updates.
        lr (float, optional): SGD learning rate.
        cd_steps (int, optional): Gibbs steps per negative sample.
        log_every (int, optional): Epoch interval for progress logging.

    Returns:
        list[float]: Contrastive objective value per epoch.
    """
    optimizer = torch.optim.SGD(bm.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        s_positive = bm.infer_from_gaussian(data)
        s_negative = bm.gibbs_sample(n_step=cd_steps, s_gaussian=data)
        loss = bm.objective(s_positive, s_negative)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if log_every and (epoch + 1) % log_every == 0:
            print(f"epoch {epoch + 1:>4d}  objective {loss.item():>10.4f}")
    return history


def train_with_solver(bm, data, epochs=300, lr=0.05):
    """Train the GBRBM with the Kaiwu SDK sampler as the negative phase.

    This is the quantum-plugin counterpart of :func:`train`: instead of a
    local Gibbs sweep, the Bernoulli-side Ising model is solved by the
    Kaiwu SDK optimizer through ``bm.sample(sampler)``. It requires a Kaiwu
    license (``kw.license.init``), unlike the license-free local path.

    Args:
        bm (GaussianBernoulliRestrictedBoltzmannMachine): Model to train.
        data (torch.Tensor): Continuous data of shape (n_sample, num_gaussian).
        epochs (int, optional): Number of parameter updates.
        lr (float, optional): SGD learning rate.

    Returns:
        list[float]: Contrastive objective value per epoch.
    """
    import os

    import kaiwu as kw
    from kaiwu.classical import SimulatedAnnealingOptimizer

    # Fill in after registering on platform.qboson.com:
    # kw.license.init(os.getenv("USER_ID"), os.getenv("SDK_CODE"))
    sampler = SimulatedAnnealingOptimizer()

    optimizer = torch.optim.SGD(bm.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        s_positive = bm.infer_from_gaussian(data)
        s_negative = bm.sample(sampler)
        loss = bm.objective(s_positive, s_negative)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch + 1:>4d}  objective {loss.item():>10.4f}")
    return history


def generate(bm, n_sample, n_step=60, n_burnin=30, seed=0):
    """Generate continuous samples from the trained model.

    A Gibbs chain is started from random Gaussian states and run for
    ``n_step`` sweeps; the first ``n_burnin`` sweeps warm the chain up and
    the remaining sweeps are returned as model samples.

    Args:
        bm (GaussianBernoulliRestrictedBoltzmannMachine): Trained model.
        n_sample (int): Number of parallel chains to run.
        n_step (int, optional): Total Gibbs sweeps per chain.
        n_burnin (int, optional): Warm-up sweeps discarded per chain.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.Tensor: Generated states of shape (n_kept * n_sample, num_nodes).
    """
    torch.manual_seed(seed)
    return bm.gibbs_sample(n_step=n_step, n_burnin=n_burnin, n_sample=n_sample)


def main():
    torch.manual_seed(0)

    num_visible = 4  # Gaussian (continuous) units
    num_hidden = 3  # Bernoulli (binary) units
    data = make_dataset(n_sample=256, num_visible=num_visible, seed=0)

    bm = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=num_visible,
        num_hidden=num_hidden,
    )

    print("training on continuous data (local Gibbs negative phase)...")
    if USE_SOLVER:
        history = train_with_solver(bm, data)
    else:
        history = train(bm, data)
    print(f"objective: start {history[0]:.4f} -> end {history[-1]:.4f}")

    samples = generate(bm, n_sample=256)
    generated = samples[:, : bm.num_gaussian]

    print("\nper-dimension mean (data vs generated):")
    for j in range(num_visible):
        print(
            f"  dim {j}: {data[:, j].mean().item():>8.4f}"
            f"  {generated[:, j].mean().item():>8.4f}"
        )

    print("\nper-dimension std (data vs generated):")
    for j in range(num_visible):
        print(
            f"  dim {j}: {data[:, j].std().item():>8.4f}"
            f"  {generated[:, j].std().item():>8.4f}"
        )

    energy_data = bm.marginal_energy(data)
    energy_generated = bm.marginal_energy(generated)
    print(
        f"\nmarginal energy (free energy of Gaussian states):"
        f" data {energy_data.mean().item():.4f},"
        f" generated {energy_generated.mean().item():.4f}"
    )


if __name__ == "__main__":
    main()
