import torch

from torch.optim import SGD
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import BoltzmannMachine

#  这里添加licence认证


if __name__ == "__main__":
    SAMPLE_SIZE = 5

    sampler = SimulatedAnnealingOptimizer(alpha=0.99, size_limit=5)
    sample_kwargs = {}
    num_nodes = 5
    num_visible = 2
    x = 1.0 * torch.randint(0, 2, (SAMPLE_SIZE, num_visible))

    # Instantiate the model
    rbm = BoltzmannMachine(num_nodes)

    # Instantiate the optimizer
    opt_rbm = SGD(rbm.parameters())

    # Example of one iteration in a training loop
    # Generate a sample set from the model

    # Keep each observation's conditional solutions together: a sampler may
    # return a different number of solutions for each observation.
    conditional_states = [
        rbm.condition_sample(sampler, row.unsqueeze(0)) for row in x
    ]
    s = rbm.sample(sampler)
    opt_rbm.zero_grad()
    # Compute the objective---this objective yields the same gradient as the negative
    # log likelihood of the model
    positive_energy = torch.stack(
        [rbm(states).mean() for states in conditional_states]
    ).mean()
    objective = positive_energy - rbm(s).mean()
    # Backpropgate gradients
    print("call backward")
    objective.backward()
    print("after backward")
    # Update model weights with a step of stochastic gradient descent
    opt_rbm.step()
    print(objective)
