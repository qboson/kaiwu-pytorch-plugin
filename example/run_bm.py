import torch

from torch.optim import SGD
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import BoltzmannMachine

#  这里添加licence认证


if __name__ == "__main__":
    USE_QPU = False
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

    x = rbm.condition_sample(sampler, x)
    s = rbm.sample(sampler)
    opt_rbm.zero_grad()
    # Compute the objective---this objective yields the same gradient as the negative
    # log likelihood of the model
    objective = rbm.objective(x, s)
    # Backpropgate gradients
    print("call backward")
    objective.backward()
    print("after backward")
    # Update model weights with a step of stochastic gradient descent
    opt_rbm.step()
    print(objective)
