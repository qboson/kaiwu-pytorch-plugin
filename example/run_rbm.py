import torch

from torch.optim import SGD
import kaiwu as kw
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.cim import CIMOptimizer, PrecisionReducer

from kaiwu.cim import CIMOptimizer, PrecisionReducer


# 添加licence认证
# print("User ID:", os.getenv("USER_ID"), "SDK Code:", os.getenv("SDK_CODE"))
# kw.license.init(os.getenv("USER_ID"), os.getenv("SDK_CODE"))


if __name__ == "__main__":
    USE_QPU = False
    NUM_READS = 1
    SAMPLE_SIZE = 1
    USE_CIM = False

    if USE_CIM:
        kw.common.CheckpointManager.save_dir = "./tmp"
        sampler = CIMOptimizer(task_name="test_kpp", wait=True)
        sampler = PrecisionReducer(
            sampler,
            precision=8,
            truncated_precision=10,
            target_bits=550,
            only_feasible_solution=False,
        )
    else:
        sampler = SimulatedAnnealingOptimizer()
    num_nodes = 5
    num_visible = 2
    x = 1.0 * torch.randint(0, 2, (SAMPLE_SIZE, num_visible))

    # Instantiate the model
    rbm = RestrictedBoltzmannMachine(
        num_visible,
        num_nodes - num_visible,
        quadratic_coef=torch.FloatTensor(
            [
                [2, -3, 0],
                [-1, 2, 0],
            ]
        ),
        linear_bias=torch.FloatTensor([1, 1, 0, -1, 2]),
    )
    # Instantiate the optimizer
    opt_rbm = SGD(rbm.parameters())

    # Example of one iteration in a training loop
    # Generate a sample set from the model
    x = rbm.get_hidden(x, bernoulli=True)
    s = rbm.sample(sampler)
    opt_rbm.zero_grad()
    # Compute the objective---this objective yields the same gradient as the negative
    # log likelihood of the model
    objective = rbm.objective(x, s)
    # Update model weights with a step of stochastic gradient descent
    objective.backward()
