import os

import torch

from torch.optim import SGD
import kaiwu as kw
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.cim import CIMOptimizer, PrecisionReducer


def _ensure_license():
    """Initialize the Kaiwu SDK license from the environment or exit.

    The sampler used below requires a Kaiwu license; without credentials
    the SDK drops into an interactive prompt that crashes non-interactive
    runs with ``ValueError: EOF when reading a line``.

    Raises:
        SystemExit: When ``USER_ID`` or ``SDK_CODE`` is not set, with
            setup instructions.
    """
    user_id = os.getenv("USER_ID")
    sdk_code = os.getenv("SDK_CODE")
    if user_id and sdk_code:
        kw.license.init(user_id, sdk_code)
        return
    raise SystemExit(
        "This example needs a Kaiwu SDK license to run.\n"
        "1. Register on https://platform.qboson.com and get your credentials\n"
        "   (see docs/source/getting_started/installation.md, section\n"
        "   'Kaiwu SDK Configuration & License').\n"
        "2. Export them before running the example:\n"
        '   export USER_ID="<your-user-id>"\n'
        '   export SDK_CODE="<your-sdk-code>"'
    )


if __name__ == "__main__":
    _ensure_license()

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
