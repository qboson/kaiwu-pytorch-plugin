"""Local CPU/CUDA parity checks for EP matrices, exact states and gradients."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ep_reference_tests", ROOT / "tests/test_equilibrium_propagation.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("factory", [reference.linear_model, reference.conv_model])
def test_cuda_matches_cpu(factory):
    assert torch.cuda.is_available(), "GPU verification must run on a CUDA device"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cpu = factory()
    gpu = copy.deepcopy(cpu).cuda()
    x = torch.linspace(-0.9, 0.8, 2 * cpu.layer_sizes[0], dtype=torch.float64).reshape(2, -1)
    y = torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64)
    sampler = reference.EnumerateSampler()
    states = []
    for model, inputs, targets in [(cpu, x, y), (gpu, x.cuda(), y.cuda())]:
        free = model.equilibrate(inputs, sampler)
        nudged = model.equilibrate(inputs, sampler, targets, beta=1.7)
        loss = model.objective(inputs, free, nudged, beta=1.7)
        loss.backward()
        assert all(t.device == inputs.device for t in free + nudged)
        assert all(p.grad.device == inputs.device for p in model.parameters())
        states.append((free, nudged, loss))
    for beta in (0., 1.7):
        np.testing.assert_allclose(cpu.ising_matrix(x[0], y[0], beta),
                                   gpu.ising_matrix(x[0].cuda(), y[0].cuda(), beta), rtol=1e-10, atol=1e-10)
    for a, b in zip(states[0][0] + states[0][1], states[1][0] + states[1][1]):
        torch.testing.assert_close(a, b.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(states[0][2], states[1][2].cpu(), rtol=1e-10, atol=1e-10)
    for a, b in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(a.grad, b.grad.cpu(), rtol=1e-10, atol=1e-10)
    torch.optim.SGD(cpu.parameters(), lr=0.05).step()
    torch.optim.SGD(gpu.parameters(), lr=0.05).step()
    for a, b in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(a, b.cpu(), rtol=1e-10, atol=1e-10)
