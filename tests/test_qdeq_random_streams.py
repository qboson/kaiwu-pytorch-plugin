"""Caller-owned QDEQ randomness, exercised through real CPU solves."""

from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig
from kaiwu.torch_plugin.qdeq import _search_directions


class ExactSampler:
    """Enumerate tiny Ising problems without licenses or cloud submissions."""

    def __init__(self):
        self.matrices = []

    def solve(self, matrix):
        """Maximize the submitted SDK spin objective independently."""
        self.matrices.append(matrix.copy())
        spins = np.array(list(product((-1., 1.), repeat=matrix.shape[0])))
        scores = np.einsum("bi,ij,bj->b", spins, matrix, spins)
        return spins[[int(scores.argmax())]]


class Contraction(torch.nn.Module):
    """Deterministic layer with an independently known equilibrium."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))

    def forward(self, state, inputs):
        """Return a contractive affine map."""
        return self.scale * state + inputs


@pytest.fixture(autouse=True)
def preserve_global_rng():
    """Keep assertions and unrelated tests independent of this suite's seeds."""
    with torch.random.fork_rng(devices=[]):
        yield


def model(generator=None, regularize=False, **kwargs):
    """Build a small model whose solver needs random padding directions."""
    config = QDEQConfig(f_thres=12, f_eps=1e-6, b_thres=40, b_eps=1e-8,
                       num_basis=3, coeff_bits=2,
                       jac_loss_vecs=2 if regularize else 0)
    return QDEQ(Contraction(), ExactSampler(), config, generator=generator, **kwargs)


def inputs():
    """Fixed data requiring no random setup."""
    return torch.tensor([[0.2, -0.4], [0.1, 0.3]], dtype=torch.float64)


@pytest.mark.parametrize("regularize", [False, True])
def test_public_forward_preserves_global_rng_and_consumes_owned_stream(regularize):
    generator = torch.Generator().manual_seed(23)
    deq = model(generator, regularize)
    global_state, owned_state = torch.get_rng_state(), generator.get_state()
    result = deq(inputs())
    assert torch.equal(torch.get_rng_state(), global_state)
    assert not torch.equal(generator.get_state(), owned_state)
    assert deq.sampler.matrices
    assert torch.isfinite(result).all()
    if regularize:
        assert torch.isfinite(deq.last_jac_loss)


@pytest.mark.parametrize("regularize", [False, True])
def test_seeded_public_solve_and_gradients_ignore_unrelated_random_draws(regularize):
    def run(noise):
        torch.rand(noise)
        deq = model(torch.Generator().manual_seed(19), regularize)
        data = inputs().requires_grad_()
        output = deq(data)
        penalty = deq.last_jac_loss if regularize else 0
        (output.square().sum() + penalty).backward()
        return output.detach(), data.grad, deq.layer.scale.grad, deq.sampler.matrices

    first, second = run(5), run(173)
    for left, right in zip(first[:3], second[:3]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert len(first[3]) == len(second[3]) > 0
    for left, right in zip(first[3], second[3]):
        np.testing.assert_array_equal(left, right)


def test_checkpointed_generator_state_replays_the_next_training_call(tmp_path):
    deq = model(torch.Generator().manual_seed(31), regularize=True)
    deq(inputs())
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": deq.state_dict(), "rng": deq.generator.get_state()}, path)
    expected = deq(inputs())
    expected_penalty = deq.last_jac_loss.detach().clone()
    final_state = deq.generator.get_state()

    restored = model(torch.Generator(), regularize=True)
    checkpoint = torch.load(path, weights_only=True)
    restored.load_state_dict(checkpoint["model"])
    restored.generator.set_state(checkpoint["rng"])
    torch.rand(97)
    actual = restored(inputs())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(restored.last_jac_loss, expected_penalty, rtol=0, atol=0)
    assert torch.equal(restored.generator.get_state(), final_state)


def test_different_owned_seeds_change_the_submitted_search_problem():
    first = model(torch.Generator().manual_seed(1))
    second = model(torch.Generator().manual_seed(2))
    first(inputs())
    second(inputs())
    assert not np.array_equal(first.sampler.matrices[0], second.sampler.matrices[0])


def test_default_global_rng_behavior_is_unchanged():
    residual = inputs()
    torch.manual_seed(71)
    expected = [-residual, *(torch.randint_like(residual, 0, 2) * 2 - 1
                             for _ in range(3))]
    expected = torch.stack([value / (value.norm() + 1e-9) for value in expected])
    state = torch.get_rng_state()
    torch.manual_seed(71)
    actual = _search_directions(residual, [], [], [], 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), state)


def test_none_generator_keeps_existing_positional_constructor_working():
    torch.manual_seed(9)
    first = QDEQ(Contraction(), ExactSampler(), QDEQConfig(num_basis=2, coeff_bits=2))
    output = first(inputs())
    torch.manual_seed(9)
    second = QDEQ(Contraction(), ExactSampler(), first.config, generator=None)
    torch.testing.assert_close(second(inputs()), output, rtol=0, atol=0)


def test_no_padding_does_not_consume_the_owned_stream():
    generator = torch.Generator().manual_seed(12)
    state = generator.get_state()
    history = [inputs() * 2]
    directions = _search_directions(inputs(), history, [], [], 2, generator=generator)
    assert torch.equal(state, generator.get_state())
    torch.testing.assert_close(directions.norm(dim=(1, 2)), torch.ones(2, dtype=torch.float64))


def test_jacobian_probes_match_an_independent_scalar_gradient_oracle():
    generator = torch.Generator().manual_seed(53)
    reference = torch.Generator().manual_seed(53)
    scale = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    state = inputs().requires_grad_()
    probes = torch.stack([torch.randn(state.shape, dtype=state.dtype, generator=reference)
                          for _ in range(3)])
    expected_factor = probes.square().sum() / (3 * state.numel())
    global_state = torch.get_rng_state()
    penalty = QDEQ.jacobian_loss(scale * state, state, vecs=3, generator=generator)
    torch.testing.assert_close(penalty, scale.square() * expected_factor)
    penalty.backward()
    torch.testing.assert_close(scale.grad, 2 * scale.detach() * expected_factor)
    assert torch.equal(global_state, torch.get_rng_state())
    assert torch.equal(generator.get_state(), reference.get_state())


def test_owned_stream_survives_a_sampler_failure_without_touching_global_rng():
    class FailureSampler:
        """Fail after QDEQ has constructed its random search directions."""

        def solve(self, _matrix):
            raise RuntimeError("solver failed")

    generator = torch.Generator().manual_seed(7)
    deq = model(generator)
    deq.sampler = FailureSampler()
    state, global_state = generator.get_state(), torch.get_rng_state()
    with pytest.raises(RuntimeError, match="solver failed"):
        deq(inputs())
    assert torch.equal(torch.get_rng_state(), global_state)
    assert not torch.equal(generator.get_state(), state)
    generator.set_state(state)
    deq.sampler = ExactSampler()
    torch.testing.assert_close(deq(inputs()), model(torch.Generator().manual_seed(7))(inputs()),
                               rtol=0, atol=0)


def test_owned_stream_solve_preserves_the_analytic_equilibrium_and_implicit_gradient():
    deq = model(torch.Generator().manual_seed(11))
    data = inputs().requires_grad_()
    output = deq(data)
    output.sum().backward()
    scale = deq.layer.scale.detach()
    torch.testing.assert_close(output, data.detach() / (1 - scale), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(data.grad, torch.ones_like(data) / (1 - scale),
                               rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(deq.layer.scale.grad, data.detach().sum() / (1 - scale).square(),
                               rtol=1e-4, atol=1e-5)


def test_offline_replay_example_runs_outside_the_repository(tmp_path):
    script = Path(__file__).resolve().parents[1] / "example/qdeq/simple/reproducible_cpu.py"
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                            capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(result.stdout) == {
        "output_replayed": True, "jacobian_probes_replayed": True,
        "global_rng_preserved": True, "finite_gradient": True,
    }
