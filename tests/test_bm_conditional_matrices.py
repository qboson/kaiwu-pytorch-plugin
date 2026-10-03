"""Conditional matrix reuse must preserve values and isolate solver inputs."""
import contextlib
import itertools

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin.full_boltzmann_machine import BoltzmannMachine
import kaiwu.torch_plugin.full_boltzmann_machine as bm_module


@pytest.fixture(autouse=True)
def offline_context(monkeypatch):
    monkeypatch.setattr(bm_module, 'kpp_caller_context', contextlib.nullcontext)


def machine(nodes, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    return BoltzmannMachine(
        nodes, torch.randn(nodes, nodes, dtype=dtype, generator=generator),
        torch.randn(nodes, dtype=dtype, generator=generator), device='cpu',
    )


def legacy_matrix(model, visible):
    with torch.no_grad():
        upper = model.quadratic_coef.triu(1)
        quadratic = upper + upper.T
        width = visible.numel()
        hh = quadratic[width:, width:]
        bias = quadratic[width:, :width] @ visible + model.linear_bias[width:]
        out = torch.zeros((len(bias) + 1, len(bias) + 1), dtype=bias.dtype)
        out[:-1, :-1] = hh / 8
        field = bias / 4 + hh.sum(dim=0) / 8
        out[:-1, -1] = field
        out[-1, :-1] = field
        return out.numpy()


class RecordingSampler:
    def __init__(self, mutate=False):
        self.matrices = []
        self.mutate = mutate

    def solve(self, matrix):
        self.matrices.append(matrix.copy())
        count = 1 + len(self.matrices) % 3
        spins = np.ones((count, len(matrix)), dtype=np.float32)
        # Alternate gauges while preserving the same decoded binary states.
        spins[1::2] *= -1
        if self.mutate:
            matrix[:] = 12345
        return spins


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('nodes', [4, 8, 16])
@pytest.mark.parametrize('width_kind', ['none', 'one', 'last', 'all'])
@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('seed', [2, 17])
def test_all_conditional_matrices_match_legacy(dtype, nodes, width_kind, batch, seed):
    model = machine(nodes, dtype, seed)
    width = {'none': 0, 'one': 1, 'last': nodes - 1, 'all': nodes}[width_kind]
    visible = torch.rand(batch, width, dtype=dtype)
    sampler = RecordingSampler()
    output = model.condition_sample(sampler, visible, dtype=dtype)
    position = 0
    for i, row in enumerate(visible):
        expected = legacy_matrix(model, row)
        np.testing.assert_array_equal(sampler.matrices[i], expected)
        np.testing.assert_array_equal(model._hidden_to_ising_matrix(row), expected)
        count = 1 + (i + 1) % 3
        assert torch.equal(output[position:position + count, :width], row.expand(count, -1))
        assert torch.equal(output[position:position + count, width:], torch.ones(count, nodes - width, dtype=dtype))
        position += count
    assert position == len(output)
    assert output.dtype == dtype


def test_symmetrize_once_per_public_call(monkeypatch):
    model = machine(8, torch.float32, 2)
    calls = []
    original = model.symmetrized_quadratic_coef
    def counted():
        calls.append(1)
        return original()
    monkeypatch.setattr(model, 'symmetrized_quadratic_coef', counted)
    model.condition_sample(RecordingSampler(), torch.rand(7, 5))
    assert len(calls) == 1


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_independent_conditional_energy_differences(dtype):
    model = machine(6, dtype, 12)
    visible = torch.tensor([[0., 1., 0.], [.25, .75, .5]], dtype=dtype)
    hidden = torch.tensor(list(itertools.product([0., 1.], repeat=3)), dtype=dtype)
    sampler = RecordingSampler()
    model.condition_sample(sampler, visible, dtype=dtype)
    # Sum only original upper-triangle pair weights, independent of the conversion.
    for row, matrix in zip(visible, sampler.matrices):
        joint = torch.cat([row.expand(8, -1), hidden], dim=1)
        energies = -(joint * model.linear_bias.detach()).sum(dim=1)
        for i in range(6):
            for j in range(i + 1, 6):
                energies -= model.quadratic_coef[i, j].detach() * joint[:, i] * joint[:, j]
        spins = np.concatenate([2 * hidden.numpy() - 1, np.ones((8, 1))], axis=1)
        converted = -np.einsum('bi,ij,bj->b', spins, matrix, spins)
        np.testing.assert_allclose(energies.numpy() - energies[0].item(), converted - converted[0], atol=2e-6 if dtype == torch.float32 else 1e-12)


def test_mutating_solver_does_not_pollute_next_condition():
    model = machine(8, torch.float64, 4)
    visible = torch.rand(5, 3, dtype=torch.float64)
    original = visible.clone()
    sampler = RecordingSampler(mutate=True)
    model.condition_sample(sampler, visible, dtype=torch.float64)
    for row, matrix in zip(visible, sampler.matrices):
        np.testing.assert_array_equal(matrix, legacy_matrix(model, row))
    assert torch.equal(visible, original)


def test_new_call_recomputes_updated_weights():
    model = machine(8, torch.float32, 4)
    visible = torch.rand(3, 4)
    first = RecordingSampler()
    model.condition_sample(first, visible)
    with torch.no_grad():
        model.quadratic_coef.add_(0.25)
        model.linear_bias.add_(0.5)
    second = RecordingSampler()
    model.condition_sample(second, visible)
    assert not np.array_equal(first.matrices[0], second.matrices[0])
    for row, matrix in zip(visible, second.matrices):
        np.testing.assert_array_equal(matrix, legacy_matrix(model, row))


@pytest.mark.parametrize('weight_dtype,bias_dtype', [(torch.float32, torch.float64), (torch.float64, torch.float32), (torch.float32, torch.float16)])
def test_mixed_parameter_dtypes_preserve_promoted_matrix(weight_dtype, bias_dtype):
    model = machine(5, weight_dtype, 4)
    model.linear_bias = torch.nn.Parameter(model.linear_bias.detach().to(bias_dtype))
    visible = torch.rand(3, 2, dtype=weight_dtype)
    sampler = RecordingSampler()
    model.condition_sample(sampler, visible)
    for row, matrix in zip(visible, sampler.matrices):
        np.testing.assert_array_equal(matrix, legacy_matrix(model, row))


def test_condition_input_gradient_is_preserved():
    model = machine(5, torch.float32, 2)
    visible = torch.rand(3, 2, requires_grad=True)
    sampler = RecordingSampler()
    output = model.condition_sample(sampler, visible)
    output.sum().backward()
    assert torch.equal(visible.grad, torch.tensor([[2., 2.], [3., 3.], [1., 1.]]))
    assert all(p.grad is None for p in model.parameters())


def test_noncontiguous_visible_inputs():
    model = machine(6, torch.float32, 7)
    visible = torch.rand(4, 6)[:, ::2]
    assert not visible.is_contiguous()
    sampler = RecordingSampler()
    model.condition_sample(sampler, visible)
    for row, matrix in zip(visible, sampler.matrices):
        np.testing.assert_array_equal(matrix, legacy_matrix(model, row))


def test_failure_does_not_cache_terms_between_calls():
    model = machine(6, torch.float32, 7)
    visible = torch.rand(4, 3)
    class FailingSampler:
        def solve(self, matrix):
            matrix[:] = 999
            raise RuntimeError('controlled solver failure')
    with pytest.raises(RuntimeError, match='controlled solver failure'):
        model.condition_sample(FailingSampler(), visible)
    with torch.no_grad():
        model.linear_bias.add_(1)
    sampler = RecordingSampler()
    model.condition_sample(sampler, visible)
    for row, matrix in zip(visible, sampler.matrices):
        np.testing.assert_array_equal(matrix, legacy_matrix(model, row))


def test_empty_batch_does_not_prepare_terms(monkeypatch):
    model = machine(6, torch.float32, 7)
    def unexpected():
        pytest.fail('prepared couplings without any conditions')
    monkeypatch.setattr(model, 'symmetrized_quadratic_coef', unexpected)
    with pytest.raises((RuntimeError, ValueError)):
        model.condition_sample(RecordingSampler(), torch.empty(0, 3))


@pytest.mark.parametrize('autocast_dtype', [torch.float16, torch.bfloat16])
def test_autocast_preserves_legacy_matrix_with_float32_bias(autocast_dtype):
    model = machine(6, torch.float32, 7)
    visible = torch.rand(4, 3)
    sampler = RecordingSampler()
    with torch.autocast('cpu', dtype=autocast_dtype):
        model.condition_sample(sampler, visible)
        for row, matrix in zip(visible, sampler.matrices):
            np.testing.assert_array_equal(matrix, legacy_matrix(model, row))


def test_autocast_preserves_promoted_field_dtype():
    model = machine(6, torch.float32, 7)
    model.linear_bias = torch.nn.Parameter(model.linear_bias.detach().half())
    visible = torch.rand(4, 3)
    sampler = RecordingSampler()
    with torch.autocast('cpu', dtype=torch.float16):
        model.condition_sample(sampler, visible)
        for row, matrix in zip(visible, sampler.matrices):
            reference = legacy_matrix(model, row)
            assert matrix.dtype == reference.dtype == np.float16
            np.testing.assert_array_equal(matrix, reference)
