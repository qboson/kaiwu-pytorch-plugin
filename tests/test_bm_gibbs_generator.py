"""Full BM Gibbs draws can use a caller-owned random stream."""
import pytest
import torch

from kaiwu.torch_plugin.full_boltzmann_machine import BoltzmannMachine


def machine(device='cpu'):
    q = torch.arange(25, dtype=torch.float32).reshape(5, 5) / 25 - .4
    bias = torch.tensor([.1, -.2, .3, -.4, .5])
    return BoltzmannMachine(5, q, bias, device=device)


def condition(kind):
    if kind == 'none':
        return None
    if kind == 'partial':
        return torch.tensor([[0., 1.], [1., 0.], [1., 1.], [0., 0.]])
    if kind == 'empty_width':
        return torch.empty(4, 0)
    return torch.tensor([[0., 1., 0., 1., 0.]]).expand(4, -1)


def reference_chain(model, steps, visible, generator=None):
    """Independent legacy draw order and full single-site update reference."""
    with torch.no_grad():
        initial = torch.full((4, model.num_nodes), .5, device=model.device)
        if generator is None:
            state = torch.bernoulli(initial)
        else:
            state = torch.bernoulli(initial, generator=generator)
        width = 0 if visible is None else visible.shape[1]
        if visible is not None:
            state[:, :width] = visible
        upper = torch.triu(model.quadratic_coef, 1)
        weights = upper + upper.T
        for _ in range(steps):
            if generator is None:
                order = torch.randperm(model.num_nodes, device=model.device)
            else:
                order = torch.randperm(model.num_nodes, device=model.device, generator=generator)
            for unit in order:
                if unit < width:
                    continue
                probability = torch.sigmoid(state @ weights[:, unit] + model.linear_bias[unit])
                if generator is None:
                    noise = torch.rand_like(probability)
                else:
                    noise = torch.rand(probability.shape, dtype=probability.dtype,
                                       device=probability.device, generator=generator)
                state[:, unit] = (probability > noise).float()
        return state


KINDS = ['none', 'partial', 'empty_width', 'all']


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('steps', [0, 3, 9])
def test_private_stream_matches_reference_draws_and_state(kind, steps):
    model = machine()
    visible = condition(kind)
    actual_generator = torch.Generator().manual_seed(47)
    expected_generator = torch.Generator().manual_seed(47)
    global_before = torch.random.get_rng_state().clone()
    actual = model.gibbs_sample(steps, visible, 4, generator=actual_generator)
    expected = reference_chain(model, steps, visible, expected_generator)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_generator.get_state(), expected_generator.get_state())
    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert actual.shape == (4, 5) and not actual.requires_grad
    if visible is not None:
        assert torch.equal(actual[:, :visible.shape[1]], visible)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('steps', [0, 3, 9])
def test_default_mode_preserves_legacy_global_draw_sequence(kind, steps):
    model = machine()
    visible = condition(kind)
    torch.manual_seed(19)
    actual = model.gibbs_sample(steps, visible, 4)
    actual_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(19)
    expected = reference_chain(model, steps, visible)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_rng, torch.random.get_rng_state())
    torch.manual_seed(19)
    explicit_none = model.gibbs_sample(steps, visible, 4, generator=None)
    assert torch.equal(actual, explicit_none)
    assert torch.equal(actual_rng, torch.random.get_rng_state())


def test_private_seed_is_independent_of_other_modules_random_draws():
    model = machine()
    first = model.gibbs_sample(12, condition('partial'), generator=torch.Generator().manual_seed(29))
    torch.rand(500)
    second = model.gibbs_sample(12, condition('partial'), generator=torch.Generator().manual_seed(29))
    assert torch.equal(first, second)


def test_generator_state_can_continue_and_resume_random_stream():
    model = machine()
    generator = torch.Generator().manual_seed(29)
    before = generator.get_state().clone()
    model.gibbs_sample(3, num_sample=4, generator=generator)
    saved = generator.get_state().clone()
    assert not torch.equal(before, saved)
    next_samples = model.gibbs_sample(3, num_sample=4, generator=generator)
    resumed = torch.Generator()
    resumed.set_state(saved)
    actual = model.gibbs_sample(3, num_sample=4, generator=resumed)
    assert torch.equal(next_samples, actual)
    assert torch.equal(generator.get_state(), resumed.get_state())


@pytest.mark.parametrize('bad', [True, 1, 'cpu', torch.tensor(3)])
def test_invalid_generator_type_fails_before_rng_or_chain_initialization(monkeypatch, bad):
    model = machine()
    before = torch.random.get_rng_state().clone()
    def unexpected(*args, **kwargs):
        pytest.fail('chain initialized before generator validation')
    monkeypatch.setattr(torch, 'bernoulli', unexpected)
    with pytest.raises(TypeError, match='generator'):
        model.gibbs_sample(3, num_sample=4, generator=bad)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize('device', ['meta', 'cuda:0', 'mps'])
def test_wrong_generator_device_fails_before_any_rng_draw(monkeypatch, device):
    model = machine()
    # No accelerator is needed: validation must reject this before allocation.
    model.device = torch.device(device)
    generator = torch.Generator().manual_seed(29)
    before, global_before = generator.get_state().clone(), torch.random.get_rng_state().clone()
    def unexpected(*args, **kwargs):
        pytest.fail('chain initialized before generator device validation')
    monkeypatch.setattr(torch, 'bernoulli', unexpected)
    with pytest.raises(ValueError, match='generator.*device'):
        model.gibbs_sample(3, num_sample=4, generator=generator)
    assert torch.equal(generator.get_state(), before)
    assert torch.equal(torch.random.get_rng_state(), global_before)


@pytest.mark.parametrize('device', ['cpu', torch.device('cpu'), 'cpu:0'])
def test_cpu_device_aliases_accept_cpu_generator(device):
    model = machine(device)
    actual = model.gibbs_sample(3, num_sample=4, generator=torch.Generator().manual_seed(29))
    expected = reference_chain(model, 3, None, torch.Generator().manual_seed(29))
    assert torch.equal(actual, expected)


def test_missing_initialization_keeps_both_rng_streams_unchanged():
    model = machine()
    generator = torch.Generator().manual_seed(29)
    before, global_before = generator.get_state().clone(), torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match='Either'):
        model.gibbs_sample(generator=generator)
    assert torch.equal(generator.get_state(), before)
    assert torch.equal(torch.random.get_rng_state(), global_before)


def test_inputs_parameters_modes_and_existing_gradients_are_preserved():
    model = machine().eval()
    visible = condition('partial').requires_grad_()
    original = visible.detach().clone()
    params = list(model.parameters())
    values = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.ones_like(p)
    with torch.enable_grad():
        output = model.gibbs_sample(3, visible, generator=torch.Generator().manual_seed(29))
        assert torch.is_grad_enabled()
    assert not output.requires_grad and visible.grad is None
    assert torch.equal(visible, original) and not model.training
    for p, value in zip(params, values):
        assert torch.equal(p, value) and p.requires_grad
        assert torch.equal(p.grad, torch.ones_like(p))
