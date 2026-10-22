"""CPU Gibbs acceptance tests against the actual enumerated binary joint energy."""
from itertools import product

import pytest
import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def model(dtype=torch.float64):
    return RestrictedBoltzmannMachine(
        2, 2, device="cpu",
        quadratic_coef=torch.tensor([[.7, -.4], [.2, .9]], dtype=dtype),
        linear_bias=torch.tensor([.25, -.3, -.15, .4], dtype=dtype),
    )


def energy_tables(machine):
    """Normalize every actual joint energy; never use the sigmoid formula."""
    states = machine.quadratic_coef.new_tensor(list(product((0., 1.), repeat=4)))
    binary = states[:4, 2:]
    with torch.no_grad():
        joint = torch.softmax(-machine(states), 0).reshape(4, 4)
    return (binary, joint, joint / joint.sum(1, keepdim=True),
            joint.t() / joint.sum(0).unsqueeze(1))


def indices(binary):
    return (binary[:, 0] * 2 + binary[:, 1]).long()


def exact_chain(machine, steps, generator, start, count=4):
    """Use energy-derived conditional marginals and the specified random stream."""
    binary, _, hidden_conditional, visible_conditional = energy_tables(machine)
    visible_first = start != "hidden"
    if start == "random":
        states = torch.bernoulli(machine.quadratic_coef.new_full((count, 2), .5),
                                 generator=generator)
    else:
        states = binary.clone()
    retained = []
    for _ in range(steps):
        if visible_first:
            hidden = torch.bernoulli(hidden_conditional[indices(states)] @ binary,
                                     generator=generator)
            visible = torch.bernoulli(visible_conditional[indices(hidden)] @ binary,
                                      generator=generator)
            states = visible
        else:
            visible = torch.bernoulli(visible_conditional[indices(states)] @ binary,
                                      generator=generator)
            hidden = torch.bernoulli(hidden_conditional[indices(visible)] @ binary,
                                     generator=generator)
            states = hidden
        retained.append(torch.cat((visible, hidden), dim=1))
    return (torch.stack(retained) if retained else
            machine.quadratic_coef.new_empty((0, count, 4)))


def options(machine, start, count=4):
    binary = energy_tables(machine)[0]
    if start == "visible":
        return {"s_visible": binary.clone()}
    if start == "hidden":
        return {"s_hidden": binary.clone()}
    return {"n_sample": count}


@pytest.mark.parametrize("start", ["visible", "hidden", "random"])
@pytest.mark.parametrize("steps,burnin", [(0, 0), (0, 3), (1, 0), (5, 0),
                                         (5, 2), (5, 5), (5, 7)])
def test_actual_transitions_and_exact_burnin_suffix(start, steps, burnin):
    machine = model()
    expected_generator = torch.Generator().manual_seed(239)
    actual_generator = torch.Generator().manual_seed(239)
    expected = exact_chain(machine, steps, expected_generator, start)[burnin:]
    initial = options(machine, start)
    original = {key: value.clone() for key, value in initial.items()
                if isinstance(value, torch.Tensor)}
    actual = machine.gibbs_sample(steps, burnin, generator=actual_generator, **initial)
    torch.testing.assert_close(actual, expected.reshape(-1, 4), atol=0, rtol=0)
    torch.testing.assert_close(actual_generator.get_state(), expected_generator.get_state())
    for key, value in original.items():
        torch.testing.assert_close(initial[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("start", ["visible", "hidden"])
def test_actual_one_sweep_matches_joint_transition_from_energy(start):
    machine = model()
    binary, _, hidden_conditional, visible_conditional = energy_tables(machine)
    repetitions = 12000
    initial = binary.repeat_interleave(repetitions, dim=0)
    actual = machine.gibbs_sample(
        1, generator=torch.Generator().manual_seed(104), **{f"s_{start}": initial})
    state_indices = indices(actual[:, :2]) * 4 + indices(actual[:, 2:])
    for old in range(4):
        counts = torch.bincount(state_indices[old * repetitions:(old + 1) * repetitions],
                                minlength=16).double() / repetitions
        if start == "visible":
            expected = visible_conditional.t() * hidden_conditional[old]
        else:
            expected = visible_conditional[old, :, None] * hidden_conditional
        torch.testing.assert_close(counts, expected.flatten(), atol=.014, rtol=0)
    assert actual.unique(dim=0).shape[0] == 16


@pytest.mark.parametrize("visible_first", [True, False])
def test_energy_derived_block_transition_preserves_joint_distribution(visible_first):
    """An independent exact stationary oracle for the two Gibbs scan orders."""
    _, joint, hidden_conditional, visible_conditional = energy_tables(model())
    transition = torch.empty(16, 16, dtype=torch.float64)
    for old, new in product(range(16), repeat=2):
        old_visible, old_hidden = divmod(old, 4)
        new_visible, new_hidden = divmod(new, 4)
        if visible_first:
            transition[old, new] = (hidden_conditional[old_visible, new_hidden]
                                    * visible_conditional[new_hidden, new_visible])
        else:
            transition[old, new] = (visible_conditional[old_hidden, new_visible]
                                    * hidden_conditional[new_visible, new_hidden])
    torch.testing.assert_close(transition.sum(1), torch.ones(16).double(),
                               atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(joint.flatten() @ transition, joint.flatten(),
                               atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64,
                                    torch.float16, torch.bfloat16])
@pytest.mark.parametrize("start", ["visible", "hidden", "random"])
def test_sampling_uses_current_parameters_dtype_and_detaches_initial_state(dtype, start):
    machine = model().double()
    if dtype == torch.float32:
        machine.float()
    elif dtype == torch.float16:
        machine.half()
    elif dtype == torch.bfloat16:
        machine.bfloat16()
    initial = options(machine, start)
    for key, value in initial.items():
        if isinstance(value, torch.Tensor):
            initial[key] = value.float().requires_grad_()
    output = machine.gibbs_sample(3, generator=torch.Generator().manual_seed(7), **initial)
    assert output.dtype == dtype
    assert output.device == machine.quadratic_coef.device
    assert output.shape == (12, 4)
    assert not output.requires_grad and output.grad_fn is None
    assert torch.all((output == 0) | (output == 1))
    assert all(parameter.grad is None for parameter in machine.parameters())


def test_explicit_generator_preserves_global_rng_and_default_generator_is_repeatable():
    machine = model()
    before = torch.get_rng_state().clone()
    first = machine.gibbs_sample(4, n_sample=8,
                                 generator=torch.Generator().manual_seed(65))
    torch.testing.assert_close(torch.get_rng_state(), before)
    second = machine.gibbs_sample(4, n_sample=8,
                                  generator=torch.Generator().manual_seed(65))
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    default_first = machine.gibbs_sample(4, n_sample=8)
    assert not torch.equal(torch.get_rng_state(), before)
    torch.set_rng_state(before)
    default_second = machine.gibbs_sample(4, n_sample=8)
    torch.testing.assert_close(default_first, default_second, atol=0, rtol=0)


def test_real_objective_backward_and_sgd_match_fixed_sample_moments():
    machine = model()
    optimizer = torch.optim.SGD(machine.parameters(), lr=.07)
    observed = torch.tensor([[0., 1.], [1., 0.], [1., 1.]], dtype=torch.float64)
    # Enumerated posterior expectation supplies the positive joint state's hidden means.
    binary, _, hidden_conditional, _ = energy_tables(machine)
    positive = torch.cat((observed, hidden_conditional[indices(observed)] @ binary), 1)
    negative = machine.gibbs_sample(7, 2, n_sample=6,
                                    generator=torch.Generator().manual_seed(22))
    assert not negative.requires_grad
    before = {name: parameter.detach().clone() for name, parameter in machine.named_parameters()}
    loss = machine.objective(positive, negative)
    positive_interactions = positive[:, :2, None] * positive[:, None, 2:]
    negative_interactions = negative[:, :2, None] * negative[:, None, 2:]
    bias_gradient = negative.mean(0) - positive.mean(0)
    weight_gradient = negative_interactions.mean(0) - positive_interactions.mean(0)
    independent_energy = lambda states: (
        -sum((states[:, i] * before["linear_bias"][i]) for i in range(4))
        -sum((states[:, i] * states[:, 2 + j] * before["quadratic_coef"][i, j])
             for i, j in product(range(2), repeat=2)))
    torch.testing.assert_close(loss, independent_energy(positive).mean()
                               - independent_energy(negative).mean())
    loss.backward()
    torch.testing.assert_close(machine.linear_bias.grad, bias_gradient)
    torch.testing.assert_close(machine.quadratic_coef.grad, weight_gradient)
    optimizer.step()
    torch.testing.assert_close(machine.linear_bias, before["linear_bias"] - .07 * bias_gradient)
    torch.testing.assert_close(machine.quadratic_coef,
                               before["quadratic_coef"] - .07 * weight_gradient)
    # A subsequent chain uses the updated parameters, not cached transition probabilities.
    generator = torch.Generator().manual_seed(13)
    expected = exact_chain(machine, 4, generator, "visible")
    actual_generator = torch.Generator().manual_seed(13)
    actual = machine.gibbs_sample(4, s_visible=binary, generator=actual_generator)
    torch.testing.assert_close(actual, expected.reshape(-1, 4), atol=0, rtol=0)


@pytest.mark.parametrize("keyword,value", [("n_step", -1), ("n_step", 1.5),
    ("n_step", True), ("n_burnin", -1), ("n_burnin", .5), ("n_burnin", False),
    ("n_sample", 0), ("n_sample", -2), ("n_sample", 2.5), ("n_sample", True)])
def test_invalid_counts_fail_without_consuming_rng(keyword, value):
    machine = model()
    generator = torch.Generator().manual_seed(12)
    before = generator.get_state().clone()
    arguments = {"n_step": 3, "n_sample": 4, keyword: value, "generator": generator}
    with pytest.raises(ValueError, match=keyword):
        machine.gibbs_sample(**arguments)
    torch.testing.assert_close(generator.get_state(), before)


@pytest.mark.parametrize("initial", [torch.ones(2), torch.ones(2, 3),
    torch.empty(0, 2), torch.tensor([[0., .5]]), torch.tensor([[float("nan"), 1.]]),
    torch.tensor([[float("inf"), 1.]]), torch.tensor([[-1., 1.]])])
@pytest.mark.parametrize("start", ["visible", "hidden"])
def test_invalid_initial_binary_state_is_rejected(start, initial):
    with pytest.raises(ValueError, match=f"s_{start}"):
        model().gibbs_sample(2, **{f"s_{start}": initial})


def test_ambiguous_initialization_and_chain_count_are_rejected():
    machine = model()
    binary = energy_tables(machine)[0]
    with pytest.raises(ValueError, match="s_visible.*s_hidden"):
        machine.gibbs_sample(2, s_visible=binary, s_hidden=binary)
    with pytest.raises(ValueError, match="n_sample"):
        machine.gibbs_sample(2)
    with pytest.raises(ValueError, match="n_sample"):
        machine.gibbs_sample(2, s_visible=binary, n_sample=5)
    matching = machine.gibbs_sample(2, s_visible=binary, n_sample=4)
    assert matching.shape == (8, 4)


def test_non_tensor_initialization_and_generator_are_rejected():
    with pytest.raises(TypeError, match="s_visible"):
        model().gibbs_sample(2, s_visible=[[0., 1.]])
    with pytest.raises(TypeError, match="generator"):
        model().gibbs_sample(2, n_sample=2, generator=7)
