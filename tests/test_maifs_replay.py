"""Reject exhausted training streams before changing any training state."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kaiwu.torch_plugin.maifs.plugin import FeatureSelectionWrapper


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_equal(first, second)
    else:
        assert left == right


@pytest.fixture
def training_setup():
    with torch.random.fork_rng():
        torch.manual_seed(3)
        wrapper = FeatureSelectionWrapper(nn.Sequential(nn.Linear(2, 1), nn.Dropout(0)), 2)
    optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.1, momentum=0.9)
    # Seed optimizer state and existing gradients, then set mixed module modes.
    for parameter in wrapper.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    wrapper.eval()
    wrapper.model[0].train()
    batches = [(torch.tensor([[1., 2.], [2., 1.]]), torch.tensor([[1.], [2.]]))]
    return wrapper, optimizer, batches


@pytest.mark.parametrize("epochs,interval,completed", [
    (2, None, 0), (1, 1, 0), (1, 3, 2), (2, 5, 3), (2, 5, 0),
])
@pytest.mark.parametrize("stream_type", ["iterator", "generator"])
def test_required_replay_fails_before_state_changes(training_setup, epochs, interval,
                                                   completed, stream_type):
    wrapper, optimizer, batches = training_setup
    wrapper.mask_update_epochs = interval
    wrapper._trained_epochs = completed
    state = deepcopy(wrapper.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    modes = [module.training for module in wrapper.modules()]
    gradients = [parameter.grad.clone() for parameter in wrapper.parameters()]
    stream = iter(batches) if stream_type == "iterator" else (batch for batch in batches)
    with pytest.raises(ValueError, match="re-iterable"):
        wrapper.fit_weights(stream, nn.MSELoss(), optimizer, epochs)
    assert_equal(wrapper.state_dict(), state)
    assert_equal(optimizer.state_dict(), optimizer_state)
    assert wrapper._trained_epochs == completed
    assert [module.training for module in wrapper.modules()] == modes
    for parameter, gradient in zip(wrapper.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    assert next(stream) is batches[0]


@pytest.mark.parametrize("interval,completed", [(None, 0), (0, 0), (-1, 0), (3, 0), (3, 1)])
def test_single_pass_stream_remains_supported(training_setup, interval, completed):
    wrapper, optimizer, batches = training_setup
    wrapper.mask_update_epochs = interval
    wrapper._trained_epochs = completed
    loss = wrapper.fit_weights(iter(batches), nn.MSELoss(), optimizer, 1)
    assert isinstance(loss, float)
    assert wrapper._trained_epochs == completed + 1


@pytest.mark.parametrize("loader_type", ["list", "dataloader"])
def test_reiterable_sources_support_epochs_and_mask_updates(training_setup, loader_type):
    wrapper, optimizer, batches = training_setup
    wrapper.mask_update_epochs = 2
    source = (batches if loader_type == "list" else
              DataLoader(TensorDataset(*batches[0]), batch_size=1))
    loss = wrapper.fit_weights(source, nn.MSELoss(), optimizer, 3)
    assert isinstance(loss, float)
    assert wrapper._trained_epochs == 3
    assert wrapper.mask.shape == (2,)
