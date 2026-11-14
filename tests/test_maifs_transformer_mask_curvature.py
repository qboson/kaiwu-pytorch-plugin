"""Mask curvature for ordinary Transformer classifiers uses real attention."""

import math
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, TensorDataset

from kaiwu.torch_plugin import FeatureSelectionWrapper


class TransformerClassifier(nn.Module):
    """Small standard encoder, with smooth activations for finite differences."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=4, nhead=2, dim_feedforward=8, dropout=0.0,
            activation="gelu", batch_first=True,
        )
        self.head = nn.Linear(4, 3)

    def forward(self, values):
        """Pool token features into three class logits."""
        return self.head(self.encoder(values).mean(dim=1))


def _backend_flags():
    return (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
    )


def _fixture():
    generator = torch.Generator().manual_seed(4402)
    values = torch.randn(7, 3, 4, generator=generator, dtype=torch.float64) * 0.2
    labels = torch.tensor([0, 1, 2, 1, 0, 2, 1])
    with torch.random.fork_rng():
        torch.manual_seed(1948)
        model = TransformerClassifier().double()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(values), labels).backward()
        optimizer.step()
    wrapper = FeatureSelectionWrapper(model, 4).double()
    with torch.no_grad():
        wrapper.mask.copy_(torch.tensor([1.0, 0.0, 1.0, 1.0]))
    return wrapper, values, labels


def _explicit_reference(model, values):
    """Evaluate the encoder with explicit QK softmax, without SDPA or autograd."""
    layer = model.encoder
    attention = layer.self_attn
    query, key, value = F.linear(
        values, attention.in_proj_weight, attention.in_proj_bias,
    ).chunk(3, dim=-1)
    batch, tokens, width = values.shape
    head_width = width // attention.num_heads

    def split_heads(tensor):
        return tensor.reshape(batch, tokens, attention.num_heads, head_width).transpose(1, 2)

    query, key, value = (split_heads(tensor) for tensor in (query, key, value))
    probabilities = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(head_width), dim=-1)
    attended = (probabilities @ value).transpose(1, 2).reshape(batch, tokens, width)
    projected = F.linear(attended, attention.out_proj.weight, attention.out_proj.bias)
    normalized = F.layer_norm(
        values + projected, layer.norm1.normalized_shape,
        layer.norm1.weight, layer.norm1.bias, layer.norm1.eps,
    )
    feedforward = F.linear(
        F.gelu(F.linear(normalized, layer.linear1.weight, layer.linear1.bias)),
        layer.linear2.weight, layer.linear2.bias,
    )
    encoded = F.layer_norm(
        normalized + feedforward, layer.norm2.normalized_shape,
        layer.norm2.weight, layer.norm2.bias, layer.norm2.eps,
    )
    return F.linear(encoded.mean(dim=1), model.head.weight, model.head.bias)


@pytest.mark.parametrize("mode", ["full", "diagonal"])
def test_trained_transformer_mask_curvature_matches_explicit_attention(mode):
    """Numerical loss differences independently check both derivative modes."""
    wrapper, values, labels = _fixture()
    original_flags = _backend_flags()
    original_requires_grad = [
        parameter.requires_grad for parameter in wrapper.parameters()
    ]
    original_gradients = [parameter.grad for parameter in wrapper.parameters()]
    gradient_values = [None if grad is None else grad.clone() for grad in original_gradients]
    original_state = {
        key: value.clone() for key, value in wrapper.state_dict().items()
    }
    loader = DataLoader(TensorDataset(values, labels), batch_size=3)
    gradient, hessian = wrapper.compute_mask_derivatives(loader, nn.CrossEntropyLoss(), mode)
    assert _backend_flags() == original_flags
    assert wrapper.training and wrapper.model.training
    assert [parameter.requires_grad for parameter in wrapper.parameters()] == original_requires_grad
    for parameter, original, expected in zip(
        wrapper.parameters(), original_gradients, gradient_values
    ):
        assert parameter.grad is original
        if expected is not None:
            torch.testing.assert_close(parameter.grad, expected)
    for key, expected in original_state.items():
        torch.testing.assert_close(wrapper.state_dict()[key], expected, rtol=0, atol=0)

    wrapper.eval()
    mask = wrapper.mask.clone()
    with torch.no_grad():
        actual = wrapper.model(values * mask)
        reference = _explicit_reference(wrapper.model, values * mask)
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)

    def loss_value(candidate):
        with torch.no_grad():
            logits = _explicit_reference(wrapper.model, values * candidate)
            return F.cross_entropy(logits, labels).item()

    step = 1e-3
    directions = torch.eye(4, dtype=torch.float64) * step
    expected_gradient = np.array([
        (loss_value(mask + delta) - loss_value(mask - delta)) / (2 * step)
        for delta in directions
    ])
    expected_hessian = np.array([
        [
            (loss_value(mask + left + right) - loss_value(mask + left - right)
             - loss_value(mask - left + right) + loss_value(mask - left - right)) / (4 * step ** 2)
            for right in directions
        ]
        for left in directions
    ])
    if mode == "diagonal":
        expected_hessian = np.diag(np.diag(expected_hessian))
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-4, atol=2e-6)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=5e-4, atol=3e-6)


def test_backend_choice_restores_on_success_and_loss_failure():
    """Derivative scoping restores an explicit caller choice and model state."""
    wrapper, values, labels = _fixture()
    original_requires_grad = [
        parameter.requires_grad for parameter in wrapper.parameters()
    ]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        flags = _backend_flags()
        wrapper.compute_mask_derivatives([(values, labels)], nn.CrossEntropyLoss())
        assert _backend_flags() == flags

        def fail(_prediction, _target):
            raise RuntimeError("loss failed")

        with pytest.raises(RuntimeError, match="loss failed"):
            wrapper.compute_mask_derivatives([(values, labels)], fail)
        assert _backend_flags() == flags
    assert [parameter.requires_grad for parameter in wrapper.parameters()] == original_requires_grad
    assert wrapper.training and wrapper.model.training


def test_automatic_mask_refresh_keeps_normal_training_backend():
    """Only the frozen-model curvature pass overrides attention backend flags."""
    wrapper, values, labels = _fixture()
    wrapper.mask_update_epochs = 1
    optimizer = torch.optim.Adam(wrapper.model.parameters(), lr=0.01)
    loader = DataLoader(TensorDataset(values, labels), batch_size=4)
    flags = _backend_flags()
    observed = []

    def record_backend(model, _args):
        trainable = any(parameter.requires_grad for parameter in model.parameters())
        observed.append((trainable, _backend_flags()))

    hook = wrapper.model.register_forward_pre_hook(record_backend)
    try:
        loss = wrapper.fit_weights(loader, nn.CrossEntropyLoss(), optimizer, train_epochs=2)
    finally:
        hook.remove()
    assert np.isfinite(loss)
    assert _backend_flags() == flags
    training_flags = [choice for trainable, choice in observed if trainable]
    derivative_flags = [choice for trainable, choice in observed if not trainable]
    assert training_flags and all(choice == flags for choice in training_flags)
    assert len(derivative_flags) == 2
    assert all(choice == (False, True, False, False) for choice in derivative_flags)



def test_concurrent_derivatives_serialize_backend_contexts():
    """Two library calls cannot restore process-global flags out of order."""
    first, values, labels = _fixture()
    second, _, _ = _fixture()
    first_entered, first_release = Event(), Event()
    second_started, second_entered, second_inspect = Event(), Event(), Event()
    second_flags = []
    flags = _backend_flags()

    def pause_first(_model, _inputs):
        first_entered.set()
        if not first_release.wait(10):
            raise RuntimeError("first derivative was not released")

    def inspect_second(_model, _inputs):
        second_entered.set()
        if not second_inspect.wait(10):
            raise RuntimeError("second derivative was not released")
        second_flags.append(_backend_flags())

    def run_second():
        second_started.set()
        return second.compute_mask_derivatives([(values, labels)], nn.CrossEntropyLoss())

    first_hook = first.model.register_forward_pre_hook(pause_first)
    second_hook = second.model.register_forward_pre_hook(inspect_second)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_result = pool.submit(
                first.compute_mask_derivatives, [(values, labels)], nn.CrossEntropyLoss(),
            )
            try:
                assert first_entered.wait(10)
                second_result = pool.submit(run_second)
                assert second_started.wait(10)
                premature_entry = second_entered.wait(0.2)
                first_release.set()
                first_gradient, first_hessian = first_result.result(timeout=10)
                assert second_entered.wait(10)
                second_inspect.set()
                second_gradient, second_hessian = second_result.result(timeout=10)
            finally:
                first_release.set()
                second_inspect.set()
    finally:
        first_hook.remove()
        second_hook.remove()
    assert not premature_entry
    assert second_flags == [(False, True, False, False)]
    assert _backend_flags() == flags
    np.testing.assert_allclose(first_gradient, second_gradient, rtol=0, atol=0)
    np.testing.assert_allclose(first_hessian, second_hessian, rtol=0, atol=0)
