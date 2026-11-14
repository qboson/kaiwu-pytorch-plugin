"""Real tiny ESM coverage for safe per-call context reuse during inference."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('omegaconf')
pytest.importorskip('transformers')
pytest.importorskip('kaiwu')

from transformers import EsmConfig
from torch.nn.modules import module as module_hooks

# Examples are outside the installed src-layout package. Limit their path to imports.
with pytest.MonkeyPatch.context() as import_context:
    import_context.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from example.qdiffusion.dplm.models.backbone import DPLMBackbone
    from example.qdiffusion.dplm.models.bm import BMConditionedEnergyModel
    from example.qdiffusion.dplm.models.common import DPLMFeatureEncoder, masked_mean_pool
    from example.qdiffusion.dplm.models.esm_patch import _EsmForDPLM


@pytest.fixture
def scorer(monkeypatch):
    tokenizer = SimpleNamespace(mask_token_id=3, pad_token_id=0, cls_token_id=1,
                                eos_token_id=2, _token_to_id={'X': 4})
    monkeypatch.setattr('example.qdiffusion.dplm.models.esm_patch.AutoTokenizer.from_pretrained',
                        lambda *args, **kwargs: tokenizer)
    config = EsmConfig(vocab_size=16, pad_token_id=0, mask_token_id=3, hidden_size=8,
                       num_hidden_layers=2, num_attention_heads=2, intermediate_size=16,
                       position_embedding_type='rotary', token_dropout=False)
    network = _EsmForDPLM(config, dropout=.2)
    encoder = DPLMFeatureEncoder(DPLMBackbone(net=network))
    return BMConditionedEnergyModel(encoder, 4, 2, sampler=object()).eval()


def inputs():
    noisy = torch.tensor([[1, 3, 6, 7, 2, 0], [1, 8, 3, 6, 5, 2]])
    noisy = noisy[:, None, :].expand(2, 4, 6).reshape(8, 6)
    candidate = noisy.clone()
    candidate[:, 1] = torch.arange(8) + 5
    return noisy, candidate, candidate.ne(0)


def original_features(model, noisy, candidate, mask):
    noisy_hidden = model.encoder.encode_tokens(noisy, attention_mask=mask)
    candidate_hidden = model.encoder.encode_tokens(candidate, attention_mask=mask)
    return torch.cat((
        masked_mean_pool(noisy_hidden, mask),
        masked_mean_pool(candidate_hidden, mask),
    ), dim=-1)


def record_rows(monkeypatch):
    rows = []
    encode = DPLMFeatureEncoder.encode_tokens

    def record(self, tokens, attention_mask=None):
        rows.append(tokens.size(0))
        return encode(self, tokens, attention_mask=attention_mask)

    monkeypatch.setattr(DPLMFeatureEncoder, 'encode_tokens', record)
    return rows


def test_real_esm_reuses_repeated_context_and_preserves_rng(scorer, monkeypatch):
    noisy, candidate, mask = inputs()
    rows = record_rows(monkeypatch)
    with torch.no_grad():
        expected = original_features(scorer, noisy, candidate, mask)
        rows.clear()
        before = torch.random.get_rng_state().clone()
        actual = scorer.build_conditioned_features(noisy, candidate, mask)
    torch.testing.assert_close(actual, expected)
    assert rows == [2, 8]
    assert torch.equal(before, torch.random.get_rng_state())


def test_mask_participates_in_context_identity(scorer, monkeypatch):
    noisy, candidate, mask = inputs()
    mask[0, 2] = False
    rows = record_rows(monkeypatch)
    with torch.no_grad():
        expected = original_features(scorer, noisy, candidate, mask)
        rows.clear()
        actual = scorer.build_conditioned_features(noisy, candidate, mask)
    torch.testing.assert_close(actual, expected)
    assert rows == [3, 8]


@pytest.mark.parametrize('guard', [
    'scorer_train', 'child_train', 'forward_hook', 'pre_hook', 'global_hook',
    'global_rng_hook', 'global_pre_hook', 'child_forward_override',
    'encoder_override', 'custom_encoder', 'custom_backbone', 'custom_net', 'compiled', 'float_mask',
])
def test_custom_or_stochastic_paths_keep_independent_rows(scorer, monkeypatch, guard):
    noisy, candidate, mask = inputs()
    network = scorer.encoder.backbone.net
    handle = None
    if guard == 'scorer_train':
        scorer.training = True
    elif guard == 'child_train':
        network.esm.encoder.layer[0].train()
    elif guard in ('forward_hook', 'global_hook'):
        def coupled(module, args, output):
            if module is network:
                output = dict(output)
                hidden = output['last_hidden_state']
                output['last_hidden_state'] = hidden + torch.arange(hidden.size(0))[:, None, None] * .01
            return output
        handle = (network.register_forward_hook(coupled) if guard == 'forward_hook'
                  else module_hooks.register_module_forward_hook(coupled))
    elif guard == 'global_rng_hook':
        def consume(module, args, output):
            if module is network:
                torch.rand(output['last_hidden_state'].size(0))
        handle = module_hooks.register_module_forward_hook(consume)
    elif guard in ('pre_hook', 'global_pre_hook'):
        handle = (network.register_forward_pre_hook(lambda module, args: None)
                  if guard == 'pre_hook' else
                  module_hooks.register_module_forward_pre_hook(lambda module, args: None))
    elif guard == 'child_forward_override':
        old_forward = network.esm.forward
        def coupled_forward(*args, **kwargs):
            result = old_forward(*args, **kwargs)
            hidden = result.last_hidden_state
            result.last_hidden_state = hidden + torch.arange(hidden.size(0))[:, None, None] * .01
            return result
        monkeypatch.setattr(network.esm, 'forward', coupled_forward)
    elif guard == 'encoder_override':
        old_encode = scorer.encoder.encode_tokens
        monkeypatch.setattr(scorer.encoder, 'encode_tokens', old_encode)
    elif guard == 'compiled':
        network.compile(backend='eager', dynamic=False)
    elif guard == 'custom_encoder':
        class CustomEncoder(DPLMFeatureEncoder):
            pass
        scorer.encoder = CustomEncoder(scorer.encoder.backbone).eval()
    elif guard == 'custom_backbone':
        class CustomBackbone(DPLMBackbone):
            pass
        scorer.encoder.backbone = CustomBackbone(net=network).eval()
    elif guard == 'custom_net':
        class CustomNet(_EsmForDPLM):
            pass
        custom = CustomNet(deepcopy(network.config), dropout=.2)
        custom.load_state_dict(network.state_dict())
        scorer.encoder.backbone.net = custom.eval()
    else:
        mask = mask.float()
    rows = record_rows(monkeypatch)
    try:
        with torch.no_grad():
            torch.manual_seed(42)
            expected = original_features(scorer, noisy, candidate, mask)
            expected_rng = torch.random.get_rng_state().clone()
            rows.clear()
            torch.manual_seed(42)
            actual = scorer.build_conditioned_features(noisy, candidate, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(expected_rng, torch.random.get_rng_state())
        # An instance override binds the original method before row observation.
        if guard != 'encoder_override':
            assert rows == [8, 8]
    finally:
        if handle is not None:
            handle.remove()
        if guard == 'compiled':
            torch._dynamo.reset()


def test_gradient_enabled_eval_keeps_all_parameter_gradients(scorer):
    noisy, candidate, mask = inputs()
    gradients = []
    for function in (original_features, lambda model, *args: model.build_conditioned_features(*args)):
        scorer.zero_grad()
        value = function(scorer, noisy, candidate, mask)
        value.square().sum().backward()
        gradients.append({name: parameter.grad.clone() for name, parameter in scorer.named_parameters()
                          if parameter.grad is not None})
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)


def test_no_features_persist_after_weights_change(scorer):
    noisy, candidate, mask = inputs()
    with torch.no_grad():
        before = scorer.build_conditioned_features(noisy, candidate, mask)
        scorer.encoder.backbone.net.esm.embeddings.word_embeddings.weight[:, 0].add_(.125)
        expected = original_features(scorer, noisy, candidate, mask)
        after = scorer.build_conditioned_features(noisy, candidate, mask)
    torch.testing.assert_close(after, expected)
    assert not torch.equal(before, after)


@pytest.mark.parametrize('alias_path', ['encoder_buffer', 'candidate_hook'])
def test_fallback_preserves_two_encodes_before_pooling(scorer, alias_path):
    noisy, candidate, mask = inputs()
    hook = None
    captured = []
    if alias_path == 'encoder_buffer':
        class ReusingEncoder(DPLMFeatureEncoder):
            shared = None

            def encode_tokens(self, tokens, attention_mask=None):
                value = super().encode_tokens(tokens, attention_mask=attention_mask)
                if self.shared is None or self.shared.shape != value.shape:
                    self.shared = torch.empty_like(value)
                self.shared.copy_(value)
                return self.shared

        scorer.encoder = ReusingEncoder(scorer.encoder.backbone).eval()
    else:
        def mutate_previous_output(module, args, output):
            if captured:
                captured[0].add_(.125)
            else:
                captured.append(output['last_hidden_state'])

        hook = scorer.encoder.backbone.net.register_forward_hook(mutate_previous_output)
    try:
        with torch.no_grad():
            expected = original_features(scorer, noisy, candidate, mask)
            captured.clear()
            actual = scorer.build_conditioned_features(noisy, candidate, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if alias_path == 'encoder_buffer':
            half = actual.size(1) // 2
            torch.testing.assert_close(actual[:, :half], actual[:, half:], rtol=0, atol=0)
    finally:
        if hook is not None:
            hook.remove()


@pytest.mark.parametrize('child_behavior', ['stochastic', 'stateful'])
def test_custom_deep_child_keeps_full_rows_and_state(scorer, monkeypatch, child_behavior):
    noisy, candidate, mask = inputs()
    attention = scorer.encoder.backbone.net.esm.encoder.layer[0].attention.self
    original_query = attention.query

    class CustomLinear(torch.nn.Linear):
        processed_rows = 0

        def forward(self, value):
            result = super().forward(value)
            if child_behavior == 'stochastic':
                return result + .05 * torch.randn_like(result)
            self.processed_rows += value.size(0)
            return result + .005 * self.processed_rows

    query = CustomLinear(original_query.in_features, original_query.out_features,
                         bias=original_query.bias is not None)
    query.load_state_dict(original_query.state_dict())
    attention.query = query.eval()
    rows = record_rows(monkeypatch)
    with torch.no_grad():
        torch.manual_seed(42)
        expected = original_features(scorer, noisy, candidate, mask)
        expected_rng = torch.random.get_rng_state().clone()
        expected_state = query.processed_rows
        query.processed_rows = 0
        rows.clear()
        torch.manual_seed(42)
        actual = scorer.build_conditioned_features(noisy, candidate, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    assert query.processed_rows == expected_state
    assert rows == [8, 8]


@pytest.mark.parametrize('mask_shape', ['shared_batch', 'shared_length', 'shared_both',
                                      'rank3', 'rank4'])
def test_broadcast_masks_retain_original_path(scorer, monkeypatch, mask_shape):
    noisy, candidate, mask = inputs()
    if mask_shape == 'shared_batch':
        mask = mask[:1]
    elif mask_shape == 'shared_length':
        mask = mask[:, :1]
    elif mask_shape == 'shared_both':
        mask = mask[:1, :1]
    elif mask_shape == 'rank3':
        mask = mask[:1].unsqueeze(0)
    else:
        mask = mask[:1].unsqueeze(0).unsqueeze(0)
    rows = record_rows(monkeypatch)
    with torch.no_grad():
        expected = original_features(scorer, noisy, candidate, mask)
        rows.clear()
        actual = scorer.build_conditioned_features(noisy, candidate, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert rows == [8, 8]


@pytest.mark.parametrize('subclass_input', ['noisy', 'candidate', 'mask'])
def test_tensor_subclasses_retain_original_path(scorer, monkeypatch, subclass_input):
    class CustomTensor(torch.Tensor):
        pass

    noisy, candidate, mask = inputs()
    if subclass_input == 'noisy':
        noisy = noisy.as_subclass(CustomTensor)
    elif subclass_input == 'candidate':
        candidate = candidate.as_subclass(CustomTensor)
    else:
        mask = mask.as_subclass(CustomTensor)
    rows = record_rows(monkeypatch)
    with torch.no_grad():
        expected = original_features(scorer, noisy, candidate, mask)
        rows.clear()
        actual = scorer.build_conditioned_features(noisy, candidate, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert rows == [8, 8]


@pytest.mark.parametrize('broadcast_input', ['noisy', 'candidate'])
def test_broadcast_token_batches_retain_original_path(scorer, monkeypatch, broadcast_input):
    noisy, candidate, mask = inputs()
    if broadcast_input == 'noisy':
        noisy = noisy[:1]
    else:
        candidate = candidate[:1]
    rows = record_rows(monkeypatch)
    with torch.no_grad():
        expected = original_features(scorer, noisy, candidate, mask)
        rows.clear()
        actual = scorer.build_conditioned_features(noisy, candidate, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert rows == [noisy.size(0), candidate.size(0)]
