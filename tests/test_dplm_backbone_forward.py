"""Forward-pass contract for the DPLM backbone wrapper."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from models.backbone import DPLMBackbone  # noqa: E402


class FakeNet:
    """Returns distinct logits/hidden states and records the input ids."""

    def __init__(self):
        from types import SimpleNamespace

        self.config = SimpleNamespace(hidden_size=8)
        self.tokenizer = "TOKENIZER"
        self.mask_id = 1
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3
        self.x_id = 4
        self.calls = []
        self.logits = torch.full((1, 3, 5), 7.0)
        self.hidden = torch.full((1, 3, 8), 9.0)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"logits": self.logits, "last_hidden_state": self.hidden}


def test_forward_returns_logits_from_the_wrapped_net():
    """The default forward returns logits for the given ids."""
    net = FakeNet()
    wrapper = DPLMBackbone(net=net)
    tokens = torch.tensor([[1, 2, 3]])
    output = wrapper(tokens)
    assert net.calls == [{"input_ids": tokens}]
    assert torch.equal(output, net.logits)


def test_forward_can_return_the_last_hidden_state():
    """``return_last_hidden_state=True`` returns both tensors in order."""
    net = FakeNet()
    wrapper = DPLMBackbone(net=net)
    tokens = torch.tensor([[1, 2, 3]])
    logits, hidden = wrapper(tokens, return_last_hidden_state=True)
    assert torch.equal(logits, net.logits)
    assert torch.equal(hidden, net.hidden)
