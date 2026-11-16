"""Exercise decoding with a proposal backbone and no energy backend."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import QDiffusion, QDiffusionConfig, SequenceTokenSpec


class LocalProposal(nn.Module):
    tokenizer = None
    pad_id, bos_id, eos_id, mask_id, x_id = range(5)

    def forward(self, tokens):
        logits = torch.full((*tokens.shape, 8), -100.0, device=tokens.device)
        logits[..., 5] = 100.0
        return logits


def test_proposal_only_decoding_preserves_fixed_tokens():
    spec = SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4)
    generator = QDiffusion(LocalProposal(), None, spec, QDiffusionConfig(disable_resample=True))
    inputs = torch.tensor([[1, 6, 7, 2, 0]])
    fixed = torch.tensor([[False, True, False, False, False]])

    output = generator.generate(inputs, partial_masks=fixed, max_steps=3)

    torch.testing.assert_close(output, torch.tensor([[1, 6, 5, 2, 0]]))
    with pytest.raises(RuntimeError, match="energy scoring requires an energy model"):
        generator.objective({"targets": inputs})
    with pytest.raises(ValueError, match="num_candidates=1"):
        QDiffusion(LocalProposal(), None, spec, QDiffusionConfig(num_candidates=2))


def test_esm2_baseline_builder_loads_only_the_proposal_checkpoint(monkeypatch):
    pytest.importorskip("omegaconf")
    pytest.importorskip("transformers")
    pytest.importorskip("esm")
    from example.qdiffusion.dplm.utils import dplm_builder
    from example.qdiffusion.dplm.workflows.esm2_eval import build_generator_for_eval

    loads = []

    def load(checkpoint, **kwargs):
        loads.append(checkpoint)
        return LocalProposal()

    monkeypatch.setattr(dplm_builder, "load_dplm_backbone", load)
    config = SimpleNamespace(
        proposal_ckpt="proposal", energy_ckpt="energy", freeze_proposal=True,
        bm_sampler_type="sa", bm_sampler_kwargs={},
    )
    generator = build_generator_for_eval(
        config, device=torch.device("cpu"), num_candidates=1, proposal_only=True,
    )

    assert loads == ["proposal"]
    assert generator.energy_model is None
    output = generator.generate(torch.tensor([[1, 6, 2]]), max_steps=1)
    torch.testing.assert_close(output, torch.tensor([[1, 5, 2]]))
