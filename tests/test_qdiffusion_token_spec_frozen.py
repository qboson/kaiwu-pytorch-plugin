"""SequenceTokenSpec must stay immutable once validated."""

import dataclasses

import pytest

from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec


def test_spec_attributes_cannot_be_reassigned():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3)

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.pad_id = 9
