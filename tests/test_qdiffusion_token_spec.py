"""SequenceTokenSpec must reject ids that cannot identify distinct tokens."""

import pytest

from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec


def _spec(**overrides):
    values = {"pad_id": 0, "bos_id": 1, "eos_id": 2, "mask_id": 3}
    values.update(overrides)
    return SequenceTokenSpec(**values)


def test_duplicate_special_ids_are_rejected():
    with pytest.raises(ValueError, match="distinct"):
        _spec(bos_id=0)


def test_a_reserved_id_clashing_with_a_special_id_is_rejected():
    with pytest.raises(ValueError, match="distinct"):
        SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=3)


@pytest.mark.parametrize("field", ["pad_id", "bos_id", "eos_id", "mask_id"])
def test_negative_ids_are_rejected(field):
    with pytest.raises(ValueError, match="non-negative"):
        _spec(**{field: -1})


def test_non_integer_ids_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        _spec(pad_id=0.0)


def test_a_valid_spec_keeps_its_ids():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)

    assert (spec.pad_id, spec.bos_id, spec.eos_id, spec.mask_id, spec.x_id) == (0, 1, 2, 3, 4)
