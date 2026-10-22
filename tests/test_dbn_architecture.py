"""UnsupervisedDBN must validate its layer architecture at construction."""

import pytest

from kaiwu.torch_plugin import UnsupervisedDBN


def test_empty_structure_is_rejected() -> None:
    with pytest.raises(ValueError, match="must contain at least one layer"):
        UnsupervisedDBN([])


@pytest.mark.parametrize("structure", [[3, 0], [3, -1]])
def test_non_positive_layer_width_is_rejected(structure) -> None:
    with pytest.raises(ValueError, match="must be positive integers"):
        UnsupervisedDBN(structure)


@pytest.mark.parametrize("structure", [[3, 2.5], [3, True], 4])
def test_non_integer_layer_width_is_rejected(structure) -> None:
    with pytest.raises(TypeError, match="hidden_layers_structure"):
        UnsupervisedDBN(structure)


def test_valid_structure_builds_all_layers() -> None:
    dbn = UnsupervisedDBN([3, 2]).create_rbm_layer(4)
    assert dbn.num_layers == 2
    assert [rbm.num_visible for rbm in dbn.rbm_layers] == [4, 3]
    assert [rbm.num_hidden for rbm in dbn.rbm_layers] == [3, 2]


@pytest.mark.parametrize("input_dim", [0, -4])
def test_non_positive_input_dim_is_rejected(input_dim) -> None:
    with pytest.raises(ValueError, match="input_dim must be a positive integer"):
        UnsupervisedDBN([3]).create_rbm_layer(input_dim)


@pytest.mark.parametrize("input_dim", [4.0, True])
def test_non_integer_input_dim_is_rejected(input_dim) -> None:
    with pytest.raises(TypeError, match="input_dim must be a positive integer"):
        UnsupervisedDBN([3]).create_rbm_layer(input_dim)
