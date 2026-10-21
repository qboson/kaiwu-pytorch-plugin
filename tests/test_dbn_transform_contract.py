"""Pin the layer-by-layer composition performed by UnsupervisedDBN.transform."""

import numpy as np
import torch

from kaiwu.torch_plugin import UnsupervisedDBN


def test_transform_matches_manual_layer_forwarding() -> None:
    torch.manual_seed(0)
    dbn = UnsupervisedDBN([3, 2]).create_rbm_layer(4).mark_as_trained()
    x = np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32)

    h1 = dbn.get_rbm_layer(0).get_hidden(torch.FloatTensor(x))[:, 4:].numpy()
    h2 = dbn.get_rbm_layer(1).get_hidden(torch.FloatTensor(h1))[:, 3:].numpy()

    transformed = dbn.transform(x)
    assert transformed.shape == (2, 2)
    np.testing.assert_allclose(transformed, h2, rtol=1e-6, atol=1e-6)


def test_transform_only_extracts_hidden_units() -> None:
    torch.manual_seed(0)
    dbn = UnsupervisedDBN([3]).create_rbm_layer(4).mark_as_trained()
    x = np.ones((1, 4), dtype=np.float32)

    full_state = dbn.get_rbm_layer(0).get_hidden(torch.FloatTensor(x))
    transformed = dbn.transform(x)
    np.testing.assert_allclose(transformed, full_state[:, 4:].numpy(), rtol=1e-6)
    assert transformed.shape == (1, 3)
