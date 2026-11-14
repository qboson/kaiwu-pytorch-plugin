"""Cell extraction avoids reconstruction while preserving output/RNG extensions."""
import copy
import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("sklearn")
pytest.importorskip("kaiwu")
from kaiwu.classical import BruteForceOptimizer
from kaiwu.torch_plugin import BoltzmannMachine


@pytest.fixture
def cell():
    directory = Path(__file__).resolve().parents[1] / "example/qvae_cell"
    previous = sys.modules.pop("models", None)
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("cell_extraction", directory / "trainer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        models = sys.modules["models"]
    finally:
        sys.path.pop(0)
        sys.modules.pop("models", None)
        if previous is not None:
            sys.modules["models"] = previous
    return module, models


def make_model(cell, loss="mse", model_class=None):
    _, models = cell
    config = SimpleNamespace(num_latent_units=4, loss_type=loss, dist_beta=10.,
                             kl_beta=1e-5, weight_decay=0.)
    torch.manual_seed(47)
    cls = model_class or models.CellQVAE
    model = cls(7, torch.nn.ReLU(), config,
                encoder=models.QVAEEncoder(7, 8, 4),
                decoder=models.QVAEDecoder(6, 8, 7),
                bm=BoltzmannMachine(4, device="cpu"),
                sampler=BruteForceOptimizer(), n_batches=2)
    model.set_dataset_mean(torch.linspace(.1, .7, 7))
    model.set_train_bias(torch.linspace(.1, .7, 7))
    return model.eval()


def loader():
    values = torch.arange(119, dtype=torch.float32).reshape(17, 7) / 119
    return DataLoader(TensorDataset(values, torch.arange(17) % 2), batch_size=6)


def legacy_extract(model, batches, representation):
    model.eval()
    result = []
    with torch.no_grad():
        for values, labels in batches:
            _, _, q, zeta = model(values, labels)
            result.append((q if representation == "q" else zeta).numpy())
    return np.concatenate(result)


def run(operation):
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    result = operation()
    return result, torch.get_rng_state().clone(), random.getstate(), copy.deepcopy(np.random.get_state())


def assert_equal(left, right):
    np.testing.assert_array_equal(left[0], right[0])
    assert torch.equal(left[1], right[1])
    assert left[2] == right[2]
    assert left[3][0] == right[3][0]
    np.testing.assert_array_equal(left[3][1], right[3][1])
    assert left[3][2:] == right[3][2:]


def check_parity(cell, model, representation="q"):
    trainer = cell[0].Trainer(SimpleNamespace(representation=representation), torch.device("cpu"))
    state = copy.deepcopy(model.state_dict())
    old = run(lambda: legacy_extract(model, loader(), representation))
    old_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    new = run(lambda: trainer.extract_representation(model, loader()))
    assert_equal(old, new)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, old_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("loss", ["mse", "bernoulli"])
@pytest.mark.parametrize("representation", ["q", "zeta"])
def test_native_extraction_avoids_the_decoder(cell, loss, representation):
    model = make_model(cell, loss)
    trainer = cell[0].Trainer(SimpleNamespace(representation=representation), torch.device("cpu"))
    expected = run(lambda: legacy_extract(model, loader(), representation))
    calls = []
    decoder_code = cell[1].QVAEDecoder.forward.__code__
    previous = sys.getprofile()
    def trace(frame, event, _):
        if event == "call" and frame.f_code is decoder_code:
            calls.append(1)
    sys.setprofile(trace)
    try:
        observed = run(lambda: trainer.extract_representation(model, loader()))
    finally:
        sys.setprofile(previous)
    assert_equal(expected, observed)
    assert not calls, "unused reconstruction ran during representation extraction"


@pytest.mark.parametrize("extension", ["forward_hook", "prehook", "decoder_hook", "child_hook",
                                        "global_hook", "subclass", "instance_forward", "custom_decoder",
                                        "custom_child", "stateful_encoder", "decoder_input", "training_decoder",
                                        "compiled"])
def test_extensions_keep_full_forward_semantics(cell, extension):
    from types import MethodType
    model = make_model(cell)
    handles = []
    def consume_rng(*_):
        random.random()
        np.random.rand()
        torch.rand(())
    def alter_q(_, __, output):
        return output[0], output[1], output[2] + 3, output[3]
    if extension == "forward_hook":
        handles.append(model.register_forward_hook(alter_q))
    elif extension == "prehook":
        handles.append(model.register_forward_pre_hook(lambda _, args: (args[0] / 2, args[1])))
    elif extension == "decoder_hook":
        handles.append(model.decoder.register_forward_hook(consume_rng))
    elif extension == "child_hook":
        handles.append(model.decoder.fc1.register_forward_pre_hook(consume_rng))
    elif extension == "global_hook":
        handles.append(torch.nn.modules.module.register_module_forward_hook(consume_rng))
    elif extension == "subclass":
        class Modified(cell[1].CellQVAE):
            def forward(self, values, labels=None):
                result = super().forward(values, labels)
                return result[0], result[1], result[2] + 3, result[3]
        model = make_model(cell, model_class=Modified)
    elif extension == "instance_forward":
        original = model.forward
        model.forward = lambda *args: alter_q(None, None, original(*args))
    elif extension == "custom_decoder":
        class StochasticDecoder(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, values):
                consume_rng()
                return self.base(values)
        model.decoder = StochasticDecoder(model.decoder)
    elif extension == "custom_child":
        class StochasticLinear(torch.nn.Linear):
            def forward(self, values):
                consume_rng()
                return super().forward(values)
        model.decoder.fc1 = StochasticLinear(6, 8)
    elif extension == "stateful_encoder":
        class StatefulEncoder(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.register_buffer("calls", torch.zeros(()))
            def forward(self, values):
                consume_rng()
                self.calls.add_(1)
                return self.base(values) + self.calls * .001
        model.encoder = StatefulEncoder(model.encoder)
    elif extension == "decoder_input":
        original = model._decoder_input
        def modified(*args):
            consume_rng()
            return original(*args)
        model._decoder_input = modified
    elif extension == "training_decoder":
        model.decoder.dropout.train(True)
        model.decoder.dropout.train = MethodType(lambda self, mode=True: self, model.decoder.dropout)
    elif extension == "compiled":
        model.compile(backend="eager")
    try:
        check_parity(cell, model, "q" if extension in ("forward_hook", "subclass", "instance_forward") else "zeta")
    finally:
        for handle in handles:
            handle.remove()


def test_loader_can_install_a_hook_before_yielding_a_batch(cell):
    model = make_model(cell)
    trainer = cell[0].Trainer(SimpleNamespace(representation="q"), torch.device("cpu"))
    handles = []
    class InstrumentedLoader:
        def __iter__(self):
            def alter_q(_, __, output):
                return output[0], output[1], output[2] + 2, output[3]
            handles.append(model.register_forward_hook(alter_q))
            yield next(iter(loader()))
    try:
        expected = run(lambda: legacy_extract(model, InstrumentedLoader(), "q"))
        for handle in handles:
            handle.remove()
        handles.clear()
        observed = run(lambda: trainer.extract_representation(model, InstrumentedLoader()))
        assert_equal(expected, observed)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("representation, expected", [("q", 2.), ("zeta", 5.)])
def test_generic_forward_protocol_remains_supported(cell, representation, expected):
    class SuppliedModel:
        def eval(self):
            return self
        def __call__(self, values, labels):
            return None, None, torch.full((len(values), 4), 2.), torch.full((len(values), 4), 5.)
    trainer = cell[0].Trainer(SimpleNamespace(representation=representation), torch.device("cpu"))
    actual = trainer.extract_representation(SuppliedModel(), loader())
    np.testing.assert_array_equal(actual, np.full((17, 4), expected))
