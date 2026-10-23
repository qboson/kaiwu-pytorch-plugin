"""Regression tests for DPLM ESM attention-mask device placement.

``_ModifiedEsmModel.forward`` builds or forwards an attention mask and optional
fused embeddings into ``EsmEmbeddings``. Those embeddings multiply the mask and
add the position-embedding parameters. A parent ``nn.Module.to`` moves the
weights through ``_apply`` and does not call the child's ``.to`` (merged #164),
so a CPU mask meets parameters that already moved:

``RuntimeError: Tensor on device meta is not on the expected device cpu!``

The mask and fused embeddings must follow the live embedding parameters. Caller
tensors are not modified in place.
"""

# Parent-initiated moves go through ``_apply`` (see merged #164).
# The recording subclass exists only to count ``.to`` calls.
# pylint: disable=protected-access,too-few-public-methods

import importlib.util
import unittest
import warnings
from pathlib import Path

import torch
from torch import nn

warnings.filterwarnings("ignore", message=r".*GenerationMixin.*")

try:
    from transformers import EsmConfig
except ImportError:  # pragma: no cover - optional example dependency
    EsmConfig = None


def _load_esm_patch():
    """Load the example module without importing the DPLM package init."""
    path = (
        Path(__file__).resolve().parents[1]
        / "example"
        / "qdiffusion"
        / "dplm"
        / "models"
        / "esm_patch.py"
    )
    spec = importlib.util.spec_from_file_location("esm_patch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if EsmConfig is not None:
    _ESM_PATCH = _load_esm_patch()

    class RecordingEsmModel(_ESM_PATCH._ModifiedEsmModel):
        """ESM backbone that records direct ``.to`` calls."""

        def __init__(self, config) -> None:
            super().__init__(config, add_pooling_layer=False)
            self.direct_to_calls: list[tuple] = []

        def to(self, *args, **kwargs):
            """Record the call, then perform the real move."""
            self.direct_to_calls.append((args, kwargs))
            return super().to(*args, **kwargs)

else:  # pragma: no cover - optional example dependency
    _ESM_PATCH = None

    class RecordingEsmModel(nn.Module):
        """Placeholder so collection succeeds without transformers."""

        direct_to_calls: list[tuple] = []

        def forward(self, *_args, **_kwargs):  # pragma: no cover
            """Unused stand-in. Tests that need the real model are skipped."""
            raise RuntimeError("transformers is required")


def _tiny_config():
    """Return a CPU-cheap ESM config with absolute position embeddings."""
    return EsmConfig(
        vocab_size=20,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        pad_token_id=0,
        mask_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
        token_dropout=False,
        emb_layer_norm_before=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


def _cpu_model() -> RecordingEsmModel:
    """Build one eval-mode tiny backbone on CPU."""
    model = RecordingEsmModel(_tiny_config())
    model.eval()
    return model


def _parent_meta(model: RecordingEsmModel) -> nn.Module:
    """Move ``model`` to ``meta`` the way a parent module does."""
    parent = nn.Module()
    parent.model = model
    parent._apply(lambda tensor: tensor.to(torch.device("meta")))
    return parent


@unittest.skipUnless(EsmConfig is not None, "requires transformers")
class TestEsmMaskParameterDevice(unittest.TestCase):
    """Meta-device mechanism tests that run on every machine."""

    def test_parent_apply_accepts_cpu_token_ids(self):
        """CPU token ids must run after a parent move that skips ``.to``."""
        model = _cpu_model()
        parent = _parent_meta(model)
        tokens = torch.tensor([[2, 5, 6, 3]])

        hidden = model(input_ids=tokens, return_dict=True).last_hidden_state

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(next(model.parameters()).device, torch.device("meta"))
        self.assertIs(parent.model, model)
        self.assertEqual(hidden.device, torch.device("meta"))
        self.assertEqual(hidden.shape, (1, 4, 16))
        self.assertEqual(tokens.device, torch.device("cpu"))

    def test_provided_cpu_mask_follows_parameters_not_the_caller(self):
        """A caller mask stays on CPU while the embedding mask is ``meta``."""
        model = _cpu_model()
        _parent_meta(model)
        tokens = torch.tensor([[2, 5, 6, 0]])
        mask = tokens.ne(0)
        mask_ptr = mask.data_ptr()
        seen = {}
        original = model.embeddings.forward

        def _spy(*args, **kwargs):
            seen["mask"] = kwargs["attention_mask"]
            return original(*args, **kwargs)

        model.embeddings.forward = _spy

        hidden = model(
            input_ids=tokens, attention_mask=mask, return_dict=True
        ).last_hidden_state

        self.assertEqual(seen["mask"].device, torch.device("meta"))
        self.assertEqual(hidden.device, torch.device("meta"))
        self.assertEqual(mask.device, torch.device("cpu"))
        self.assertEqual(mask.data_ptr(), mask_ptr)

    def test_cpu_inputs_embeds_follow_position_embedding_parameters(self):
        """Fused embeddings must meet position weights on one device."""
        model = _cpu_model()
        _parent_meta(model)
        embeds = torch.ones(1, 4, 16)
        embed_ptr = embeds.data_ptr()
        mask = torch.ones(1, 4)
        seen = {}
        original = model.embeddings.forward

        def _spy(*args, **kwargs):
            seen["inputs_embeds"] = kwargs["inputs_embeds"]
            return original(*args, **kwargs)

        model.embeddings.forward = _spy

        hidden = model(
            inputs_embeds=embeds, attention_mask=mask, return_dict=True
        ).last_hidden_state

        self.assertEqual(seen["inputs_embeds"].device, torch.device("meta"))
        self.assertEqual(hidden.device, torch.device("meta"))
        self.assertEqual(hidden.shape, (1, 4, 16))
        self.assertEqual(embeds.device, torch.device("cpu"))
        self.assertEqual(embeds.data_ptr(), embed_ptr)

    def test_dtype_only_move_keeps_cpu_device(self):
        """``.to(dtype=...)`` must not send a CPU backbone off CPU."""
        model = _cpu_model()
        model.to(dtype=torch.float64)
        tokens = torch.tensor([[2, 5, 6, 3]])

        hidden = model(input_ids=tokens, return_dict=True).last_hidden_state

        self.assertEqual(next(model.parameters()).device, torch.device("cpu"))
        self.assertEqual(next(model.parameters()).dtype, torch.float64)
        self.assertEqual(hidden.dtype, torch.float64)
        self.assertEqual(hidden.device, torch.device("cpu"))

    def test_same_device_alignment_keeps_autograd(self):
        """A CPU batch must still gradient the embedding weights and the input."""
        model = _cpu_model()
        embeds = torch.randn(2, 4, 16, requires_grad=True)
        mask = torch.ones(2, 4)

        hidden = model(
            inputs_embeds=embeds, attention_mask=mask, return_dict=True
        ).last_hidden_state
        hidden.sum().backward()

        self.assertEqual(hidden.device, torch.device("cpu"))
        self.assertIsNotNone(embeds.grad)
        self.assertEqual(embeds.grad.device, torch.device("cpu"))
        weight_grad = model.embeddings.position_embeddings.weight.grad
        self.assertIsNotNone(weight_grad)
        self.assertEqual(weight_grad.device, torch.device("cpu"))

    def test_parameter_device_is_reread_after_leaving_meta(self):
        """A later ``_apply`` back to CPU must win over the meta placement."""
        model = _cpu_model()
        _parent_meta(model)

        def _empty_cpu(tensor: torch.Tensor) -> torch.Tensor:
            return torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")

        model._apply(_empty_cpu)
        tokens = torch.tensor([[2, 5, 6, 3]])
        hidden = model(input_ids=tokens, return_dict=True).last_hidden_state

        self.assertEqual(next(model.parameters()).device, torch.device("cpu"))
        self.assertEqual(hidden.device, torch.device("cpu"))
        self.assertEqual(hidden.shape, (1, 4, 16))

    def test_parameter_device_falls_back_when_embeddings_have_no_parameters(self):
        """An empty module must not raise and must use the caller device."""
        fallback = torch.zeros(1)

        device = _ESM_PATCH._parameter_device(nn.Identity(), fallback)

        self.assertEqual(device, torch.device("cpu"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipUnless(EsmConfig is not None, "requires transformers")
class TestEsmMaskParameterDeviceCuda(unittest.TestCase):
    """CUDA coverage of the construct-on-CUDA / move-to-CPU path."""

    def test_parent_to_cuda_accepts_cpu_tokens(self):
        """A parent CUDA move must score a CPU batch on the parameter device."""
        model = _cpu_model()
        parent = nn.Module()
        parent.model = model
        parent.to("cuda")
        tokens = torch.tensor([[2, 5, 6, 3]])

        hidden = model(input_ids=tokens, return_dict=True).last_hidden_state

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(hidden.device.type, "cuda")
        self.assertEqual(tokens.device.type, "cpu")

    def test_parent_round_trip_accepts_the_other_devices_batch(self):
        """CPU tokens after CUDA, then CUDA tokens after the move back."""
        model = _cpu_model()
        parent = nn.Module()
        parent.model = model
        parent.to("cuda")
        cpu_tokens = torch.tensor([[2, 5, 6, 3]])
        cuda_hidden = model(input_ids=cpu_tokens, return_dict=True).last_hidden_state
        parent.to("cpu")
        cuda_tokens = torch.tensor([[2, 5, 6, 3]], device="cuda")
        cpu_hidden = model(input_ids=cuda_tokens, return_dict=True).last_hidden_state

        self.assertEqual(cuda_hidden.device.type, "cuda")
        self.assertEqual(cpu_hidden.device.type, "cpu")
        self.assertEqual(cuda_tokens.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
