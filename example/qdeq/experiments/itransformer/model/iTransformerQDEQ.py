"""Q-DEQ-Ising-SA variant of iTransformer: the stacked EncoderLayer list is
replaced by a single shared EncoderLayer solved to a fixed point via the
Ising/SA coefficient solver (see model/deq_ising_ts.py). Interface-compatible
with model/iTransformer.py so it plugs into run.py / exp_basic.py unchanged.
"""

import torch
import torch.nn as nn

from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import EncoderLayer
from layers.Embed import DataEmbedding_inverted
from model.deq_ising_ts import EncoderLayerIsingDEQ, EncoderLayerAndersonDEQ, replace_linear_recursive


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.class_strategy = configs.class_strategy

        shared_layer = EncoderLayer(
            AttentionLayer(
                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                              output_attention=configs.output_attention),
                configs.d_model, configs.n_heads
            ),
            configs.d_model,
            configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
        )
        deq_variant = getattr(configs, "deq_variant", "ising_sa")
        if deq_variant == "ising_sa":
            self.deq = EncoderLayerIsingDEQ(shared_layer, configs)
        elif deq_variant == "anderson":
            self.deq = EncoderLayerAndersonDEQ(shared_layer, configs)
        else:
            raise ValueError(f"Unsupported deq_variant={deq_variant!r}")
        self.final_norm = nn.LayerNorm(configs.d_model)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        if deq_variant == "ising_sa" and getattr(configs, "quant_enabled", False):
            replace_linear_recursive(self.deq.layer, bits=getattr(configs, "quant_bits", 8))

        init_ckpt = getattr(configs, "init_checkpoint", None)
        if init_ckpt:
            self._load_pretrained(init_ckpt)
        if getattr(configs, "deq_orthogonal_reinit", False):
            self._orthogonal_reinit_shared_layer()
        if getattr(configs, "deq_wnorm", False):
            self._apply_weight_norm_shared_layer()

    def _load_pretrained(self, path):
        """Load a 1-layer explicit iTransformer checkpoint (trained via the
        unmodified model/iTransformer.py with --e_layers 1) as the initial
        shared-layer weights (theta0) for the DEQ solver."""
        state = torch.load(path, map_location="cpu")
        remapped = {}
        for k, v in state.items():
            if k.startswith("enc_embedding."):
                remapped[k] = v
            elif k.startswith("encoder.attn_layers.0."):
                remapped["deq.layer." + k[len("encoder.attn_layers.0."):]] = v
            elif k.startswith("encoder.norm."):
                remapped["final_norm." + k[len("encoder.norm."):]] = v
            elif k.startswith("projector."):
                remapped[k] = v
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        expected_missing = {"deq.norm_inject.weight", "deq.norm_inject.bias"}
        real_missing = [m for m in missing if m not in expected_missing]
        print(f"[Q-DEQ init] loaded pretrained theta0 from {path}")
        print(f"[Q-DEQ init] missing (expected: deq.norm_inject.*): {missing}")
        print(f"[Q-DEQ init] unexpected: {unexpected}")
        if real_missing:
            raise RuntimeError(f"Unexpected missing keys after theta0 remap: {real_missing}")

    def _orthogonal_reinit_shared_layer(self):
        """Reinitialize the shared DEQ layer's Q/K/V/O + conv1/conv2 weight
        matrices with orthogonal init, keeping theta0's biases/norms/
        embedding/projector untouched. Motivated by an empirical finding:
        theta0 (a 1-epoch, single-application-trained explicit layer) has
        badly-conditioned weight matrices -- condition numbers in the
        thousands for ECL/traffic (datasets where DEQ trails baseline by
        8-10%) vs ~200 for ETTh1/ETTh2 (where DEQ matches or beats baseline)
        -- consistent with the DEQ literature's finding that equilibrium
        models are highly sensitive to the higher-order statistics of their
        initialization (arXiv:2207.09432)."""
        layer = self.deq.layer
        attn = layer.attention
        for proj in (attn.query_projection, attn.key_projection, attn.value_projection, attn.out_projection):
            nn.init.orthogonal_(proj.weight)
        nn.init.orthogonal_(layer.conv1.weight)
        nn.init.orthogonal_(layer.conv2.weight)
        print("[Q-DEQ init] applied orthogonal reinit to shared layer Q/K/V/O + conv1/conv2 weights", flush=True)

    def _apply_weight_norm_shared_layer(self):
        """Reparameterize the shared DEQ layer's Q/K/V/O + conv1/conv2 weights
        via weight normalization (W = g * v/||v||, dim=0), matching locuslab/deq's
        DEQ-Transformer `wnorm()` (see DEQ-Sequence/models/deq_transformer.py).
        Unlike a one-time init (orthogonal reinit, which we tried and found made
        things worse), this is a continuous reparameterization that keeps the
        shared layer's effective Lipschitz constant controlled throughout
        training, not just at step 0. The reference WT103 config (d_model=700,
        wider than our ECL's 512) enables --wnorm and trains stably despite
        being a much wider fixed-point map, suggesting wnorm -- not sequence/
        token count -- is what lets it scale."""
        from torch.nn.utils.parametrizations import weight_norm
        layer = self.deq.layer
        attn = layer.attention
        for proj in (attn.query_projection, attn.key_projection, attn.value_projection, attn.out_projection):
            weight_norm(proj, name="weight", dim=0)
        weight_norm(layer.conv1, name="weight", dim=0)
        weight_norm(layer.conv2, name="weight", dim=0)
        print("[Q-DEQ init] applied weight_norm (dim=0) to shared layer Q/K/V/O + conv1/conv2 weights", flush=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        z_star = self.deq(enc_out)
        enc_out = self.final_norm(z_star)

        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
