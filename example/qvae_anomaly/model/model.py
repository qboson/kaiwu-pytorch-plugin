# -*- coding: utf-8 -*-
"""Clean Energy-Supervised QVAE.

Only the model class lives here; loss/RBM-stat helpers are in losses.py.
"""
import torch
import torch.nn.functional as F
from kaiwu.torch_plugin import QVAE, RestrictedBoltzmannMachine
from kaiwu.torch_plugin.qvae_dist_util import MixtureGeneric
from kaiwu.classical import SimulatedAnnealingOptimizer

from .networks import CleanDecoder, CleanEncoder
from .losses import (
    _compute_clean_rbm_statistics,
    _compose_energy_supervised_qvae_loss_terms,
    _normalize_anomaly_loss_type,
    _normalize_energy_score_type,
    _normalize_infonce_anchor_mode,
    _normalize_negative_phase_mode,
    _sample_visible_decoder_latent,
    _select_score_energy,
    _weighted_qvae_weight_decay_loss,
)
from utils.exception import BuildError, SizeError
from utils.logging import get_logger

logger = get_logger("model")


class CleanEnergyQVAE(QVAE):
    """Energy-supervised deterministic QVAE with a clean RBM latent split."""

    def __init__(self, input_dimension=None, activation_fct=None, config=None, **kwargs):
        super().__init__(input_dimension=input_dimension, activation_fct=activation_fct,
                         config=config, **kwargs)
        self._model_type = "QVAE-Anomaly"
        self.variant = "clean_det"
        self.loss_type = "mse"
        self.is_supervised = False
        self.supervision_mode = "energy"
        self.latent_dim = self.bm.num_visible
        self.energy_margin = float(getattr(config, "energy_margin", 0.0))
        self.lambda_anom = float(getattr(config, "lambda_anom", 1.0))
        self.energy_score_type = _normalize_energy_score_type(getattr(config, "energy_score_type", "free"))
        self.anomaly_loss_type = _normalize_anomaly_loss_type(getattr(config, "anomaly_loss_type", "margin"))
        self.infonce_temperature = float(getattr(config, "infonce_temperature", 1.0))
        self.infonce_anchor_mode = _normalize_infonce_anchor_mode(getattr(config, "infonce_anchor_mode", "normal_anchor"))
        self.negative_phase_mode = _normalize_negative_phase_mode(getattr(config, "negative_phase_mode", "sampled"))
        self.recon_loss_weight = float(getattr(config, "recon_loss_weight", 1.0))
        self.anomaly_margin_weight = float(getattr(config, "anomaly_margin_weight", 1.0))
        self.anomaly_infonce_weight = float(getattr(config, "anomaly_infonce_weight", 1.0))
        self.bm_weight_decay_weight = float(getattr(config, "bm_weight_decay_weight", 1.0))
        self.negative_phase_sample_calls = 0

    def __repr__(self):
        return (f"CleanEnergyQVAE(in={self._input_dimension}, "
                f"hidden={self.config.hidden_dim}, "
                f"RBM={self.bm.num_visible}x{self.bm.num_hidden}, "
                f"lambda={self.lambda_anom}, neg_phase={self.negative_phase_mode})")

    def _create_encoder(self):
        return CleanEncoder(input_dim=self._input_dimension, hidden_dim=self.config.hidden_dim,
                            latent_dim=self._latent_dimensions, activation_fct=self._activation_fct,
                            backbone_type=getattr(self.config, "backbone_type", "residual"), weight_decay=0.0)

    def _create_decoder(self):
        return CleanDecoder(input_dim=self._input_dimension, hidden_dim=self.config.hidden_dim,
                            latent_dim=self._latent_dimensions, activation_fct=self._activation_fct,
                            backbone_type=getattr(self.config, "backbone_type", "residual"), weight_decay=0.0)

    def _create_bm(self, bm_type="rbm"):
        if bm_type != "rbm":
            raise ValueError(f"only rbm supported, got {bm_type}")
        n1 = int(getattr(self.config, "num_var1", self._latent_dimensions))
        n2 = int(getattr(self.config, "num_var2", n1))
        return RestrictedBoltzmannMachine(num_visible=n1, num_hidden=n2)

    def _create_sampler(self, sampler_type="sa"):
        if sampler_type == "cim":
            import kaiwu as kw
            from kaiwu.cim import CIMOptimizer, PrecisionReducer
            kw.common.CheckpointManager.save_dir = "./tmp"
            s = CIMOptimizer(task_name="clean_qvae_sampling", wait=True)
            return PrecisionReducer(s, precision=8, truncated_precision=10, target_bits=550, only_feasible_solution=False)
        if sampler_type == "sa":
            return SimulatedAnnealingOptimizer(alpha=0.95)
        raise ValueError(f"unknown sampler {sampler_type}")

    def _native_forward(self, x, include_negative_reference=False):
        self.is_training = self.training
        visible_logits = self.encoder(x)
        stats = _compute_clean_rbm_statistics(self, visible_logits,
                                              include_negative_reference=include_negative_reference)
        zeta = _sample_visible_decoder_latent(self, visible_logits)
        return self.decoder(zeta), visible_logits, zeta, stats

    def forward(self, x):
        rx, vl, zeta, stats = self._native_forward(x)
        return rx, MixtureGeneric(vl, self.dist_beta), vl, zeta

    def get_cross_entropy(self, x):
        return self._native_forward(x, include_negative_reference=True)[3]["cross_entropy_per_sample"]

    def get_expected_energy(self, x):
        return self._native_forward(x)[3]["expected_energy_per_sample"]

    def get_free_energy(self, x):
        return self._native_forward(x)[3]["free_energy_per_sample"]

    def get_score_energy(self, x, energy_score_type=None):
        return _select_score_energy(self._native_forward(x)[3], energy_score_type or self.energy_score_type)

    def get_reconstruction_error(self, x):
        return F.mse_loss(self._native_forward(x)[0], x, reduction="none").sum(dim=1)

    def free_energy(self, q, mode="probabilistic"):
        v = (q > 0).float() if mode == "binary" else torch.sigmoid(q)
        vp = v[:, : self.bm.num_visible]
        ha = vp @ self.bm.quadratic_coef + self.bm.hidden_bias
        return -(vp @ self.bm.visible_bias) - torch.sum(F.softplus(ha), dim=1)

    def energy(self, x, loss_type=None):
        return self.get_score_energy(x)

    def predict_anomaly_score(self, x, w1=0.5, w2=0.5, style="energy", sample_size=1):
        return self.get_score_energy(x)

    def loss_terms(self, x, kl_beta, labels=None, alpha=0.5):
        rx, vl, zeta, stats = self._native_forward(x, include_negative_reference=True)
        rs = F.mse_loss(rx, x, reduction="none").sum(dim=1)
        terms = _compose_energy_supervised_qvae_loss_terms(
            rs, stats["cross_entropy_per_sample"], stats["entropy_per_sample"],
            stats["expected_energy_per_sample"],
            _select_score_energy(stats, self.energy_score_type),
            _weighted_qvae_weight_decay_loss(self), kl_beta,
            lambda_anom=self.lambda_anom, labels=labels,
            energy_margin=self.energy_margin, anomaly_loss_type=self.anomaly_loss_type,
            infonce_temperature=self.infonce_temperature,
            infonce_anchor_mode=self.infonce_anchor_mode,
            anomaly_margin_weight=self.anomaly_margin_weight,
            anomaly_infonce_weight=self.anomaly_infonce_weight,
            recon_loss_weight=self.recon_loss_weight)
        terms.update({"recon_like": rx, "q": vl, "zeta": zeta,
                      "visible_probs": stats["visible_probs"],
                      "hidden_probs": stats["hidden_probs"],
                      "expected_energy_loss": stats["expected_energy_per_sample"].mean(),
                      "free_energy_loss": stats["free_energy_per_sample"].mean(),
                      "negative_reference_loss": stats["negative_reference"],
                      "negative_phase_mode": stats["negative_phase_mode"],
                      "cross_entropy_loss": stats["cross_entropy_per_sample"].mean(),
                      "entropy_loss": stats["entropy_per_sample"].mean()})
        return terms

    def neg_elbo(self, x, kl_beta, labels=None, alpha=0.5):
        t = self.loss_terms(x, kl_beta, labels=labels, alpha=alpha)
        return (t["total_loss"], t["gen_loss"], t["cls_loss"], None, t["recon_like"], t["q"], t["zeta"])

    def bm_loss(self, q, bm_weight_decay=0.0):
        return _weighted_qvae_weight_decay_loss(self)

    def _weight_decay_loss(self):
        return _weighted_qvae_weight_decay_loss(self)

    def _cross_entropy(self, logit_q):
        return _compute_clean_rbm_statistics(self, logit_q, include_negative_reference=True)["cross_entropy_per_sample"]

    def loss(self, x, recon_x, posterior):
        return self.loss_terms(x, kl_beta=self.kl_beta, labels=None)["total_loss"]


def build_model(cfg: "Config") -> "CleanEnergyQVAE":
    """Factory: build CleanEnergyQVAE from Config."""
    # Validate config
    if cfg.input_dimension <= 0:
        raise SizeError(f"input_dimension must be positive, got {cfg.input_dimension}")
    if cfg.hidden_dim <= 0:
        raise SizeError(f"hidden_dim must be positive, got {cfg.hidden_dim}")
    if cfg.num_var1 <= 0 or cfg.num_var2 <= 0:
        raise SizeError(f"RBM dims must be positive, got {cfg.num_var1}x{cfg.num_var2}")

    try:
        model = CleanEnergyQVAE(
            input_dimension=cfg.input_dimension,
            activation_fct=cfg.activation_fct,
            config=cfg,
        )
    except Exception as e:
        raise BuildError(f"Failed to build CleanEnergyQVAE: {e}") from e

    logger.info("[build] input=%d hidden=%d RBM=%dx%d params=%s",
              cfg.input_dimension, cfg.hidden_dim, cfg.num_var1, cfg.num_var2,
              f"{sum(p.numel() for p in model.parameters()):,}")
    return model
