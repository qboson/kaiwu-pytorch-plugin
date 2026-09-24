# -*- coding: utf-8 -*-
"""Loss functions and RBM statistics for the energy-supervised QVAE.

Depends only on torch/nn; does not import the model class.
"""
import torch
import torch.nn.functional as F
from kaiwu.torch_plugin.qvae_dist_util import MixtureGeneric


def _zero_scalar_like(values):
    """Create a scalar zero on the same device/dtype as a reference tensor."""
    return values.new_zeros(())


def _subset_mean_or_zero(values, mask):
    """Average a masked subset, returning differentiable zero when empty."""
    if mask.any():
        return values[mask].mean()
    return _zero_scalar_like(values)


def _bernoulli_entropy_from_probs(probs):
    probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)
    return -torch.sum(
        probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs), dim=1
    )


def _normalize_negative_phase_mode(mode):
    if mode in {None, "sampled", "enabled", "on"}:
        return "sampled"
    if mode in {"none", "disabled", "off"}:
        return "none"
    raise ValueError(f"Unsupported negative_phase_mode: {mode}")


def _normalize_energy_score_type(energy_score_type):
    if energy_score_type in {None, "expected", "expected_energy"}:
        return "expected"
    if energy_score_type in {"free", "free_energy"}:
        return "free"
    raise ValueError(f"Unsupported energy_score_type: {energy_score_type}")


def _normalize_anomaly_loss_type(anomaly_loss_type):
    if anomaly_loss_type in {None, "margin", "hinge", "margin_hinge"}:
        return "margin"
    if anomaly_loss_type in {"infonce", "info_nce", "nce"}:
        return "infonce"
    if anomaly_loss_type in {"margin_infonce", "infonce_margin", "both"}:
        return "margin_infonce"
    raise ValueError(f"Unsupported anomaly_loss_type: {anomaly_loss_type}")


def _normalize_infonce_anchor_mode(anchor_mode):
    if anchor_mode in {None, "normal", "normal_anchor"}:
        return "normal_anchor"
    if anchor_mode in {"anomaly", "anomaly_anchor"}:
        return "anomaly_anchor"
    if anchor_mode in {"both", "symmetric"}:
        return "both"
    raise ValueError(f"Unsupported infonce_anchor_mode: {anchor_mode}")


def _device_aware_dvae_reparameterize(posterior):
    """Draw the single DVAE++ decoder latent used by the maintained model."""
    q = torch.sigmoid(posterior.logit_mu)
    bernoulli_uniform = torch.rand_like(q)
    z = (bernoulli_uniform < q).to(q.dtype)
    beta = posterior.smoothing_dist.beta.to(device=q.device, dtype=q.dtype)
    smoothing_uniform = torch.rand_like(q)
    base_zeta = -torch.log(
        1.0 - (1.0 - torch.exp(-beta)) * smoothing_uniform
    ) / beta
    zeta = torch.where(z == 0.0, base_zeta, 1.0 - base_zeta)

    pdf_0 = posterior.smoothing_dist.pdf(zeta)
    pdf_1 = posterior.smoothing_dist.pdf(1.0 - zeta)
    cdf_0 = posterior.smoothing_dist.cdf(zeta)
    cdf_1 = 1.0 - posterior.smoothing_dist.cdf(1.0 - zeta)
    grad_q = ((cdf_0 - cdf_1) / (q * pdf_1 + (1.0 - q) * pdf_0)).detach()
    grad_term = grad_q * q
    grad_term = grad_term - grad_term.detach()
    return zeta.detach() + grad_term


def _sample_visible_decoder_latent(model, visible_logits):
    """Sample one DVAE++ latent for reconstruction; RBM energy remains PM-only."""
    posterior = MixtureGeneric(visible_logits, model.dist_beta)
    smoothing_beta = posterior.smoothing_dist.beta
    if torch.is_tensor(smoothing_beta):
        posterior.smoothing_dist.beta = smoothing_beta.to(
            device=visible_logits.device,
            dtype=visible_logits.dtype,
        )
    return _device_aware_dvae_reparameterize(posterior)


def _compute_clean_rbm_statistics(model, visible_logits, include_negative_reference=False):
    """Compute clean RBM statistics."""
    if model.bm.num_visible != visible_logits.shape[1]:
        raise ValueError(
            f"The number of visible variables in the Boltzmann machine {model.bm.num_visible} "
            f"does not match the encoder logits shape {visible_logits.shape[1]}."
        )

    visible_probs = torch.sigmoid(visible_logits)
    hidden_activation = visible_probs @ model.bm.quadratic_coef + model.bm.hidden_bias
    hidden_probs = torch.sigmoid(hidden_activation)

    visible_term = visible_probs @ model.bm.visible_bias
    hidden_term = hidden_probs @ model.bm.hidden_bias
    interaction_term = torch.sum(
        (visible_probs @ model.bm.quadratic_coef) * hidden_probs, dim=1
    )
    expected_energy_per_sample = -visible_term - hidden_term - interaction_term
    free_energy_per_sample = -visible_term - torch.sum(
        F.softplus(hidden_activation), dim=1
    )

    entropy_per_sample = _bernoulli_entropy_from_probs(visible_probs)

    stats = {
        "visible_probs": visible_probs,
        "hidden_probs": hidden_probs,
        "expected_energy_per_sample": expected_energy_per_sample,
        "free_energy_per_sample": free_energy_per_sample,
        "entropy_per_sample": entropy_per_sample,
    }
    if include_negative_reference:
        negative_phase_mode = _normalize_negative_phase_mode(
            getattr(model, "negative_phase_mode", "sampled")
        )
        if negative_phase_mode == "sampled":
            negative_particles = model.bm.sample(model.sampler).detach()
            negative_reference = model.bm(negative_particles).mean()
            model.negative_phase_sample_calls = int(
                getattr(model, "negative_phase_sample_calls", 0)
            ) + 1
        else:
            negative_reference = _zero_scalar_like(free_energy_per_sample)
        cross_entropy_per_sample = free_energy_per_sample - negative_reference
        stats.update(
            {
                "negative_phase_mode": negative_phase_mode,
                "negative_reference": negative_reference,
                "cross_entropy_per_sample": cross_entropy_per_sample,
                "kl_per_sample": cross_entropy_per_sample - entropy_per_sample,
            }
        )
    return stats


def _select_score_energy(kl_stats, energy_score_type):
    energy_score_type = _normalize_energy_score_type(energy_score_type)
    if energy_score_type == "free":
        return kl_stats["free_energy_per_sample"]
    return kl_stats["expected_energy_per_sample"]


def _weighted_qvae_weight_decay_loss(model):
    wd_weight = float(getattr(model, "bm_weight_decay_weight", 1.0))
    wd = 0.0
    if hasattr(model.bm, "quadratic_coef"):
        wd += torch.sum(model.bm.quadratic_coef ** 2)
    if hasattr(model.bm, "linear_bias"):
        wd += 0.5 * torch.sum(model.bm.linear_bias ** 2)
    return wd_weight * float(model.weight_decay) * wd


def _energy_infonce_loss(
    score_energy_per_sample,
    normal_mask,
    anomaly_mask,
    temperature=1.0,
    anchor_mode="normal_anchor",
):
    """Contrast energies so labeled anomalies are higher-energy than normals."""
    if not normal_mask.any() or not anomaly_mask.any():
        return _zero_scalar_like(score_energy_per_sample)

    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("infonce_temperature must be positive.")
    anchor_mode = _normalize_infonce_anchor_mode(anchor_mode)

    normal_energy = score_energy_per_sample[normal_mask]
    anomaly_energy = score_energy_per_sample[anomaly_mask]

    losses = []
    if anchor_mode in {"normal_anchor", "both"}:
        normal_positive_logits = -normal_energy[:, None] / temperature
        anomaly_negative_logits = -anomaly_energy[None, :] / temperature
        anomaly_negative_logits = anomaly_negative_logits.expand(
            normal_energy.shape[0], anomaly_energy.shape[0]
        )
        normal_logits = torch.cat([normal_positive_logits, anomaly_negative_logits], dim=1)
        normal_targets = torch.zeros(
            normal_energy.shape[0], dtype=torch.long, device=score_energy_per_sample.device
        )
        losses.append(F.cross_entropy(normal_logits, normal_targets))

    if anchor_mode in {"anomaly_anchor", "both"}:
        anomaly_positive_logits = anomaly_energy[:, None] / temperature
        normal_negative_logits = normal_energy[None, :] / temperature
        normal_negative_logits = normal_negative_logits.expand(
            anomaly_energy.shape[0], normal_energy.shape[0]
        )
        anomaly_logits = torch.cat([anomaly_positive_logits, normal_negative_logits], dim=1)
        anomaly_targets = torch.zeros(
            anomaly_energy.shape[0], dtype=torch.long, device=score_energy_per_sample.device
        )
        losses.append(F.cross_entropy(anomaly_logits, anomaly_targets))

    return torch.stack(losses).mean()


def _compose_energy_supervised_qvae_loss_terms(
    recon_per_sample,
    cross_entropy_per_sample,
    entropy_per_sample,
    expected_energy_per_sample,
    score_energy_per_sample,
    wd_loss,
    kl_beta,
    lambda_anom=1.0,
    labels=None,
    energy_margin=0.0,
    anomaly_loss_type="margin",
    infonce_temperature=1.0,
    infonce_anchor_mode="normal_anchor",
    anomaly_margin_weight=1.0,
    anomaly_infonce_weight=1.0,
    recon_loss_weight=1.0,
):
    """Compose the energy-supervised QVAE loss."""
    if labels is None:
        kl_per_sample = cross_entropy_per_sample - entropy_per_sample
        recon_loss = recon_per_sample.mean()
        kl_loss = kl_per_sample.mean()
        gen_loss = recon_loss_weight * recon_loss + kl_beta * kl_loss + wd_loss
        return {
            "total_loss": gen_loss,
            "gen_loss": gen_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "wd_loss": wd_loss,
            "cls_loss": None,
        }

    labels = labels.view(-1)
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    if (~(normal_mask | anomaly_mask)).any():
        raise ValueError("Energy-supervised QVAE expects binary labels encoded as 0/1.")
    anomaly_loss_type = _normalize_anomaly_loss_type(anomaly_loss_type)

    normal_recon_loss = _subset_mean_or_zero(recon_per_sample, normal_mask)
    normal_cross_entropy_loss = _subset_mean_or_zero(cross_entropy_per_sample, normal_mask)
    normal_entropy_loss = _subset_mean_or_zero(entropy_per_sample, normal_mask)
    normal_kl_loss = normal_cross_entropy_loss - normal_entropy_loss

    anomaly_expected_energy_loss = _subset_mean_or_zero(expected_energy_per_sample, anomaly_mask)
    anomaly_score_energy_loss = _subset_mean_or_zero(score_energy_per_sample, anomaly_mask)
    anomaly_margin_per_sample = F.relu(energy_margin - score_energy_per_sample)
    anomaly_margin_loss = _subset_mean_or_zero(anomaly_margin_per_sample, anomaly_mask)
    anomaly_infonce_loss = _energy_infonce_loss(
        score_energy_per_sample,
        normal_mask,
        anomaly_mask,
        temperature=infonce_temperature,
        anchor_mode=infonce_anchor_mode,
    )

    recon_loss_weight = float(recon_loss_weight)
    recon_loss = normal_recon_loss
    kl_loss = normal_kl_loss
    weighted_recon_loss = recon_loss_weight * recon_loss
    weighted_normal_kl_loss = kl_beta * normal_kl_loss
    anomaly_margin_weight = float(anomaly_margin_weight)
    anomaly_infonce_weight = float(anomaly_infonce_weight)
    if anomaly_loss_type == "margin":
        anomaly_loss = anomaly_margin_weight * anomaly_margin_loss
        weighted_anomaly_margin_loss = lambda_anom * anomaly_margin_weight * anomaly_margin_loss
        weighted_anomaly_infonce_loss = _zero_scalar_like(anomaly_infonce_loss)
    elif anomaly_loss_type == "infonce":
        anomaly_loss = anomaly_infonce_weight * anomaly_infonce_loss
        weighted_anomaly_margin_loss = _zero_scalar_like(anomaly_margin_loss)
        weighted_anomaly_infonce_loss = lambda_anom * anomaly_infonce_weight * anomaly_infonce_loss
    else:
        anomaly_loss = (
            anomaly_margin_weight * anomaly_margin_loss
            + anomaly_infonce_weight * anomaly_infonce_loss
        )
        weighted_anomaly_margin_loss = lambda_anom * anomaly_margin_weight * anomaly_margin_loss
        weighted_anomaly_infonce_loss = lambda_anom * anomaly_infonce_weight * anomaly_infonce_loss
    weighted_anomaly_loss = weighted_anomaly_margin_loss + weighted_anomaly_infonce_loss
    gen_loss = weighted_recon_loss + weighted_normal_kl_loss + weighted_anomaly_loss + wd_loss

    return {
        "total_loss": gen_loss,
        "gen_loss": gen_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "wd_loss": wd_loss,
        "cls_loss": None,
        "normal_recon_loss": normal_recon_loss,
        "weighted_recon_loss": weighted_recon_loss,
        "normal_cross_entropy_loss": normal_cross_entropy_loss,
        "normal_entropy_loss": normal_entropy_loss,
        "normal_kl_loss": normal_kl_loss,
        "anomaly_expected_energy_loss": anomaly_expected_energy_loss,
        "anomaly_score_energy_loss": anomaly_score_energy_loss,
        "anomaly_margin_loss": anomaly_margin_loss,
        "anomaly_infonce_loss": anomaly_infonce_loss,
        "anomaly_loss": anomaly_loss,
        "weighted_normal_kl_loss": weighted_normal_kl_loss,
        "weighted_anomaly_margin_loss": weighted_anomaly_margin_loss,
        "weighted_anomaly_infonce_loss": weighted_anomaly_infonce_loss,
        "weighted_anomaly_loss": weighted_anomaly_loss,
        "energy_margin": energy_margin,
        "lambda_anom": lambda_anom,
        "recon_loss_weight": recon_loss_weight,
        "anomaly_loss_type": anomaly_loss_type,
        "infonce_temperature": infonce_temperature,
        "infonce_anchor_mode": _normalize_infonce_anchor_mode(infonce_anchor_mode),
        "anomaly_margin_weight": anomaly_margin_weight,
        "anomaly_infonce_weight": anomaly_infonce_weight,
        "normal_fraction": normal_mask.float().mean(),
        "anomaly_fraction": anomaly_mask.float().mean(),
    }
