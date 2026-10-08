"""Configuration dataclass for QVAE-Anomaly.

Serves two roles:
- External parameter surface for the training / evaluation scripts.
- Attribute bag passed to the kaiwu base class ``QVAE(config=...)``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn


def _default_device() -> str:
    """Auto-select cuda if available, else cpu."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    # 输入与结构
    input_dimension: int = 1280
    hidden_dim: int = 256
    num_var1: int = 200
    num_var2: int = 200
    backbone_type: str = "residual"
    encoder_hidden_nodes: List[int] = field(default_factory=lambda: [256])
    decoder_hidden_nodes: List[int] = field(default_factory=lambda: [256])

    # RBM / 采样
    bm_type: str = "rbm"
    sampler_type: str = "sa"
    num_latent_units: int = 200  # = num_var1

    # VAE 正则
    kl_beta: float = 1e-4
    dist_beta: float = 10.0
    loss_type: str = "mse"

    # 能量监督
    energy_score_type: str = "free"
    anomaly_loss_type: str = "infonce"
    infonce_anchor_mode: str = "both"
    infonce_temperature: float = 1.0
    lambda_anom: float = 1.0
    energy_margin: float = 0.0
    negative_phase_mode: str = "sampled"  # 默认 sampled（SA 负相位）
    recon_loss_weight: float = 1.0
    anomaly_margin_weight: float = 1.0
    anomaly_infonce_weight: float = 1.0
    bm_weight_decay_weight: float = 1.0
    weight_decay: float = 0.01

    # 训练管道
    batch_size: int = 512
    lr: float = 5e-4
    epochs: int = 20
    seed: int = 42
    device: str = field(default_factory=_default_device)
    out_dir: str = "results"

    # 派生
    activation_fct: nn.Module = field(default_factory=nn.ReLU)

    def __post_init__(self):
        from utils.exception import ValueError, SizeError
        self.num_latent_units = self.num_var1

        # Validate key fields
        if self.input_dimension <= 0:
            raise SizeError(f"input_dimension must be positive, got {self.input_dimension}")
        if self.hidden_dim <= 0:
            raise SizeError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_var1 <= 0 or self.num_var2 <= 0:
            raise SizeError(f"RBM dims must be positive, got {self.num_var1}x{self.num_var2}")
        if self.kl_beta < 0:
            raise ValueError(f"kl_beta must be non-negative, got {self.kl_beta}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
