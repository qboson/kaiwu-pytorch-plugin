# -*- coding: utf-8 -*-
"""Encoder/decoder networks for the clean energy-supervised QVAE."""
import torch
from torch import nn
import torch.nn.functional as F

from utils.logging import get_logger
from utils.exception import SizeError, ValueError

logger = get_logger("networks")


class Network(nn.Module):
    """Base class for all encoder/decoder networks."""

    def __init__(self, node_sequence=None, activation_fct=None, create_module_list=True, **kwargs):
        super().__init__(**kwargs)
        self._layers = nn.ModuleList([]) if create_module_list else None
        self._node_sequence = node_sequence
        self._activation_fct = activation_fct

        if self._node_sequence and create_module_list:
            self._create_network()

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

    def _create_network(self):
        for node in self._node_sequence:
            self._layers.append(nn.Linear(node[0], node[1]))

    def get_activation_fct(self):
        return f"{self._activation_fct}".replace("()", "")


class BasicEncoder(Network):
    """Encoder with linear layers and activation function (for VAE baselines)."""

    def __init__(self, weight_decay=0.0, **kwargs):
        super().__init__(**kwargs)
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        for layer in self._layers:
            if self._activation_fct:
                x = self._activation_fct(layer(x))
            else:
                x = layer(x)
        return x

    def decode(self, x):
        raise NotImplementedError("Decoder not implemented for encoder")

    def get_weight_decay(self) -> torch.Tensor:
        if self.weight_decay == 0.0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        wd = sum(torch.sum(layer.weight ** 2) for layer in self._layers
                 if isinstance(layer, nn.Linear))
        return self.weight_decay * wd


class BasicDecoder(Network):
    """Decoder with linear layers and optional output activation."""

    def __init__(self, output_activation_fct=None, weight_decay=0.0, **kwargs):
        super().__init__(**kwargs)
        self._output_activation_fct = output_activation_fct
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.decode(x)

    def decode(self, x):
        nr_layers = len(self._layers)
        for idx, layer in enumerate(self._layers):
            if idx == nr_layers - 1 and self._output_activation_fct:
                x = self._output_activation_fct(layer(x))
            else:
                x = self._activation_fct(layer(x))
        return x

    def encode(self, x):
        raise NotImplementedError("Encoder not implemented for decoder")

    def get_weight_decay(self) -> torch.Tensor:
        if self.weight_decay == 0.0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        wd = sum(torch.sum(layer.weight ** 2) for layer in self._layers
                 if isinstance(layer, nn.Linear))
        return self.weight_decay * wd


class SimpleEncoder(Network):
    """Simplified encoder for discrete-latent VAE baselines."""

    def __init__(self, smoothing_distribution=None, **kwargs):
        super().__init__(**kwargs)
        self.smoothing_distribution = smoothing_distribution
        self.num_latent_hierarchy_levels = 4
        self.num_latent_units = 100
        self.num_det_units = 200
        self.num_det_layers = 2

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        for layer in self._layers:
            if self._activation_fct:
                x = self._activation_fct(layer(x))
            else:
                x = layer(x)
        return x

    def decode(self, x):
        raise NotImplementedError("Decoder not implemented for encoder")


class SimpleDecoder(Network):
    """Simplified decoder for discrete-latent VAE baselines."""

    def __init__(self, output_activation_fct=None, **kwargs):
        super().__init__(**kwargs)
        self._output_activation_fct = output_activation_fct

    def forward(self, z):
        return self.decode(z)

    def decode(self, x):
        nr_layers = len(self._layers)
        x_prime = None
        for idx, layer in enumerate(self._layers):
            if idx == nr_layers - 1:
                if self._output_activation_fct:
                    x_prime = self._output_activation_fct(layer(x))
                else:
                    x_prime = self._activation_fct(layer(x))
            else:
                x = self._activation_fct(layer(x))
        return x_prime

    def encode(self, x):
        raise NotImplementedError("Encoder not implemented for decoder")


class Decoder(BasicDecoder):
    """Sequential decoder built from node_sequence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._network = self._create_network()

    def _create_network(self):
        module_layers = []
        for idx, (n_in, n_out) in enumerate(self._node_sequence):
            module_layers.append(nn.Linear(n_in, n_out))
            act_fct = (self._output_activation_fct
                       if idx == len(self._node_sequence) - 1 else self._activation_fct)
            module_layers.append(act_fct)
        return nn.Sequential(*module_layers)

    def decode(self, x):
        return self._network(x)

    def encode(self, x):
        raise NotImplementedError("Encoder not implemented for decoder")


class ResidualMLPBlock(nn.Module):
    """Residual MLP block: Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = F.gelu(self.norm1(self.fc1(x)))
        hidden = self.dropout(hidden)
        hidden = self.norm2(self.fc2(hidden))
        return F.gelu(hidden + residual)


class CleanEncoder(Network):
    """Deterministic encoder: input -> RBM visible logits."""

    def __init__(self, input_dim, hidden_dim, latent_dim,
                 activation_fct=None, backbone_type="residual", weight_decay=1e-4, **kwargs):
        super().__init__(create_module_list=False, **kwargs)
        if input_dim <= 0:
            raise SizeError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise SizeError(f"hidden_dim must be positive, got {hidden_dim}")
        if latent_dim <= 0:
            raise SizeError(f"latent_dim must be positive, got {latent_dim}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.weight_decay = weight_decay
        if backbone_type not in {"mlp", "residual"}:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")
        self.backbone_type = backbone_type

        if self.backbone_type == "mlp":
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim))
        else:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)
            self.res_block = ResidualMLPBlock(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, latent_dim)
        logger.debug("[encoder] in=%d hidden=%d latent=%d backbone=%s",
                     input_dim, hidden_dim, latent_dim, self.backbone_type)

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        if self.backbone_type == "mlp":
            return self.fc(x)
        h = F.gelu(self.input_norm(self.input_proj(x)))
        h = self.res_block(h)
        return self.output_proj(h)

    def decode(self, x):
        raise NotImplementedError

    def get_weight_decay(self):
        if self.weight_decay == 0.0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        wd = sum(torch.sum(m.weight ** 2) for m in self.modules() if isinstance(m, nn.Linear))
        return self.weight_decay * wd


class CleanDecoder(Network):
    """Deterministic decoder: DVAE++ zeta -> input reconstruction."""

    def __init__(self, input_dim, hidden_dim, latent_dim,
                 activation_fct=None, backbone_type="residual", weight_decay=1e-4, **kwargs):
        super().__init__(create_module_list=False, **kwargs)
        if input_dim <= 0:
            raise SizeError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise SizeError(f"hidden_dim must be positive, got {hidden_dim}")
        if latent_dim <= 0:
            raise SizeError(f"latent_dim must be positive, got {latent_dim}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.weight_decay = weight_decay
        if backbone_type not in {"mlp", "residual"}:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")
        self.backbone_type = backbone_type

        if self.backbone_type == "mlp":
            self.fc = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, input_dim))
        else:
            self.input_proj = nn.Linear(latent_dim, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)
            self.res_block = ResidualMLPBlock(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, input_dim)
        logger.debug("[decoder] latent=%d hidden=%d out=%d backbone=%s",
                     latent_dim, hidden_dim, input_dim, self.backbone_type)

    def forward(self, x):
        return self.decode(x)

    def decode(self, x):
        if self.backbone_type == "mlp":
            return self.fc(x)
        h = F.gelu(self.input_norm(self.input_proj(x)))
        h = self.res_block(h)
        return self.output_proj(h)

    def encode(self, x):
        raise NotImplementedError

    def get_weight_decay(self):
        if self.weight_decay == 0.0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        wd = sum(torch.sum(m.weight ** 2) for m in self.modules() if isinstance(m, nn.Linear))
        return self.weight_decay * wd
