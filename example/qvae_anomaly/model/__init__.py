"""Model package for QVAE-Anomaly.

Exposes the public surface used by training / evaluation scripts:

- :class:`CleanEnergyQVAE` — the energy-supervised QVAE model.
- :func:`build_model` — factory that constructs it from a :class:`Config`.
- :class:`Config` — dataclass of hyperparameters.
- :class:`CleanEncoder` / :class:`CleanDecoder` / :class:`ResidualMLPBlock`
  — the MLP backbone blocks, exported for inspection / subclassing.
"""
from .model import CleanEnergyQVAE, build_model
from .networks import CleanEncoder, CleanDecoder, ResidualMLPBlock
from .config import Config

__all__ = ["CleanEnergyQVAE", "CleanEncoder", "CleanDecoder", "ResidualMLPBlock", "Config",
           "build_model"]
