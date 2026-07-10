# -*- coding: utf-8 -*-
"""
Autoencoders module containing various encoder/decoder network definitions.
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

#Base Class
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
        """Encode input x. Must be implemented in subclasses."""
        raise NotImplementedError

    def decode(self, x):
        """Decode latent representation x. Must be implemented in subclasses."""
        raise NotImplementedError

    def _create_network(self):
        """Create linear layers from node_sequence."""
        for node in self._node_sequence:
            self._layers.append(nn.Linear(node[0], node[1]))

    def get_activation_fct(self):
        """Return string representation of activation function."""
        return f"{self._activation_fct}".replace("()", "")

#Implements encoder
class BasicEncoder(Network):
    """Encoder with linear layers and activation function."""

    def __init__(self, weight_decay=0.0, **kwargs):
        super().__init__(**kwargs)
        self.weight_decay = weight_decay

    def forward(self, x):
        """Forward pass: encode input."""
        return self.encode(x)

    def encode(self, x):
        """Encode input through layers."""
        logger.debug("encode")
        for layer in self._layers:
            if self._activation_fct:
                x = self._activation_fct(layer(x))
            else:
                x = layer(x)
        return x

    def decode(self, x):
        """Decode not implemented for encoder."""
        raise NotImplementedError("Decoder not implemented for encoder")

    def get_weight_decay(self) -> torch.Tensor:
        """
        Compute L2 regularization loss for all linear layers.

        Returns:
            torch.Tensor: Weight decay loss (0.0 if weight_decay == 0.0).
        """
        if self.weight_decay == 0.0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        wd = 0.0
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                wd += torch.sum(layer.weight ** 2)
        return self.weight_decay * wd

#Implements decoder
class BasicDecoder(Network):
    """Decoder with linear layers and optional output activation."""

    def __init__(self, output_activation_fct=None, weight_decay=0.0, **kwargs):
        super().__init__(**kwargs)
        self._output_activation_fct = output_activation_fct
        self.weight_decay = weight_decay

    def forward(self, x):
        """Forward pass: decode input."""
        return self.decode(x)

    def decode(self, x):
        """Decode latent representation through layers."""
        logger.debug("Decoder::decode")
        nr_layers = len(self._layers)
        for idx, layer in enumerate(self._layers):
            if idx == nr_layers - 1 and self._output_activation_fct:
                x = self._output_activation_fct(layer(x))
            else:
                x = self._activation_fct(layer(x))
        return x

    def encode(self, x):
        """Encode not implemented for decoder."""
        raise NotImplementedError("Encoder not implemented for decoder")

    def get_weight_decay(self) -> torch.Tensor:
        """Compute L2 regularization loss for all linear layers."""
        if self.weight_decay == 0.0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        wd = 0.0
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                wd += torch.sum(layer.weight ** 2)
        return self.weight_decay * wd

class SimpleEncoder(Network):
    """Simplified encoder with fixed architecture."""

    def __init__(self, smoothing_distribution=None, **kwargs):
        super().__init__(**kwargs)
        self.smoothing_distribution = smoothing_distribution
        self.num_latent_hierarchy_levels = 4
        self.num_latent_units = 100
        self.num_det_units = 200
        self.num_det_layers = 2

    def forward(self, x):
        """Forward pass: encode input."""
        return self.encode(x)

    def encode(self, x):
        """Encode input through layers."""
        logger.debug("encode")
        for layer in self._layers:
            if self._activation_fct:
                x = self._activation_fct(layer(x))
            else:
                x = layer(x)
        return x

    def decode(self, x):
        """Decode not implemented for encoder."""
        raise NotImplementedError("Decoder not implemented for encoder")

    def hierarchical_posterior(self, x, is_training=True):
        """
        Define a hierarchical approximate posterior distribution.

        Args:
            x: Input tensor.
            is_training: Whether in training mode (unused).

        Returns:
            tuple: (posterior list, post_samples list)
        """
        # pylint: disable=unused-argument
        logger.debug("ERROR Encoder::hierarchical_posterior")
        posterior = []
        post_samples = []
        for _ in range(self.num_latent_hierarchy_levels):
            qprime = self.encode(x)
            sigmoid = nn.Sigmoid()
            q = sigmoid(qprime)
            rho = torch.rand(q.size())
            posterior_dist = self.smoothing_distribution
            samples = posterior_dist.icdf(rho, q)
            posterior.append(posterior_dist)
            post_samples.append(samples)
        return posterior, post_samples


class SimpleDecoder(Network):
    """Simplified decoder with optional output activation."""

    def __init__(self, output_activation_fct=None, **kwargs):
        super().__init__(**kwargs)
        self._output_activation_fct = output_activation_fct

    def forward(self, z):
        """Forward pass: decode latent."""
        return self.decode(z)

    def decode(self, x):
        """Decode latent representation through layers."""
        logger.debug("Decoder::decode")
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
        """Encode not implemented for decoder."""
        raise NotImplementedError("Encoder not implemented for decoder")

    def decode_posterior_sample(self, zeta):
        """Decode a posterior sample."""
        logger.debug("Decoder::decode")
        nr_layers = len(self._layers)
        x_prime = None
        for idx, layer in enumerate(self._layers):
            if idx == nr_layers - 1:
                x_prime = self._output_activation_fct(layer(zeta))
            else:
                zeta = self._activation_fct(layer(zeta))
        return x_prime


class Decoder(BasicDecoder):
    """Alternative decoder using sequential network."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._network = self._create_network()

    def _create_network(self):
        """Create sequential network from node_sequence."""
        layers = self._node_sequence
        module_layers = []

        for idx, (n_in, n_out) in enumerate(layers):
            module_layers.append(nn.Linear(n_in, n_out))
            # Apply activation function: output activation for last layer
            act_fct = (self._output_activation_fct
                if idx == len(layers) - 1 else self._activation_fct)
            module_layers.append(act_fct)

        return nn.Sequential(*module_layers)

    def decode(self, x):
        """Decode posterior sample."""
        logger.debug("Decoder::decode")
        return self._network(x)

    def encode(self, x):
        """Encode not implemented for decoder."""
        raise NotImplementedError("Encoder not implemented for decoder")


if __name__ == "__main__":
    logger.debug("Testing Networks")
    nw = Network()
    encoder = SimpleEncoder()
    logger.debug(encoder._layers)
    decoder = SimpleDecoder()
    logger.debug(decoder._layers)
    decoder2 = Decoder()
    logger.debug("Success")
