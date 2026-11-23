# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Deep Belief Network (DBN) model.

This module contains the DBN class and functions for training the DBN+model
or only the model. Training the DBN+model will save the likelihood values
and prediction accuracy during the training process.
"""
import numpy as np


import torch
from torch import nn

from .restricted_boltzmann_machine import RestrictedBoltzmannMachine


# =================== Unsupervised DBN General Model =====================
class UnsupervisedDBN(nn.Module):
    """A general unsupervised Deep Belief Network (DBN) architecture.

    This model is a stack of Restricted Boltzmann Machines (RBMs).

    Args:
        hidden_layers_structure (list, optional): A list of integers
            representing the number of hidden units in each layer.
            Defaults to [100, 100].
    """

    def __init__(self, hidden_layers_structure=None):
        super().__init__()
        self.hidden_layers_structure = (
            hidden_layers_structure
            if hidden_layers_structure is not None
            else [100, 100]
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rbm_layers = None
        self.input_dim = None
        self._is_trained = False

    def create_rbm_layer(self, input_dim):
        """Creates the layers of RBMs for the DBN.

        Args:
            input_dim (int): The dimension of the input data (number of visible units).

        Returns:
            UnsupervisedDBN: The instance itself with the RBM layers created.
        """
        self.input_dim = input_dim
        self.rbm_layers = nn.ModuleList()

        current_dim = input_dim
        for n_hidden in self.hidden_layers_structure:
            rbm = RestrictedBoltzmannMachine(
                num_visible=current_dim,  # Number of visible units (feature dimension)
                num_hidden=n_hidden,  # Number of hidden units
            ).to(
                self.device
            )  # Move model to specified device (CPU/GPU)
            self.rbm_layers.append(rbm)
            current_dim = n_hidden

        self._is_trained = False
        return self

    def forward(self, data_in):
        """Performs a forward pass to transform the input data.

        Args:
            data_in (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The transformed data after passing through all RBM layers.

        Raises:
            ValueError: If the model has not been built or trained yet.
        """
        if self.rbm_layers is None:
            raise ValueError("Model not built yet. Call create_rbm_layer first.")
        if not self._is_trained:
            raise ValueError(
                "Model not trained yet. Call mark_as_trained() after training."
            )

        data_in = data_in.astype(np.float32)
        for rbm in self.rbm_layers:
            with torch.no_grad():
                hidden_output = rbm.get_hidden(
                    torch.FloatTensor(data_in).to(self.device)
                )
                data_in = (
                    hidden_output[:, rbm.num_visible :].cpu().numpy()
                )  # Extract only the hidden part
        return data_in

    def transform(self, data_in):
        """An sklearn-compatible transform method.

        Args:
            data_in (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The transformed data.
        """
        return self.forward(data_in)

    def reconstruct(self, data_in, layer_index=0):
        """Reconstructs the input from a specified RBM layer.

        Args:
            data_in (numpy.ndarray): The input data to be reconstructed.

            layer_index (int, optional): The index of the RBM layer to use for reconstruction.
                Defaults to 0.

        Returns:
            numpy.ndarray: The reconstructed data.

        Raises:
            ValueError: If the model has no RBM layers or the layer index is out of range.
        """
        if self.rbm_layers is None or len(self.rbm_layers) == 0:
            raise ValueError("No RBM layers found. Please fit the model first.")

        if layer_index >= len(self.rbm_layers):
            raise ValueError(f"Layer index {layer_index} out of range.")

        rbm = self.rbm_layers[layer_index]
        return self.reconstruct_with_rbm(rbm, data_in, self.device)

    def mark_as_trained(self):
        """Marks the model as trained.

        Returns:
            UnsupervisedDBN: The instance itself.
        """
        self._is_trained = True
        return self

    def get_rbm_layer(self, index):
        """Gets the RBM layer at the specified index.

        Args:
            index (int): The index of the RBM layer.

        Returns:
            RestrictedBoltzmannMachine or None: The RBM layer if found, otherwise None.
        """
        if index < len(self.rbm_layers):
            return self.rbm_layers[index]
        return None

    @staticmethod
    def reconstruct_with_rbm(rbm, data_in, device=None):
        """Reconstructs data using a single RBM.

        Args:
            rbm (RestrictedBoltzmannMachine): The trained RBM model.

            data_in (numpy.ndarray): The input data.

            device (torch.device, optional): The device to perform computation on.
                If None, uses the RBM's device. Defaults to None.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
                - The reconstructed visible layer data.
                - The reconstruction error for each sample.
        """
        if device is None:
            device = rbm.device

        # Convert to PyTorch tensor
        data_in = torch.FloatTensor(data_in).to(device)

        with torch.no_grad():
            # Get hidden representation using RBM's get_hidden
            hidden_act = rbm.get_hidden(data_in)
            hidden_part = hidden_act[
                :, rbm.num_visible :
            ]  # Extract only the hidden part

            # Reconstruct visible layer (using transposed weights)
            visible_recon = torch.sigmoid(
                torch.matmul(hidden_part, rbm.quadratic_coef.t())
                + rbm.linear_bias[: rbm.num_visible]
            )

            # Calculate reconstruction error
            recon_errors = (
                torch.mean((data_in - visible_recon) ** 2, dim=1).cpu().numpy()
            )

        return visible_recon.cpu().numpy(), recon_errors

    @property
    def num_layers(self):
        """Returns the number of RBM layers.

        Returns:
            int: The number of layers.
        """
        return len(self.rbm_layers)

    @property
    def output_dim(self):
        """Returns the output dimension of the DBN.

        Returns:
            int: The dimension of the final hidden layer.
        """
        if len(self.rbm_layers) > 0:
            return self.rbm_layers[-1].num_hidden
        return self.input_dim
