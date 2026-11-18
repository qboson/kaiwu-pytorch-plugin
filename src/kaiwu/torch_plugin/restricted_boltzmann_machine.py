# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Restricted Boltzmann Machine"""
from numbers import Integral

import torch
from .abstract_boltzmann_machine import AbstractBoltzmannMachine


def _gibbs_count(name, value, minimum=0):
    """Validate Gibbs counts without truncating fractions or accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _gibbs_initial_state(name, state, width, parameter):
    """Validate binary chain initializers before converting to the model's precision."""
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] != width:
        raise ValueError(f"{name} must have nonempty shape (B, {width})")
    if not torch.all((state == 0) | (state == 1)):
        raise ValueError(f"{name} must contain binary states exactly equal to 0 or 1")
    return state.to(parameter)


class RestrictedBoltzmannMachine(AbstractBoltzmannMachine):
    """Create a Restricted Boltzmann Machine.

    Args:
        num_visible (int): Number of visible nodes in the model.

        num_hidden (int): Number of hidden nodes in the model.

        quadratic_coef (torch.FloatTensor, optional): quadratic coefficent,
            shape is [num_visible, num_hidden]

        linear_bias (torch.FloatTensor, optional): linear bias, shape is [num_hidden]

        device (torch.device, optional): Device to construct tensors.
    """

    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        quadratic_coef: torch.FloatTensor = None,
        linear_bias: torch.FloatTensor = None,
        device=None,
    ):
        super().__init__(device=device)
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_nodes = num_visible + num_hidden
        self.quadratic_coef = torch.nn.Parameter(
            quadratic_coef
            if quadratic_coef is not None
            else torch.randn((num_visible, num_hidden)).to(self.device) * 0.01
        )
        self.linear_bias = torch.nn.Parameter(
            linear_bias
            if linear_bias is not None
            else torch.zeros(num_hidden + num_visible).to(self.device)
        )

    @property
    def hidden_bias(self) -> torch.Tensor:
        """Return the hidden bias."""
        return self.linear_bias[self.num_visible :]

    @property
    def visible_bias(self) -> torch.Tensor:
        """Return the visible bias."""
        return self.linear_bias[: self.num_visible]

    def clip_parameters(self, h_range, j_range) -> None:
        """Clip linear and quadratic bias weights in-place.

        Args:
            h_range (tuple[float, float]): Range for quadratic weights. for example, [-1, 1]
            j_range (tuple[float, float]): Range for linear weights. for example, [-1, 1]
        """
        self.get_parameter("linear_bias").data.clamp_(*h_range)
        self.get_parameter("quadratic_coef").data.clamp_(*j_range)

    def get_hidden(
        self,
        s_visible: torch.Tensor,
        requires_grad: bool = False,
        bernoulli: bool = False,
    ) -> torch.Tensor:
        """Propagate visible spins to the hidden layer.

        Args:
            s_visible: Visible layer tensor.
            requires_grad: Whether to allow gradient backpropagation.
        """
        context = torch.enable_grad if requires_grad else torch.no_grad
        with context():
            s_all = torch.zeros(
                s_visible.size(0),
                self.num_hidden + self.num_visible,
                device=self.device,
            )
            s_all[:, : self.num_visible] = s_visible
            prob = torch.sigmoid(
                s_visible @ self.quadratic_coef + self.linear_bias[self.num_visible :]
            )
            if bernoulli:
                s_all[:, self.num_visible :] = (prob > torch.rand_like(prob)).float()
            else:
                s_all[:, self.num_visible :] = prob
            return s_all

    def get_visible(
        self, s_hidden: torch.Tensor, bernoulli: bool = False
    ) -> torch.Tensor:
        """Propagate hidden spins to the visible layer."""
        with torch.no_grad():
            s_all = torch.zeros(
                s_hidden.size(0), self.num_hidden + self.num_visible
            ).to(self.device)
            s_all[:, self.num_visible :] = s_hidden
            prob = torch.sigmoid(
                s_hidden @ self.quadratic_coef.t()
                + self.linear_bias[: self.num_visible]
            )

            if bernoulli:
                s_all[:, : self.num_visible] = (prob > torch.rand_like(prob)).float()
            else:
                s_all[:, : self.num_visible] = prob
            return s_all

    @torch.no_grad()
    def gibbs_sample(
        self,
        n_step: int,
        n_burnin: int = 0,
        s_visible: torch.Tensor = None,
        s_hidden: torch.Tensor = None,
        n_sample: int = None,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """Generate binary joint states with local PyTorch block Gibbs sampling.

        Each batch row starts a separate chain. A visible initializer updates
        hidden then visible units; a hidden initializer updates visible then
        hidden units. Both layers are resampled on every complete transition;
        initial states are never clamped. This finite-length MCMC approximation
        does not guarantee mixing or independent draws from the model.

        Args:
            n_step: Total complete transitions per chain, a nonnegative integer.
            n_burnin: Initial transitions to discard, a nonnegative integer.
                Discarding does not change transitions or random-number consumption.
            s_visible: Optional nonempty binary tensor of shape ``(B, num_visible)``.
            s_hidden: Optional nonempty binary tensor of shape ``(B, num_hidden)``.
                Provide at most one initializer. Inputs are converted to the current
                parameter dtype/device without modifying the caller's tensor.
            n_sample: Positive number of chains when no initializer is supplied;
                visible units then start from Bernoulli(0.5). If also supplied with
                an initializer, it must equal its batch size.
            generator: Optional PyTorch generator on the parameter device, used
                for initialization and every draw. None uses the default Torch RNG.

        Returns:
            torch.Tensor: Detached 0/1 states in the current parameter dtype/device,
                with visible columns followed by hidden columns. Rows are grouped
                by retained transition, with shape
                ``(max(n_step - n_burnin, 0) * B, num_nodes)``. Zero steps or fully
                discarded chains return an empty batch after the requested work.

        Raises:
            ValueError: If counts or binary initial states violate this contract.
            TypeError: If initial states or the generator have the wrong type.
        """
        n_step = _gibbs_count("n_step", n_step)
        n_burnin = _gibbs_count("n_burnin", n_burnin)
        if n_sample is not None:
            n_sample = _gibbs_count("n_sample", n_sample, minimum=1)
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        if s_visible is not None and s_hidden is not None:
            raise ValueError("Provide at most one of s_visible and s_hidden")

        visible_first = s_hidden is None
        if s_visible is None and s_hidden is None:
            if n_sample is None:
                raise ValueError("n_sample is required without an initial state")
            states = torch.bernoulli(
                self.quadratic_coef.new_full((n_sample, self.num_visible), 0.5),
                generator=generator,
            )
        else:
            states = _gibbs_initial_state(
                "s_visible" if visible_first else "s_hidden",
                s_visible if visible_first else s_hidden,
                self.num_visible if visible_first else self.num_hidden,
                self.quadratic_coef,
            )
            if n_sample is not None and n_sample != states.shape[0]:
                raise ValueError("n_sample must equal the initial state's batch size")

        samples = []
        for step in range(n_step):
            if visible_first:
                hidden = torch.bernoulli(
                    torch.sigmoid(states @ self.quadratic_coef + self.hidden_bias),
                    generator=generator,
                )
                visible = torch.bernoulli(
                    torch.sigmoid(hidden @ self.quadratic_coef.t() + self.visible_bias),
                    generator=generator,
                )
                states = visible
            else:
                visible = torch.bernoulli(
                    torch.sigmoid(states @ self.quadratic_coef.t() + self.visible_bias),
                    generator=generator,
                )
                hidden = torch.bernoulli(
                    torch.sigmoid(visible @ self.quadratic_coef + self.hidden_bias),
                    generator=generator,
                )
                states = hidden
            if step >= n_burnin:
                samples.append(torch.cat((visible, hidden), dim=1))
        if not samples:
            return self.quadratic_coef.new_empty((0, self.num_nodes))
        return torch.cat(samples, dim=0)

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian.

        Args:
            s_all (torch.tensor): Tensor of shape (B, N), where B is the batch size,
                and N is the number of variables in the model.

        Returns:
            torch.tensor: Hamiltonian of shape (B,).
        """
        tmp = s_all[:, : self.num_visible].matmul(self.quadratic_coef)
        return -s_all @ self.linear_bias - torch.sum(
            tmp * s_all[:, self.num_visible :], dim=-1
        )

    def _to_ising_matrix(self):
        """Convert the Restricted Boltzmann Machine to Ising format."""
        num_nodes = self.linear_bias.shape[-1]
        with torch.no_grad():
            ising_mat = torch.zeros((num_nodes + 1, num_nodes + 1), device=self.device)
            # Restricted Boltzmann Machine: only connections between visible and hidden layers
            ising_mat[: self.num_visible, self.num_visible : -1] = (
                self.quadratic_coef / 8
            )
            ising_mat[self.num_visible : -1, : self.num_visible] = (
                self.quadratic_coef.t() / 8
            )
            ising_bias = self.linear_bias / 4 + ising_mat.sum(dim=0)[:-1]
            ising_mat[:num_nodes, -1] = ising_bias
            ising_mat[-1, :num_nodes] = ising_bias
            return ising_mat.detach().cpu().numpy()
