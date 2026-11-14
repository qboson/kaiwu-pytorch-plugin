"""PyTorch parameters and energy for the discrete Ising EP network.

Clamped inputs feed a stack of hidden spin layers and one output layer. The
widths come from ``hidden_sizes`` plus ``n_classes * spins_per_class`` output
spins. Learnable parameters are one weight matrix and one bias per connection,
stored in ``weights`` and ``biases``. :meth:`IsingEPModel.qubo` turns the same
network into the upper-triangular QUBO Kaiwu consumes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def _hidden_widths(hidden_sizes: Sequence[int] | int) -> tuple[int, ...]:
    """Normalise a hidden-width specification to a tuple of positive integers.

    Args:
        hidden_sizes: One width, or a sequence of widths from the first hidden
            layer through the last. An ``int`` is a single hidden layer.

    Returns:
        The hidden widths in order.

    Raises:
        ValueError: If no width is given or any width is not positive.
    """
    if isinstance(hidden_sizes, int):
        widths = (int(hidden_sizes),)
    else:
        widths = tuple(int(size) for size in hidden_sizes)
    if not widths or any(size <= 0 for size in widths):
        raise ValueError(f"hidden_sizes must be positive integers, got {widths}")
    return widths


def _init_gains(gains: Sequence[float] | None, n_connections: int) -> tuple[float, ...]:
    """Normalise per-connection init scales, defaulting every gain to 1.

    Args:
        gains: One positive scale per weight matrix, input connection first.
            ``None`` means every gain is 1.
        n_connections: Number of weight matrices, ``len(hidden_sizes) + 1``.

    Returns:
        The per-connection gains in order.

    Raises:
        ValueError: If the length does not match ``n_connections`` or a gain
            is not strictly positive.
    """
    if gains is None:
        return (1.0,) * int(n_connections)
    values = tuple(float(gain) for gain in gains)
    if len(values) != int(n_connections):
        raise ValueError(
            f"init_gains must have length {n_connections}, got {len(values)}"
        )
    if any(gain <= 0.0 for gain in values):
        raise ValueError(f"init_gains must be strictly positive, got {values}")
    return values


def _build_qubo(
    couplings: Sequence[Any],
    fields: Sequence[Any],
    beta: float,
) -> tuple[torch.Tensor, float]:
    """Rewrite a stacked ``F = E + beta * C`` as an upper-triangular QUBO.

    With spin layers ``s_0, ..., s_{K-1}`` (the last one is the output), the
    local fields ``H_l`` and the spin-spin couplings ``W_l`` between layer
    ``l - 1`` and ``l``, the nudged EP objective collapses to::

        F = -sum_{l>=1} s_{l-1} W_l s_l - sum_l H_l . s_l + beta * n_output

    The clamped input never appears here: it is already inside ``H_0``.
    Substituting ``s = 2q - 1`` and using ``q**2 == q`` for binary variables
    moves all linear terms onto the diagonal of ``Q``. A single hidden layer is
    ``couplings = [V]`` and ``fields = [H_h, H_o]``.

    Args:
        couplings: One weight matrix per adjacent pair of spin layers, of shape
            ``(width_l, width_{l+1})``. Empty for a single spin layer.
        fields: Local field of every spin layer, output last. The nudge
            ``beta * target`` is already part of the last field.
        beta: Nudging coefficient. It enters ``Q`` only through the last field;
            here it contributes the ``beta * n_output`` offset.

    Returns:
        A tuple ``(Q, constant)`` where ``Q`` is an upper-triangular
        ``float64`` matrix and ``constant`` is the scalar offset such that
        ``q @ Q @ q + constant == F`` for ``q = (s + 1) / 2``.

    Raises:
        ValueError: If ``fields`` is empty or a coupling does not match the
            widths of the two layers it joins.
    """
    local = [
        torch.as_tensor(field).detach().to(dtype=torch.float64).reshape(-1) for field in fields
    ]
    if not local:
        raise ValueError("the QUBO needs at least one local field")
    if len(couplings) != len(local) - 1:
        raise ValueError(
            f"expected {len(local) - 1} couplings for {len(local)} fields, "
            f"got {len(couplings)}"
        )

    widths = [int(field.shape[0]) for field in local]
    starts = [sum(widths[:index]) for index in range(len(widths))]
    matrix = torch.zeros((sum(widths), sum(widths)), dtype=torch.float64)
    offset = torch.zeros((), dtype=torch.float64) + float(beta) * widths[-1]

    for index, coupling in enumerate(couplings):
        weight = torch.as_tensor(coupling).detach().to(dtype=torch.float64)
        left, right = starts[index], starts[index + 1]
        left_width, right_width = widths[index], widths[index + 1]
        if tuple(weight.shape) != (left_width, right_width):
            raise ValueError(
                f"coupling {index} must have shape {(left_width, right_width)}, "
                f"got {tuple(weight.shape)}"
            )
        # Diagonals carry the linear coefficients because q**2 == q.
        matrix[left : left + left_width, left : left + left_width].diagonal().add_(
            2.0 * weight.sum(dim=1)
        )
        matrix[right : right + right_width, right : right + right_width].diagonal().add_(
            2.0 * weight.sum(dim=0)
        )
        # Upper-triangular cross block: (2q_j - 1)(2q_k - 1) expands to 4 q_j q_k.
        matrix[left : left + left_width, right : right + right_width] = -4.0 * weight
        offset = offset - weight.sum()

    for start, width, field in zip(starts, widths, local):
        matrix[start : start + width, start : start + width].diagonal().add_(-2.0 * field)
        offset = offset + field.sum()
    return matrix, float(offset.item())


# `forward` is intentionally not implemented: this module is an energy function
# evaluated on solver-supplied spins, not a feed-forward network.
class IsingEPModel(nn.Module):  # pylint: disable=abstract-method
    """Ising equilibrium-propagation network with a stack of linear layers.

    Attributes:
        n_in: Number of clamped input units.
        hidden_sizes: Width of each hidden spin layer, in order.
        n_hidden: Total number of hidden spins.
        n_classes: Number of classes.
        spins_per_class: Output spins allocated to each class.
        n_output: Total output spins, ``n_classes * spins_per_class``.
        layer_sizes: ``(n_in, *hidden_sizes, n_output)``.
        weights: One ``(fan_in, fan_out)`` matrix per connection, input first.
        biases: One bias vector per spin layer, first hidden layer first.
        init_gains: Per-connection scale on the ``+-1/sqrt(fan_in)`` range.
    """

    def __init__(
        self,
        n_in: int = 784,
        hidden_sizes: Sequence[int] | int = (64,),
        n_classes: int = 10,
        spins_per_class: int = 4,
        seed: int = 0,
        *,
        init_gains: Sequence[float] | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        """Build a model with deterministic uniform initialization.

        Couplings are drawn uniformly from ``+-gain/sqrt(fan_in)``. A gain of 1
        is the usual scale that keeps the initial local fields ``O(1)``
        regardless of layer width, so the annealer starts from a
        well-conditioned energy landscape. Biases start at zero. Sampling uses
        a seeded CPU generator, so one seed fully determines the initial
        parameters. Layers are initialised from the input outward.

        Args:
            n_in: Number of clamped input units.
            hidden_sizes: Width of each hidden layer. An integer is one layer.
            n_classes: Number of classes.
            spins_per_class: Output spins per class.
            seed: Seed for the initialization generator.
            init_gains: One positive scale per weight matrix, input first.
                Defaults to 1 for every connection.
            dtype: Floating-point dtype for all parameters.
            device: Device holding the parameters. Defaults to CPU.

        Raises:
            ValueError: If ``hidden_sizes`` is empty or contains a non-positive
                width, or if ``init_gains`` has the wrong length or a
                non-positive value.
        """
        super().__init__()
        self.n_in = int(n_in)
        self.hidden_sizes = _hidden_widths(hidden_sizes)
        self.n_classes = int(n_classes)
        self.spins_per_class = int(spins_per_class)
        self.n_output = self.n_classes * self.spins_per_class
        self._dtype = dtype
        self._device = torch.device("cpu") if device is None else torch.device(device)

        fan_ins = (self.n_in, *self.hidden_sizes)
        fan_outs = (*self.hidden_sizes, self.n_output)
        self.init_gains = _init_gains(init_gains, len(fan_ins))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        weights: list[nn.Parameter] = []
        biases: list[nn.Parameter] = []
        for fan_in, fan_out, gain in zip(fan_ins, fan_outs, self.init_gains):
            weight = torch.empty((fan_in, fan_out), dtype=dtype)
            limit = float(gain) / math.sqrt(fan_in)
            weight.uniform_(-limit, limit, generator=generator)
            weights.append(nn.Parameter(weight.to(self._device)))
            biases.append(nn.Parameter(torch.zeros(fan_out, dtype=dtype, device=self._device)))
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    @property
    def n_hidden(self) -> int:
        """Total number of hidden spins across every hidden layer.

        Returns:
            The sum of :attr:`hidden_sizes`.
        """
        return sum(self.hidden_sizes)

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        """Width of every layer, including the clamped input and the output.

        Returns:
            ``(n_in, *hidden_sizes, n_output)``.
        """
        return (self.n_in, *self.hidden_sizes, self.n_output)

    @property
    def spin_widths(self) -> tuple[int, ...]:
        """Width of every spin layer that the solver is free to set.

        Returns:
            ``(*hidden_sizes, n_output)``.
        """
        return (*self.hidden_sizes, self.n_output)

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype shared by all parameters.

        Returns:
            The parameter dtype.
        """
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Device holding the parameters.

        Returns:
            The parameter device.
        """
        return self._device

    def _vector(self, value: Any) -> torch.Tensor:
        """Coerce one sample to a 1-D tensor in the model dtype and device.

        Args:
            value: Tensor or array-like holding one sample.

        Returns:
            A 1-D tensor in the model dtype and device.
        """
        return torch.as_tensor(value, dtype=self.dtype, device=self.device).reshape(-1)

    def parameter_snapshot(self) -> dict[str, torch.Tensor]:
        """Copy the learnable parameters, detached from autograd.

        Used to compare parameters before and after training, and to write
        checkpoints, without aliasing the live tensors.

        Returns:
            Independent tensors keyed ``weights.<i>`` and ``biases.<i>``.
        """
        return {name: value.detach().clone() for name, value in self.state_dict().items()}

    def target_spins(self, label: int) -> torch.Tensor:
        """Encode one class label as a target spin vector.

        Each class owns a contiguous block of ``spins_per_class`` output spins.
        The target sets that block to ``+1`` and everything else to ``-1``,
        which is what makes the EP cost linear in the output spins.

        Args:
            label: Class index in ``[0, n_classes)``.

        Returns:
            A ``(n_output,)`` tensor of ``+-1`` target spins.

        Raises:
            ValueError: If ``label`` is outside the class range.
        """
        index = int(label)
        if not 0 <= index < self.n_classes:
            raise ValueError(f"label must be in [0, {self.n_classes}), got {index}")
        target = torch.full((self.n_output,), -1.0, dtype=self.dtype, device=self.device)
        start = index * self.spins_per_class
        target[start : start + self.spins_per_class] = 1.0
        return target

    def energy(self, x: Any, states: Sequence[Any]) -> torch.Tensor:
        """Compute the stacked Ising energy.

        With spin layers ``s_0, ..., s_{K-1}`` (the last one is the output)::

            E = -x W_0 s_0 - sum_{l=1}^{K-1} s_{l-1} W_l s_l - sum_l b_l . s_l

        The training loss differentiates this energy,
        ``(E_nudged - E_free) / beta``. The cost is not part of ``E``; the
        solver objective adds ``beta * cost``, and that cost does not depend on
        the parameters, so both expressions share a parameter gradient. The
        result keeps its graph with respect to the parameters while treating
        the spins as constants.

        Args:
            x: Clamped input of shape ``(n_in,)``.
            states: One spin vector per spin layer, first hidden layer first,
                output last. Each vector lies in ``{-1, +1}``.

        Returns:
            A scalar tensor holding the energy.

        Raises:
            ValueError: If ``states`` does not contain one vector per spin layer.
        """
        spins = [self._vector(state) for state in states]
        if len(spins) != len(self.biases):
            raise ValueError(
                f"expected {len(self.biases)} spin layers, got {len(spins)}"
            )
        previous = self._vector(x)
        total = previous.new_zeros(())
        for weight, bias, current in zip(self.weights, self.biases, spins):
            total = total - previous @ weight @ current - bias @ current
            previous = current
        return total

    def cost(self, output: Any, target: Any) -> torch.Tensor:
        """Compute the EP cost ``C = n_output - target . output``.

        For ``+-1`` spins this equals ``0.5 * ||output - target||**2``, so the
        squared cost collapses to a linear term in the output spins.

        Args:
            output: Output spins in ``{-1, +1}`` of shape ``(n_output,)``.
            target: Target spins in ``{-1, +1}`` of shape ``(n_output,)``.

        Returns:
            A scalar tensor holding the cost.
        """
        return self.n_output - self._vector(target) @ self._vector(output)

    def fields(self, x: Any, target: Any, beta: float) -> list[torch.Tensor]:
        """Compute the local field of every spin layer.

        The clamped input enters only the first spin layer. The nudge enters
        only the output layer. Middle layers carry their bias alone, because
        every spin-spin coupling stays in the QUBO off-diagonal blocks.

        Args:
            x: Clamped input of shape ``(n_in,)``.
            target: Target spins of shape ``(n_output,)``.
            beta: Non-negative nudging coefficient. ``0`` gives the free phase.

        Returns:
            One field per spin layer, first hidden layer first and the output
            field last. The output field is ``b + beta * target``.
        """
        local: list[torch.Tensor] = []
        last = len(self.biases) - 1
        for index, bias in enumerate(self.biases):
            field = bias
            if index == 0:
                field = self._vector(x) @ self.weights[0] + field
            if index == last:
                field = field + float(beta) * self._vector(target)
            local.append(field)
        return local

    def qubo(self, x: Any, target: Any, beta: float) -> tuple[torch.Tensor, float]:
        """Rewrite ``F = E + beta * C`` for one sample as an upper-triangular QUBO.

        The clamped input and the nudge are folded into the local fields first,
        so the matrix only contains the free spin layers. The returned tensors
        are detached: the solver is not differentiable, and gradients flow
        through :meth:`energy` on the spins it returns.

        Args:
            x: Clamped input of shape ``(n_in,)``.
            target: Target spins of shape ``(n_output,)``.
            beta: Non-negative nudging coefficient. ``0`` gives the free phase.

        Returns:
            A tuple ``(Q, constant)`` where ``Q`` is an upper-triangular
            ``float64`` matrix and ``constant`` is the scalar offset such that
            ``q @ Q @ q + constant`` equals ``F`` for ``q = (s + 1) / 2``.
        """
        with torch.no_grad():
            fields = self.fields(x, target, beta)
            couplings = [weight.detach() for weight in self.weights[1:]]
        return _build_qubo(couplings, fields, beta)

    def split_state(self, state: Any) -> list[torch.Tensor]:
        """Split a physical spin vector into one block per spin layer.

        Args:
            state: Physical spins in ``{-1, +1}``, ordered as
                :attr:`spin_widths`. The auxiliary spin is not included.

        Returns:
            Detached ``float64`` spin vectors, first hidden layer first and the
            output last.
        """
        spins = torch.as_tensor(state).detach().to(dtype=torch.float64).reshape(-1)
        parts: list[torch.Tensor] = []
        cursor = 0
        for width in self.spin_widths:
            parts.append(spins[cursor : cursor + width].clone())
            cursor += width
        return parts

    def predict(self, states: Sequence[Any]) -> tuple[int, bool]:
        """Predict a class from the output spins, breaking ties with local fields.

        Args:
            states: One spin vector per spin layer, output last.

        Returns:
            The predicted class index and whether the block sums tied. Ties are
            resolved by the output local field from the previous layer, then by
            the smallest class index, so predictions stay deterministic.

        Raises:
            ValueError: If ``states`` is empty.
        """
        if not states:
            raise ValueError("predict requires one spin vector per layer")
        output = self._vector(states[-1])
        scores = output.reshape(self.n_classes, self.spins_per_class).sum(dim=1)
        leaders = torch.nonzero(scores == scores.max()).reshape(-1)
        if leaders.numel() == 1:
            return int(leaders.item()), False

        with torch.no_grad():
            if len(states) == 1:
                local = self.biases[-1]
            else:
                local = self._vector(states[-2]) @ self.weights[-1] + self.biases[-1]
            local = local.reshape(self.n_classes, self.spins_per_class)
        tied_scores = local.sum(dim=1)[leaders]
        best = leaders[torch.nonzero(tied_scores == tied_scores.max()).reshape(-1)]
        return int(best.min().item()), True
