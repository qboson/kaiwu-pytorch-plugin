"""NMT-style normalized Transformer decoder with DEQ-Ising mode."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from nmt_deq_ising.solvers.deq_ising import DEQIsingImplicitSolver


def cosine_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Place vectors on the unit hypersphere."""
    return F.normalize(x, p=2, dim=dim, eps=1e-6)


class NormalizedAttention(nn.Module):
    """Cosine-normalized multi-head attention."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, q_len, _ = query.shape
        k_len = key_value.shape[1]
        q = self.q(query).view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(key_value).view(batch, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(key_value).view(batch, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = cosine_norm(q)
        k = cosine_norm(k)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.logit_scale.clamp(0.01, 20.0)
        valid = query_mask.unsqueeze(1).unsqueeze(-1) & key_mask.unsqueeze(1).unsqueeze(-2)
        if pair_mask is not None:
            valid = valid & pair_mask.unsqueeze(1)
        logits = logits.masked_fill(~valid, -1e4)
        attn = torch.softmax(logits, dim=-1).masked_fill(~valid, 0.0)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, q_len, self.dim)
        return self.out(out).masked_fill(~query_mask.unsqueeze(-1), 0.0)


class NormalizedMLP(nn.Module):
    """NMT-style normalized MLP block."""

    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden = dim * hidden_mult
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class NormalizedMatchingBlock(nn.Module):
    """One bidirectional NMT-style matching update block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_mult: int = 4,
        alpha: float = 0.5,
        use_graph_attention_mask: bool = False,
        use_anchor_attention: bool = False,
    ):
        super().__init__()
        self.self_attn = NormalizedAttention(dim, num_heads)
        self.cross_attn = NormalizedAttention(dim, num_heads)
        self.anchor_self_attn = NormalizedAttention(dim, num_heads)
        self.anchor_cross_attn = NormalizedAttention(dim, num_heads)
        self.mlp = NormalizedMLP(dim, mlp_hidden_mult)
        self.alpha_self = nn.Parameter(torch.tensor(float(alpha)))
        self.alpha_cross = nn.Parameter(torch.tensor(float(alpha)))
        self.alpha_anchor_self = nn.Parameter(torch.tensor(float(alpha)))
        self.alpha_anchor_cross = nn.Parameter(torch.tensor(float(alpha)))
        self.alpha_mlp = nn.Parameter(torch.tensor(float(alpha)))
        self.use_graph_attention_mask = use_graph_attention_mask
        self.use_anchor_attention = use_anchor_attention

    def _update_one(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
        x_adj: torch.Tensor,
        x_anchor: torch.Tensor | None = None,
        y_anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pair_mask = x_adj > 0 if self.use_graph_attention_mask else None
        proposal = cosine_norm(self.self_attn(x, x, x_mask, x_mask, pair_mask))
        x = damped_norm(x, proposal, self.alpha_self, x_mask)
        if self.use_anchor_attention and x_anchor is not None:
            proposal = cosine_norm(self.anchor_self_attn(x, x_anchor, x_mask, x_mask, pair_mask))
            x = damped_norm(x, proposal, self.alpha_anchor_self, x_mask)
        proposal = cosine_norm(self.cross_attn(x, y, x_mask, y_mask))
        x = damped_norm(x, proposal, self.alpha_cross, x_mask)
        if self.use_anchor_attention and y_anchor is not None:
            proposal = cosine_norm(self.anchor_cross_attn(x, y_anchor, x_mask, y_mask))
            x = damped_norm(x, proposal, self.alpha_anchor_cross, x_mask)
        proposal = cosine_norm(self.mlp(x))
        x = damped_norm(x, proposal, self.alpha_mlp, x_mask)
        return x

    def forward(
        self,
        ha: torch.Tensor,
        hb: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        adj_a: torch.Tensor,
        adj_b: torch.Tensor,
        anchor_a: torch.Tensor | None = None,
        anchor_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_a = self._update_one(ha, hb, mask_a, mask_b, adj_a, anchor_a, anchor_b)
        next_b = self._update_one(hb, ha, mask_b, mask_a, adj_b, anchor_b, anchor_a)
        return next_a, next_b


def damped_norm(current: torch.Tensor, proposal: torch.Tensor, alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply NMT-style normalized damped residual update."""
    weight = torch.sigmoid(alpha)
    out = cosine_norm((1.0 - weight) * current + weight * proposal)
    return out.masked_fill(~mask.unsqueeze(-1), 0.0)


class NormalizedMatchingDecoder(nn.Module):
    """Explicit, shared recurrent, or DEQ-Ising normalized matching decoder."""

    def __init__(self, config: dict[str, object]):
        super().__init__()
        dim = int(config.get("dim", 128))
        num_heads = int(config.get("num_heads", 4))
        mlp_hidden_mult = int(config.get("mlp_hidden_mult", 4))
        alpha = float(config.get("alpha", 0.5))
        self.mode = str(config.get("mode", "explicit"))
        self.depth = int(config.get("depth", 4))
        self.recurrent_steps = int(config.get("recurrent_steps", 16))
        self.return_intermediate = bool(config.get("return_intermediate", True))
        self.use_graph_attention_mask = bool(config.get("use_graph_attention_mask", False))
        self.anchor_alpha = float(config.get("anchor_alpha", 0.0))
        self.anchor_mode = str(config.get("anchor_mode", "mix"))
        self.use_anchor_attention = self.anchor_mode in {"attention", "transformer_xl"}
        self.block = NormalizedMatchingBlock(
            dim,
            num_heads,
            mlp_hidden_mult,
            alpha,
            self.use_graph_attention_mask,
            self.use_anchor_attention,
        )
        if self.mode == "explicit":
            self.layers = nn.ModuleList(
                [
                    NormalizedMatchingBlock(
                        dim,
                        num_heads,
                        mlp_hidden_mult,
                        alpha,
                        self.use_graph_attention_mask,
                        self.use_anchor_attention,
                    )
                    for _ in range(self.depth)
                ]
            )
        else:
            self.layers = nn.ModuleList()
        self.deq_solver = DEQIsingImplicitSolver(config.get("ising", {}))

    def set_mode(self, mode: str) -> None:
        """Switch decoder mode without rebuilding parameters."""
        self.mode = mode

    def _apply_block(
        self,
        z: torch.Tensor,
        split: int,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        adj_a: torch.Tensor,
        adj_b: torch.Tensor,
        anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ha, hb = z[:, :split], z[:, split:]
        anchor_a, anchor_b = (None, None) if anchor is None else (anchor[:, :split], anchor[:, split:])
        next_a, next_b = self.block(ha, hb, mask_a, mask_b, adj_a, adj_b, anchor_a, anchor_b)
        out = torch.cat([next_a, next_b], dim=1)
        if anchor is not None and self.anchor_alpha > 0.0 and self.anchor_mode == "mix":
            mask = torch.cat([mask_a, mask_b], dim=1).unsqueeze(-1)
            out = cosine_norm((1.0 - self.anchor_alpha) * out + self.anchor_alpha * anchor)
            out = out.masked_fill(~mask, 0.0)
        return out

    def forward(
        self,
        ha: torch.Tensor,
        hb: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        adj_a: torch.Tensor,
        adj_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        if self.mode == "explicit":
            for layer in self.layers:
                ha, hb = layer(ha, hb, mask_a, mask_b, adj_a, adj_b)
            return ha, hb, {"solver": "explicit", "nstep": len(self.layers), "final_rel": torch.tensor(0.0, device=ha.device)}

        z0 = torch.cat([ha, hb], dim=1)
        split = ha.shape[1]
        if self.mode in {"shared", "shared_recurrent", "theta0"}:
            z = z0
            intermediates = []
            for _ in range(self.recurrent_steps):
                z = self._apply_block(z, split, mask_a, mask_b, adj_a, adj_b, anchor=z0)
                if self.return_intermediate:
                    intermediates.append((z[:, :split], z[:, split:]))
            info = {
                "solver": "shared_recurrent",
                "nstep": self.recurrent_steps,
                "final_rel": self._residual_norm(z, split, mask_a, mask_b, adj_a, adj_b, anchor=z0),
            }
            if intermediates:
                info["intermediates"] = intermediates
            return z[:, :split], z[:, split:], info

        if self.mode == "deq_ising":
            func: Callable[[torch.Tensor], torch.Tensor] = lambda z: self._apply_block(z, split, mask_a, mask_b, adj_a, adj_b, anchor=z0)
            z_star, info = self.deq_solver(func, z0)
            return z_star[:, :split], z_star[:, split:], info

        raise ValueError(f"Unsupported decoder mode: {self.mode}")

    def _residual_norm(
        self,
        z: torch.Tensor,
        split: int,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        adj_a: torch.Tensor,
        adj_b: torch.Tensor,
        anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            residual = self._apply_block(z, split, mask_a, mask_b, adj_a, adj_b, anchor=anchor) - z
            return residual.norm() / z.norm().clamp_min(1e-8)
