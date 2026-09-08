"""The sq network (DESIGN.md items 2 to 7). Consumes the 25 planes, returns
policy logits, categorical Q and the auxiliary heads. Tokens are the 81
squares, laid out (batch, 81, width). Exported to ONNX by `export`; read in
Rust by `src/net.rs`."""

from __future__ import annotations

import math
import os
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config, from_dict, to_dict
from .config import Net as NetConfig
from .planes import N_ACTIONS, N_DIRS, N_SQUARES, N_STATES, action_targets

# Lower edges of the plies-to-end classes.
TTE_EDGES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
TTE_CENTERS = [a + (b - a) / 2 for a, b in zip(TTE_EDGES, TTE_EDGES[1:] + [TTE_EDGES[-1] + 128], strict=True)]
N_TTE = len(TTE_EDGES)
N_OCC = 7
N_RELATIONS = 8
RELATION_ADJACENT = 7


class Output(NamedTuple):
    logits: torch.Tensor  # (B, 648)
    q_logits: torch.Tensor  # (B, 648, atoms)
    q: torch.Tensor  # (B, 648) expectation over atoms, float32
    occupancy: torch.Tensor | None  # (B, 7, 81)
    plies_to_end: torch.Tensor | None  # (B, 16)
    reply: torch.Tensor | None  # (B, 648)
    danger: torch.Tensor | None  # (B, 81)
    material: torch.Tensor | None  # (B, 2)


def atom_values(atoms: int, device=None) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, atoms, device=device)


def relation_tables() -> torch.Tensor:
    """(8, 7, 7) truth tables over ordered pairs of square states (own R/P/S,
    enemy R/P/S, empty): beats, beaten by, same type, both own, both enemy,
    occupied-empty, empty-occupied; the adjacency row is board-independent (zero)."""
    m = torch.zeros(N_RELATIONS, N_STATES, N_STATES)
    for a in range(6):
        ta, sa = a % 3, a // 3
        for c in range(6):
            tc, sc = c % 3, c // 3
            if sa != sc and (ta - tc) % 3 == 1:
                m[0, a, c] = 1.0
            if sa != sc and (tc - ta) % 3 == 1:
                m[1, a, c] = 1.0
            if ta == tc:
                m[2, a, c] = 1.0
            if sa == 0 and sc == 0:
                m[3, a, c] = 1.0
            if sa == 1 and sc == 1:
                m[4, a, c] = 1.0
        m[5, a, 6] = 1.0
        m[6, 6, a] = 1.0
    return m


def adjacency() -> torch.Tensor:
    """(81, 81) ones where two squares are one king step apart."""
    r = torch.arange(N_SQUARES) // 9
    c = torch.arange(N_SQUARES) % 9
    d = torch.maximum((r[:, None] - r[None]).abs(), (c[:, None] - c[None]).abs())
    return (d == 1).float()


def square_states(planes: torch.Tensor) -> torch.Tensor:
    """(B, 81) state index of every square from the six piece planes; 6 = empty."""
    pieces = planes[:, :6]
    occupied = pieces.sum(1) > 0.5
    return torch.where(occupied, pieces.argmax(1), torch.full_like(occupied, 6, dtype=torch.long))


class Smolgen(nn.Module):
    """Per-position attention bias generated from compressed square tokens;
    the two dense layers and the projection to 81x81 are shared by every block."""

    def __init__(self, heads: int, dim: int, hidden: int, latent: int):
        super().__init__()
        self.heads, self.latent = heads, latent
        self.dense1 = nn.Linear(N_SQUARES * dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.dense2 = nn.Linear(hidden, heads * latent)
        self.ln2 = nn.LayerNorm(latent)
        self.out = nn.Linear(latent, N_SQUARES * N_SQUARES, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, compressed: torch.Tensor) -> torch.Tensor:
        """compressed: (B, 81, dim) -> (B, heads, 81, 81)."""
        b = compressed.shape[0]
        h = self.ln1(F.gelu(self.dense1(compressed.reshape(b, -1))))
        h = self.ln2(F.gelu(self.dense2(h).view(b, self.heads, self.latent)))
        return self.out(h).view(b, self.heads, N_SQUARES, N_SQUARES)


class Block(nn.Module):
    """Pre-norm attention block over the 81 squares: a residual depthwise 3x3
    mix, attention with the positional table, the relation bias and optional
    Smolgen, then the feed-forward."""

    def __init__(self, cfg: NetConfig, smolgen: bool):
        super().__init__()
        c, self.heads = cfg.width, cfg.heads
        self.mix_norm = nn.LayerNorm(c)
        self.mix = nn.Conv2d(c, c, 3, padding=1, groups=c)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        self.norm1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c)
        self.proj = nn.Linear(c, c)
        self.norm2 = nn.LayerNorm(c)
        self.ffn_in = nn.Linear(c, cfg.ffn)
        self.ffn_out = nn.Linear(cfg.ffn, c)
        self.position = nn.Parameter(torch.zeros(cfg.heads, N_SQUARES, N_SQUARES))
        self.relation = nn.Parameter(torch.zeros(cfg.heads, N_RELATIONS))
        self.compress = nn.Linear(c, cfg.smolgen_dim) if smolgen else None
        self.explicit_attention = False

    def bias(
        self,
        states: torch.Tensor,
        tables: torch.Tensor,
        adjacent: torch.Tensor,
        y: torch.Tensor,
        smolgen: Smolgen | None,
        dtype,
    ) -> torch.Tensor:
        """(B, heads, 81, 81) additive attention bias for this block."""
        # relation[h, r] folded over the tables gives a (heads, 7, 7) lookup by state pair.
        pair = torch.einsum("hr,rac->hac", self.relation, tables)
        bias = pair[:, states[:, :, None], states[:, None, :]].permute(1, 0, 2, 3)
        bias = bias + self.position[None] + self.relation[:, RELATION_ADJACENT].view(1, -1, 1, 1) * adjacent
        if self.compress is not None and smolgen is not None:
            bias = bias + smolgen(self.compress(y))
        return bias.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        states: torch.Tensor,
        tables: torch.Tensor,
        adjacent: torch.Tensor,
        smolgen: Smolgen | None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        grid = self.mix_norm(x).to(x.dtype).transpose(1, 2).reshape(b, c, 9, 9)
        x = x + F.gelu(self.mix(grid)).flatten(2).transpose(1, 2).to(x.dtype)
        y = self.norm1(x).to(x.dtype)
        q, k, v = self.qkv(y).view(b, n, 3, self.heads, c // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        bias = self.bias(states, tables, adjacent, y, smolgen, q.dtype)
        if self.explicit_attention:
            att = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5) + bias
            o = torch.matmul(torch.softmax(att.float(), -1).to(q.dtype), v)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.proj(o.transpose(1, 2).reshape(b, n, c)).to(x.dtype)
        return x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x).to(x.dtype)))).to(x.dtype)


class FromToHead(nn.Module):
    """Scores every (direction, from square) as `q[from] . k[to] / sqrt(dh)
    + from_term[from, dir] + bias[dir]`; categorical when `atoms > 0`, where
    the from-term emits `atoms` logits and the score tilts them along the support."""

    def __init__(self, width: int, dh: int, atoms: int = 0):
        super().__init__()
        self.atoms = atoms
        self.q = nn.Linear(width, dh)
        self.k = nn.Linear(width, dh)
        self.from_term = nn.Linear(width, N_DIRS * max(atoms, 1))
        self.bias = nn.Parameter(torch.zeros(N_DIRS))
        self.scale = dh**-0.5
        nn.init.zeros_(self.k.weight)
        nn.init.zeros_(self.k.bias)
        if atoms:
            self.tilt = nn.Parameter(atom_values(atoms))
            nn.init.zeros_(self.from_term.weight)
            nn.init.zeros_(self.from_term.bias)
        # Target square of each (direction, from), 81 for an off-board target.
        targets = action_targets().view(N_DIRS, N_SQUARES)
        self.register_buffer("targets", torch.where(targets < 0, N_SQUARES, targets), persistent=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b = h.shape[0]
        q = self.q(h)  # (B, 81, dh)
        k = F.pad(self.k(h), (0, 0, 0, 1))  # (B, 82, dh); row 81 is zero
        shifted = k[:, self.targets]  # (B, 8, 81, dh)
        dot = (q[:, None] * shifted).sum(-1) * self.scale + self.bias.view(1, -1, 1)  # (B, 8, 81)
        ft = self.from_term(h)  # (B, 81, 8 * atoms)
        if not self.atoms:
            return (dot + ft.transpose(1, 2)).reshape(b, N_ACTIONS)
        logits = ft.view(b, N_SQUARES, N_DIRS, self.atoms).permute(0, 2, 1, 3)  # (B, 8, 81, atoms)
        logits = logits + dot[..., None] * self.tilt.view(1, 1, 1, -1)
        return logits.reshape(b, N_ACTIONS, self.atoms)


def head_norm(norm: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
    """BatchNorm over the width of every token, in float32."""
    return norm(x.float().transpose(1, 2)).transpose(1, 2)


class SqNet(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.width
        self.stem = nn.Conv2d(cfg.planes, c, 3, padding=1, bias=False)
        self.position = nn.Parameter(torch.zeros(1, N_SQUARES, c))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(Block(cfg, smolgen=i % cfg.smolgen_every == 0) for i in range(cfg.blocks))
        self.q_block = Block(cfg, smolgen=cfg.blocks % cfg.smolgen_every == 0)
        self.smolgen = Smolgen(cfg.heads, cfg.smolgen_dim, cfg.smolgen_hidden, cfg.smolgen_latent)
        self.register_buffer("tables", relation_tables(), persistent=False)
        self.register_buffer("adjacent", adjacency(), persistent=False)
        self.register_buffer("atoms", atom_values(cfg.atoms), persistent=False)
        self.head_norm = nn.BatchNorm1d(c)
        self.policy = FromToHead(c, cfg.head_dim)
        self.q_norm = nn.BatchNorm1d(c)
        self.q_head = FromToHead(c, cfg.head_dim, atoms=cfg.atoms)
        self.occupancy = nn.Linear(c, N_OCC)
        self.plies_to_end = nn.Sequential(nn.Linear(c, 128), nn.ReLU(), nn.Linear(128, N_TTE))
        self.reply = nn.Linear(c, N_DIRS)
        self.danger = nn.Linear(c, 1)
        self.material = nn.Sequential(nn.Linear(c, 128), nn.ReLU(), nn.Linear(128, 2))
        self.bf16_stream = True

    def set_explicit_attention(self, on: bool) -> None:
        for block in [*self.blocks, self.q_block]:
            block.explicit_attention = on

    def forward(self, planes: torch.Tensor, aux: bool = False) -> Output:
        """planes: (B, 25, 81). `aux=False` skips the training-only heads."""
        b = planes.shape[0]
        states = square_states(planes)
        x = self.stem(planes.reshape(b, self.cfg.planes, 9, 9)).flatten(2).transpose(1, 2)
        if self.bf16_stream and x.is_cuda:
            x = x.to(torch.bfloat16) + self.position.to(torch.bfloat16)
        else:
            x = x + self.position
        for block in self.blocks:
            x = block(x, states, self.tables, self.adjacent, self.smolgen)
        h = F.relu(head_norm(self.head_norm, x))
        logits = self.policy(h)
        xq = self.q_block(x, states, self.tables, self.adjacent, self.smolgen)
        hq = F.relu(head_norm(self.q_norm, xq))
        q_logits = self.q_head(hq)
        q = torch.softmax(q_logits.float(), -1) @ self.atoms
        if not aux:
            return Output(logits, q_logits, q, None, None, None, None, None)
        pooled = h.mean(1)
        return Output(
            logits,
            q_logits,
            q,
            self.occupancy(h).transpose(1, 2),
            self.plies_to_end(pooled),
            self.reply(h).transpose(1, 2).reshape(b, N_ACTIONS),
            self.danger(h).reshape(b, N_SQUARES),
            self.material(pooled),
        )


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def load_checkpoint(path: str, device="cpu") -> dict:
    """The checkpoint dict: `model`, `ema`, `optimizer`, `iteration`, `config`."""
    return torch.load(path, map_location=device, weights_only=False)


def build(cfg: Config, checkpoint: dict | None = None, device="cpu", weights: str = "model") -> SqNet:
    net = SqNet(cfg.net).to(device)
    if checkpoint is not None:
        net.load_state_dict(checkpoint[weights])
    return net


def save_checkpoint(path: str, net: SqNet, ema: SqNet, optimizer, iteration: int, cfg: Config) -> None:
    """Atomic write: temp file then rename."""
    optimizer_state = optimizer if isinstance(optimizer, dict) or optimizer is None else optimizer.state_dict()
    payload = {
        "model": net.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer_state,
        "iteration": iteration,
        "config": to_dict(cfg),
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def config_of(checkpoint: dict) -> Config:
    return from_dict(checkpoint["config"])


def hl_gauss(target: torch.Tensor, atoms: int, sigma_atoms: float) -> torch.Tensor:
    """(N, atoms) Gaussian histogram targets: the mass of N(target, sigma^2) in
    each atom's bin, with sigma given in atom spacings; the outer bins are open."""
    step = 2.0 / (atoms - 1)
    edges = torch.linspace(-1.0 - step / 2, 1.0 + step / 2, atoms + 1, device=target.device)
    edges[0], edges[-1] = float("-inf"), float("inf")
    z = (edges[None, :] - target.clamp(-1.0, 1.0)[:, None]) / (sigma_atoms * step * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    return cdf[:, 1:] - cdf[:, :-1]
