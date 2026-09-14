"""The conv network (DESIGN.md items 2 to 12). Consumes the 46 planes, returns
policy logits, categorical Q, the distributional state value and the
training-only heads. Tokens are the 81 squares, laid out (batch, 81, width),
viewed as a 9x9 grid inside the convolution blocks. Exported to ONNX by
`export`; read in Rust by `src/net.rs`."""

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
N_RELATIONS = 7
# Relative offsets (rank and file deltas in -8..8) between two squares of the 9x9 board.
N_SIDE = 17
N_OFFSETS = N_SIDE * N_SIDE
N_TACTICS = 3


class Output(NamedTuple):
    logits: torch.Tensor  # (B, 648)
    q_logits: torch.Tensor  # (B, 648, atoms)
    q: torch.Tensor  # (B, 648) expectation over atoms, float32
    v_logits: torch.Tensor  # (B, atoms)
    v: torch.Tensor  # (B,) expectation over atoms, float32
    tactics: torch.Tensor | None  # (B, 648, 3)
    occupancy: torch.Tensor | None  # (B, 7, 81)
    plies_to_end: torch.Tensor | None  # (B, 16)
    material: torch.Tensor | None  # (B, 2)
    wdl: torch.Tensor | None  # (B, 3) win / draw / loss logits from the mover's view
    reply: torch.Tensor | None  # (B, 648) logits of the opponent's reply, in the mover's frame
    regret: torch.Tensor | None  # (B, 2) the search control's ranking score and predicted regret


def atom_values(atoms: int, device=None) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, atoms, device=device)


def relation_tables() -> torch.Tensor:
    """(7, 7, 7) truth tables over ordered pairs of square states (own R/P/S,
    enemy R/P/S, empty): beats, beaten by, same type, both own, both enemy,
    occupied-empty, empty-occupied."""
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


def offset_index() -> torch.Tensor:
    """(81, 81) index of the relative offset `(rank_j - rank_i + 8, file_j - file_i + 8)`."""
    r = torch.arange(N_SQUARES) // 9
    c = torch.arange(N_SQUARES) % 9
    return (r[None] - r[:, None] + 8) * N_SIDE + (c[None] - c[:, None] + 8)


def square_states(planes: torch.Tensor) -> torch.Tensor:
    """(B, 81) state index of every square from the six piece planes; 6 = empty."""
    pieces = planes[:, :6]
    occupied = pieces.sum(1) > 0.5
    return torch.where(occupied, pieces.argmax(1), torch.full_like(occupied, 6, dtype=torch.long))


class Smolgen(nn.Module):
    """Per-position attention bias generated from compressed square tokens;
    the two dense layers and the projection to 81x81 are shared by every block
    that uses it."""

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


class ConvBlock(nn.Module):
    """Local block: LayerNorm over the width, then two 3x3 convolutions on the
    9x9 grid with GELU between, added back to the token stream. Zero padding;
    the position embedding tells the block where the edges are."""

    def __init__(self, cfg: NetConfig):
        super().__init__()
        c = cfg.width
        self.norm = nn.LayerNorm(c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, c = x.shape
        grid = self.norm(x).transpose(1, 2).reshape(b, c, 9, 9)
        y = self.conv2(F.gelu(self.conv1(grid)))
        return x + y.flatten(2).transpose(1, 2)


def rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """`x / rms(x)` over the last dimension, in float32."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class AttnBlock(nn.Module):
    """Pre-norm attention block over the 81 squares: attention with the
    positional table, the relation-by-offset bias and optional Smolgen, then
    the feed-forward. Local mixing is left to the convolution blocks. Queries
    and keys are RMS-normalised per head with a learned scale (QK-norm). This
    bounds the logits so fused attention can use bf16 under CUDA autocast;
    the explicit fp32 path remains available for ONNX export."""

    def __init__(self, cfg: NetConfig, smolgen: bool):
        super().__init__()
        c, self.heads = cfg.width, cfg.heads
        self.fused_attention = cfg.fused_attention
        self.norm1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c)
        self.q_scale = nn.Parameter(torch.ones(c // self.heads))
        self.k_scale = nn.Parameter(torch.ones(c // self.heads))
        self.proj = nn.Linear(c, c)
        self.norm2 = nn.LayerNorm(c)
        self.ffn_in = nn.Linear(c, cfg.ffn)
        self.ffn_out = nn.Linear(cfg.ffn, c)
        self.position = nn.Parameter(torch.zeros(cfg.heads, N_SQUARES, N_SQUARES))
        self.relation = nn.Parameter(torch.zeros(cfg.heads, N_RELATIONS, N_SIDE, N_SIDE))
        self.compress = nn.Linear(c, cfg.smolgen_dim) if smolgen else None

    def bias(
        self,
        relations: torch.Tensor,
        offsets: torch.Tensor,
        y: torch.Tensor,
        smolgen: Smolgen | None,
        dtype,
    ) -> torch.Tensor:
        """(B, heads, 81, 81) additive attention bias for this block: the
        relation table at every square pair's offset, weighted by the relations
        that hold between the pair (`relations`, (B, 81, 81) with bit r set for
        relation r), plus the positional table and Smolgen. A contraction
        rather than a gather by state pair, so its backward is a reduction and
        not an index-add; computed in the stream's precision."""
        table = self.relation.flatten(2)[:, :, offsets].to(dtype)  # (heads, 7, 81, 81)
        bias = self.position[None].to(dtype)
        for r in range(N_RELATIONS):
            held = torch.div(relations, 1 << r, rounding_mode="floor") % 2  # bit r, in ops ONNX exports
            bias = bias + held.to(dtype)[:, None] * table[:, r][None]
        if self.compress is not None and smolgen is not None:
            bias = bias + smolgen(self.compress(y)).to(dtype)
        return bias

    def forward(
        self,
        x: torch.Tensor,
        relations: torch.Tensor,
        offsets: torch.Tensor,
        smolgen: Smolgen | None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        y = self.norm1(x)
        q, k, v = self.qkv(y).view(b, n, 3, self.heads, c // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q = rms_normalize(q) * self.q_scale
        k = rms_normalize(k) * self.k_scale
        if self.fused_attention:
            # CUDA autocast makes qkv bf16; CPU inference stays float32. SDPA
            # requires q, k, v and its arbitrary additive mask to share a dtype.
            dtype = v.dtype if v.is_cuda else torch.float32
            q, k, v = (t.to(dtype) for t in (q, k, v))
            bias = self.bias(relations, offsets, y, smolgen, dtype)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        else:
            bias = self.bias(relations, offsets, y, smolgen, torch.float32)
            with torch.autocast(x.device.type, enabled=False):
                att = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5) + bias
                p = torch.softmax(att, -1)
            o = torch.matmul(p.to(v.dtype), v)
        x = x + self.proj(o.transpose(1, 2).reshape(b, n, c))
        return x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x))))


class FromToHead(nn.Module):
    """Scores every (direction, from square) as `q[from] . k[to] / sqrt(dh)
    + from_term[from, dir] + bias[dir]`; categorical when `atoms > 0`, where the
    from-term emits `atoms` logits and the score either tilts them along the
    support (the Q head) or shifts all of them equally (the tactics head)."""

    def __init__(self, width: int, dh: int, atoms: int = 0, tilt: bool = True):
        super().__init__()
        self.atoms = atoms
        self.q = nn.Linear(width, dh)
        self.k = nn.Linear(width, dh)
        self.from_term = nn.Linear(width, N_DIRS * max(atoms, 1))
        self.bias = nn.Parameter(torch.zeros(N_DIRS))
        self.scale = dh**-0.5
        nn.init.zeros_(self.k.weight)
        nn.init.zeros_(self.k.bias)
        self.tilt = nn.Parameter(atom_values(atoms)) if atoms and tilt else None
        if atoms:
            nn.init.zeros_(self.from_term.weight)
            nn.init.zeros_(self.from_term.bias)
        # Target square of each (from, direction), 81 for an off-board target.
        targets = action_targets().view(N_DIRS, N_SQUARES).t()
        self.register_buffer("targets", torch.where(targets < 0, N_SQUARES, targets).contiguous(), persistent=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b = h.shape[0]
        q = self.q(h)  # (B, 81, dh)
        k = F.pad(self.k(h), (0, 0, 0, 1))  # (B, 82, dh); row 81 is zero
        # Every from-to score at once as one matmul, then the eight king targets of each square.
        full = torch.bmm(q, k.transpose(1, 2))  # (B, 81, 82)
        dot = full.gather(2, self.targets[None].expand(b, -1, -1))  # (B, 81, 8)
        dot = dot.transpose(1, 2) * self.scale + self.bias.view(1, -1, 1)  # (B, 8, 81)
        ft = self.from_term(h)  # (B, 81, 8 * atoms)
        if not self.atoms:
            return (dot + ft.transpose(1, 2)).reshape(b, N_ACTIONS)
        logits = ft.view(b, N_SQUARES, N_DIRS, self.atoms).permute(0, 2, 1, 3)  # (B, 8, 81, atoms)
        spread = dot[..., None] if self.tilt is None else dot[..., None] * self.tilt.view(1, 1, 1, -1)
        return (logits + spread).reshape(b, N_ACTIONS, self.atoms)


class ConvNet(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.width
        self.stem = nn.Conv2d(cfg.planes, c, 3, padding=1, bias=False)
        self.position = nn.Parameter(torch.zeros(1, N_SQUARES, c))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.conv_blocks = nn.ModuleList(ConvBlock(cfg) for _ in range(cfg.pairs))
        self.attn_blocks = nn.ModuleList(
            AttnBlock(cfg, smolgen=i % cfg.smolgen_every == cfg.smolgen_first) for i in range(cfg.pairs)
        )
        self.smolgen = Smolgen(cfg.heads, cfg.smolgen_dim, cfg.smolgen_hidden, cfg.smolgen_latent)
        # (49,) bitmask of the relations that hold for the state pair `state_i * 7 + state_j`.
        bits = sum(relation_tables()[r].flatten().to(torch.uint8) << r for r in range(N_RELATIONS))
        self.register_buffer("relation_bits", bits, persistent=False)
        self.register_buffer("offsets", offset_index(), persistent=False)
        self.register_buffer("atoms", atom_values(cfg.atoms), persistent=False)
        self.head_norm = nn.LayerNorm(c)
        self.policy = FromToHead(c, cfg.head_dim)
        self.q_head = FromToHead(c, cfg.head_dim, atoms=cfg.atoms)
        # The state features behind V, WDL and regret (DESIGN.md item 10):
        # "spatial" is the per-square projection, flatten and dense 256 of the
        # 2026-09-10 amendment; "pooled" is the older mean + a1 + i9 (336) that
        # the conv_g128 checkpoints carry, kept so they still load as the
        # distillation teacher and for the arena export. The module order
        # matches the old model for the pooled variant: a checkpoint's optimizer
        # state is keyed by parameter index (train.optimizer_state).
        if cfg.state_head not in ("spatial", "pooled"):
            raise ValueError(f"unknown state head {cfg.state_head!r}")
        spatial = cfg.state_head == "spatial"
        state_dim = cfg.value_hidden if spatial else 3 * c
        self.state_encoder = (
            nn.Sequential(
                nn.Linear(c, cfg.value_squares),
                nn.GELU(),
                nn.Flatten(start_dim=1),
                nn.Linear(N_SQUARES * cfg.value_squares, cfg.value_hidden),
                nn.ReLU(),
            )
            if spatial
            else None
        )
        self.value = (
            nn.Linear(cfg.value_hidden, cfg.atoms)
            if spatial
            else nn.Sequential(
                nn.Linear(state_dim, cfg.value_hidden), nn.ReLU(), nn.Linear(cfg.value_hidden, cfg.atoms)
            )
        )
        self.tactics = FromToHead(c, cfg.head_dim, atoms=N_TACTICS, tilt=False)
        self.occupancy = nn.Linear(c, N_OCC)
        self.plies_to_end = nn.Sequential(nn.Linear(c, 128), nn.ReLU(), nn.Linear(128, N_TTE))
        self.material = nn.Sequential(nn.Linear(c, 128), nn.ReLU(), nn.Linear(128, 2))
        if not cfg.wdl:
            self.wdl = None
        elif spatial:
            self.wdl = nn.Linear(cfg.value_hidden, 3)
        else:
            self.wdl = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 3))
        # Registered after every older head (see above).
        self.reply = FromToHead(c, cfg.head_dim) if cfg.reply else None
        self.regret = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 2)) if cfg.regret else None

    def forward(self, planes: torch.Tensor, aux: bool = False) -> Output:
        """planes: (B, 46, 81). `aux=False` skips the training-only heads."""
        b = planes.shape[0]
        states = square_states(planes)
        # Which of the seven relations hold between every ordered pair of squares.
        relations = self.relation_bits[states[:, :, None] * N_STATES + states[:, None, :]]
        # The residual stream stays in float32 whatever the autocast precision of
        # the blocks' matmuls: a bf16 stream loses every block's contribution
        # once its magnitude grows, which turns a drift into a collapse.
        x = self.stem(planes.reshape(b, self.cfg.planes, 9, 9)).flatten(2).transpose(1, 2) + self.position
        for conv, attn in zip(self.conv_blocks, self.attn_blocks, strict=True):
            x = attn(conv(x), relations, self.offsets, self.smolgen)
        h = F.relu(self.head_norm(x.float()))
        logits = self.policy(h)
        q_logits = self.q_head(h)
        q = torch.softmax(q_logits.float(), -1) @ self.atoms
        pooled = h.mean(1)
        if self.state_encoder is not None:
            state = self.state_encoder(h)
        else:
            state = torch.cat([pooled, h[:, 0], h[:, N_SQUARES - 1]], dim=1)
        v_logits = self.value(state)
        v = torch.softmax(v_logits.float(), -1) @ self.atoms
        wdl = self.wdl(state) if self.wdl is not None else None
        # The search control reads the regret head at every search node.
        regret = self.regret(state) if self.regret is not None else None
        if not aux:
            return Output(logits, q_logits, q, v_logits, v, None, None, None, None, wdl, None, regret)
        return Output(
            logits,
            q_logits,
            q,
            v_logits,
            v,
            self.tactics(h),
            self.occupancy(h).transpose(1, 2),
            self.plies_to_end(pooled),
            self.material(pooled),
            wdl,
            self.reply(h) if self.reply is not None else None,
            regret,
        )


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def load_checkpoint(path: str, device="cpu") -> dict:
    """The checkpoint dict: `model`, `ema`, `optimizer`, `iteration`, `config`."""
    return torch.load(path, map_location=device, weights_only=False)


def build(cfg: Config, checkpoint: dict | None = None, device="cpu", weights: str = "model") -> ConvNet:
    net = ConvNet(cfg.net).to(device)
    if checkpoint is not None:
        state = checkpoint[weights]
        missing, unexpected = net.load_state_dict(state, strict=False)
        fresh = [k for k in missing if k.startswith(("wdl.", "reply.", "regret."))]
        assert not unexpected and missing == fresh, f"checkpoint mismatch: missing {missing}, unexpected {unexpected}"
        net.fresh = fresh
    return net


def share_fresh_heads(net: ConvNet, ema: ConvNet) -> None:
    """Give the EMA the student's copy of every head the checkpoint lacked,
    so the actor's fresh heads are the ones being trained."""
    state = net.state_dict()
    with torch.no_grad():
        for key in getattr(net, "fresh", []):
            ema.state_dict()[key].copy_(state[key])


def save_checkpoint(
    path: str, net: ConvNet, ema: ConvNet, optimizer, iteration: int, cfg: Config, extra: dict | None = None
) -> None:
    """Atomic write: temp file then rename; `extra` adds entries (the search control's state)."""
    optimizer_state = optimizer if isinstance(optimizer, dict) or optimizer is None else optimizer.state_dict()
    payload = {
        **(extra or {}),
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
    cfg = from_dict(checkpoint["config"])
    # Checkpoints written before 2026-09-10 have no `state_head` and carry
    # the pooled heads; the field's default is the spatial encoder.
    if "state_head" not in checkpoint["config"].get("net", {}):
        cfg.net.state_head = "pooled"
    return cfg


def hl_gauss(target: torch.Tensor, atoms: int, sigma_atoms: float) -> torch.Tensor:
    """(N, atoms) Gaussian histogram targets: the mass of N(target, sigma^2) in
    each atom's bin, with sigma given in atom spacings; the outer bins are open."""
    step = 2.0 / (atoms - 1)
    inner = torch.linspace(-1.0 + step / 2, 1.0 - step / 2, atoms - 1, device=target.device)
    z = (inner[None, :] - target.clamp(-1.0, 1.0)[:, None]) / (sigma_atoms * step * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    edge = cdf[:, :1]
    cdf = torch.cat([torch.zeros_like(edge), cdf, torch.ones_like(edge)], dim=1)
    return cdf[:, 1:] - cdf[:, :-1]
