"""The trainable network (DESIGN items 6, 7, 9, 10): the integer evaluator's
arithmetic in float with fake quantisation, so the exported file evaluates
exactly what was trained. Format 8 factorises the piece-square rows into a
shared factor and per-context residuals (Stockfish's feature factorisation);
the served table is their sum, and quantisation and the range constraint act
on the sum. Format 9 adds the eight goal-corner rows and factorises the eight
output heads the same way: one shared head plus a zero-started residual per
bucket, flattened into eight complete heads at export."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .features import CLOCK_BUCKETS, CONTEXTS, GOAL_ROWS, LAYOUTS, MAX_PIECES, PIECE_ROWS

QA = 255  # scale of the feature table and the accumulator clamp
QB = 64  # scale of the readout, dense and residual weights
EVAL_SCALE = 600.0  # search score per unit of raw value
DENSE = 32
ROW_STD = 0.04  # initial scale of a feature row
# The per-bucket head residuals, in the order the served head sums them.
HEAD_RESIDUALS = ("output_head", "output_head_bias", "dense_head", "dense_head_bias", "delta_head")


def fake_quant(t: torch.Tensor, scale: int) -> torch.Tensor:
    """Round to the integer grid in the forward pass, identity in the backward."""
    return t + (torch.round(t * scale) / scale - t).detach()


def accumulate(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Row sums of `weight` over the ids of every perspective, `ids` (M, slots)
    with the table's last (all-zero) row as padding. On the CPU the sum is a
    sparse CSR matrix product, five times faster than `embedding_bag` forward
    and backward (measured 18 ms against 95 ms for 4,096 perspectives); on
    the GPU the embedding bag is used. Both give identical sums."""
    pad = weight.shape[0] - 1
    if ids.device.type != "cpu":
        return F.embedding_bag(ids, weight, mode="sum", padding_idx=pad)
    m, slots = ids.shape
    col = ids.reshape(-1)
    crow = torch.arange(0, m * slots + 1, slots, dtype=torch.int64)
    values = (col != pad).to(weight.dtype)
    return torch.sparse.mm(torch.sparse_csr_tensor(crow, col, values, (m, weight.shape[0])), weight)


class NNUE(nn.Module):
    """Two perspectives share the feature table and `bias`; the concatenated
    squared clipped accumulators feed a linear readout and a shared 32-wide
    dense layer whose clipped output feeds a residual row. The table is
    `piece` (486 x H, the shared piece-square factor) plus, in formats 8 and 9,
    `context` (27 x 486 x H residuals, zero at the start), then `attack`
    (486 x H), `clock` (32 x H) and, in format 9, `goal` (8 x H). `version` is
    the file format, 6, 8 or 9.

    Format 9 serves eight output heads, one per piece-count bucket. They are
    factorised like the context rows: the shared `output`, `dense` and `delta`
    layers plus a zero-started residual per head (`output_head`,
    `output_head_bias`, `dense_head`, `dense_head_bias`, `delta_head`), so a
    format 8 network becomes eight identical heads and the export writes their
    sums."""

    def __init__(self, hidden: int = 512, version: int = 8):
        super().__init__()
        if hidden % 32 or not 32 <= hidden <= 1024:
            raise ValueError("hidden must be a multiple of 32 from 32 to 1024")
        if version not in LAYOUTS:
            raise ValueError(f"version must be one of {sorted(LAYOUTS)}")
        self.hidden = hidden
        self.layout = LAYOUTS[version]
        self.piece = nn.Parameter(torch.empty(PIECE_ROWS, hidden).normal_(std=ROW_STD))
        self.context = (
            nn.Parameter(torch.zeros(CONTEXTS, PIECE_ROWS, hidden)) if self.layout.contexts == CONTEXTS else None
        )
        self.attack = nn.Parameter(torch.empty(PIECE_ROWS, hidden).normal_(std=ROW_STD))
        self.clock = nn.Parameter(torch.empty(2 * CLOCK_BUCKETS, hidden).normal_(std=ROW_STD))
        self.goal = nn.Parameter(torch.empty(GOAL_ROWS, hidden).normal_(std=ROW_STD)) if self.layout.goal else None
        self.bias = nn.Parameter(torch.full((hidden,), 0.15))
        self.output = nn.Linear(2 * hidden, 1)
        self.dense = nn.Linear(2 * hidden, DENSE)
        self.delta = nn.Linear(DENSE, 1, bias=False)
        heads = self.layout.heads
        self.output_head = nn.Parameter(torch.zeros(heads, 1, 2 * hidden)) if heads > 1 else None
        self.output_head_bias = nn.Parameter(torch.zeros(heads, 1)) if heads > 1 else None
        self.dense_head = nn.Parameter(torch.zeros(heads, DENSE, 2 * hidden)) if heads > 1 else None
        self.dense_head_bias = nn.Parameter(torch.zeros(heads, DENSE)) if heads > 1 else None
        self.delta_head = nn.Parameter(torch.zeros(heads, 1, DENSE)) if heads > 1 else None
        nn.init.normal_(self.output.weight, std=0.025)
        nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.dense.weight, std=0.03)
        nn.init.constant_(self.dense.bias, 0.2)
        nn.init.zeros_(self.delta.weight)

    @property
    def version(self) -> int:
        return self.layout.version

    @property
    def heads(self) -> int:
        return self.layout.heads

    def rows(self, qat: bool = False) -> torch.Tensor:
        """The served feature table (features x H) in the layout's row order."""
        piece = self.piece if self.context is None else (self.piece + self.context).reshape(-1, self.hidden)
        tables = [piece, self.attack, self.clock] + ([] if self.goal is None else [self.goal])
        table = torch.cat(tables)
        return fake_quant(table, QA) if qat else table

    def head(self, index: int = 0) -> tuple[torch.Tensor, ...]:
        """The served readout weight and bias, dense weight and bias and
        residual readout of one head: the shared layer plus its residual."""
        parts = (self.output.weight, self.output.bias, self.dense.weight, self.dense.bias, self.delta.weight)
        if self.heads == 1:
            return parts
        residuals = (self.output_head, self.output_head_bias, self.dense_head, self.dense_head_bias, self.delta_head)
        return tuple(part + residual[index] for part, residual in zip(parts, residuals, strict=True))

    def value(self, act: torch.Tensor, index: int, qat: bool) -> torch.Tensor:
        """The raw value of already activated accumulators under one head."""
        weight, bias, dense, dense_bias, delta = self.head(index)
        out_w = fake_quant(weight, QB) if qat else weight
        out_b = fake_quant(bias, QB) if qat else bias
        raw = F.linear(act, out_w, out_b)
        inputs = fake_quant(act, QA) if qat else act
        dw = fake_quant(dense, QB) if qat else dense
        db = fake_quant(dense_bias, QA * QB) if qat else dense_bias
        hidden = F.linear(inputs, dw, db).clamp(0, 1)
        if qat:
            # The exact integer hidden layer, with straight-through gradients.
            with torch.no_grad():
                total = F.linear(
                    torch.round(inputs * QA).double(),
                    torch.round(dw * QB).double(),
                    torch.round(db * QA * QB).double(),
                )
                exact = torch.round(total / QB).clamp(0, QA).to(hidden.dtype) / QA
            hidden = hidden + (exact - hidden).detach()
        delta_w = fake_quant(delta, QB) if qat else delta
        return (raw + F.linear(hidden, delta_w)).squeeze(-1)

    def buckets(self, ids: torch.Tensor) -> torch.Tensor:
        """(N,) head index of every position: one piece slot per occupied
        square, so the filled slots of a perspective are the pieces on the
        board, and `bucket = min(heads - 1, (pieces - 2) * heads // 19)`."""
        pieces = (ids[:, 0, :MAX_PIECES] != self.layout.pad).sum(-1)
        return ((pieces - 2).clamp(min=0) * self.heads // 19).clamp(max=self.heads - 1)

    def forward(self, ids: torch.Tensor, qat: bool = False) -> torch.Tensor:
        """`ids` (N, 2, slots) feature ids in the model's layout; returns the raw value (N,)."""
        n = ids.shape[0]
        rows = self.rows(qat)
        weight = torch.cat([rows, rows.new_zeros(1, self.hidden)])
        bias = fake_quant(self.bias, QA) if qat else self.bias
        acc = accumulate(ids.reshape(-1, self.layout.slots), weight) + bias
        act = acc.clamp(0, 1).square().reshape(n, 2 * self.hidden)
        if self.heads == 1:
            return self.value(act, 0, qat)
        bucket = self.buckets(ids)
        index = [torch.zeros(0, dtype=torch.int64, device=ids.device)]
        value = [act.new_zeros(0)]
        for head in range(self.heads):
            rows_of = torch.nonzero(bucket == head, as_tuple=False).squeeze(-1)
            if len(rows_of):
                index.append(rows_of)
                value.append(self.value(act[rows_of], head, qat))
        return act.new_zeros(n).index_put((torch.cat(index),), torch.cat(value))

    @torch.no_grad()
    def constrain(self) -> None:
        """Keep every parameter inside its integer range; a context row's or a
        head's sum is brought back by moving its residual."""
        for table in (self.piece, self.attack, self.clock, self.goal):
            if table is not None:
                table.clamp_(-8, 8)
        if self.context is not None:
            total = self.piece + self.context
            self.context.add_(total.clamp(-8, 8) - total)
        self.bias.clamp_(-8, 8)
        self.output.weight.clamp_(-64, 64)
        self.output.bias.clamp_(-64, 64)
        self.dense.weight.clamp_(-127 / QB, 127 / QB)
        self.dense.bias.clamp_(-64, 64)
        self.delta.weight.clamp_(-64, 64)
        for shared, residual, low, high in (
            (self.output.weight, self.output_head, -64, 64),
            (self.output.bias, self.output_head_bias, -64, 64),
            (self.dense.weight, self.dense_head, -127 / QB, 127 / QB),
            (self.dense.bias, self.dense_head_bias, -64, 64),
            (self.delta.weight, self.delta_head, -64, 64),
        ):
            if residual is not None:
                total = shared + residual
                residual.add_(total.clamp(low, high) - total)

    @torch.no_grad()
    def widen_hidden(self, hidden: int, seed: int = 0, outgoing: float = 0.0) -> NNUE:
        """A wider copy: the existing columns are kept; every new column gets
        its own small seeded feature rows (so the columns differ from the first
        step on), the initial bias and, with `outgoing` 0, zero readout and
        dense columns, so the copy evaluates identically at first (DESIGN item
        33). With `outgoing` > 0 the new readout columns are +-outgoing with
        seeded signs (the dense columns stay zero): a value on the served grid
        (1/64) carries a task gradient into the new channels from the first
        quantised step, and the copy deviates from the parent by a bounded
        amount (the dense path would amplify the seeds through the residual)."""
        if hidden <= self.hidden or hidden % 32:
            raise ValueError(f"hidden must be a larger multiple of 32 than {self.hidden}")
        wide = NNUE(hidden, self.version)
        old, new = self.hidden, hidden
        generator = torch.Generator().manual_seed(seed)
        for name in ("piece", "attack", "clock", "goal"):
            table = getattr(wide, name)
            if table is None:
                continue
            table.normal_(std=ROW_STD, generator=generator)
            table[:, :old].copy_(getattr(self, name))
        if self.context is not None:
            wide.context.zero_()
            wide.context[..., :old].copy_(self.context)
        wide.bias.fill_(0.15)
        wide.bias[:old].copy_(self.bias)
        for name in ("output", "dense"):
            layer, source = getattr(wide, name), getattr(self, name)
            layer.weight.zero_()
            layer.weight[:, :old].copy_(source.weight[:, :old])
            layer.weight[:, new : new + old].copy_(source.weight[:, old:])
            layer.bias.copy_(source.bias)
            if outgoing > 0 and name == "output":
                for start in (old, new + old):
                    signs = torch.randint(0, 2, (layer.weight.shape[0], new - old), generator=generator) * 2 - 1
                    layer.weight[:, start : start + new - old].copy_(signs.to(layer.weight.dtype) * outgoing)
        wide.delta.weight.copy_(self.delta.weight)
        # The head residuals follow their shared layer: the new columns start at
        # zero, so every head deviates from the parent by what its layer does.
        for name in ("output_head", "dense_head"):
            residual = getattr(wide, name)
            if residual is not None:
                residual.zero_()
                residual[..., :old].copy_(getattr(self, name)[..., :old])
                residual[..., new : new + old].copy_(getattr(self, name)[..., old:])
        for name in ("output_head_bias", "dense_head_bias", "delta_head"):
            if getattr(wide, name) is not None:
                getattr(wide, name).copy_(getattr(self, name))
        return wide

    @torch.no_grad()
    def copy_tables(self, target: NNUE) -> None:
        """Copy the feature tables, the bias and the dense layer into `target`
        (its context residuals and goal rows too when both networks have them)."""
        for name in ("piece", "attack", "clock", "bias"):
            getattr(target, name).copy_(getattr(self, name))
        for name in ("context", "goal"):
            if getattr(self, name) is not None and getattr(target, name) is not None:
                getattr(target, name).copy_(getattr(self, name))
        target.dense.load_state_dict(self.dense.state_dict())

    @torch.no_grad()
    def widen_contexts(self) -> NNUE:
        """The format 8 copy of a format 6 network: the piece rows become the
        shared factor, the 27 context residuals start at zero, everything
        else is kept, so it evaluates identically until the residuals train."""
        if self.context is not None:
            raise ValueError("already has the context rows")
        wide = NNUE(self.hidden, 8)
        self.copy_tables(wide)
        wide.context.zero_()
        wide.output.load_state_dict(self.output.state_dict())
        wide.delta.load_state_dict(self.delta.state_dict())
        return wide

    @torch.no_grad()
    def widen_buckets(self) -> NNUE:
        """The format 9 copy of a format 8 network: the eight goal rows and the
        eight head residuals start at zero, so every bucket serves the one head
        it came from and the network evaluates identically until they train.
        Byte for byte the conversion of `nnue.export convert --to 9`."""
        if self.layout.goal or self.context is None:
            raise ValueError("needs a format 8 network")
        wide = NNUE(self.hidden, 9)
        self.copy_tables(wide)
        wide.goal.zero_()
        wide.output.load_state_dict(self.output.state_dict())
        wide.delta.load_state_dict(self.delta.state_dict())
        for name in HEAD_RESIDUALS:
            getattr(wide, name).zero_()
        return wide
