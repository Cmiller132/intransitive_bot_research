"""The trainable network (DESIGN items 6, 7, 9, 10): the integer evaluator's
arithmetic in float with fake quantisation, so the exported file evaluates
exactly what was trained. Format 8 factorises the piece-square rows into a
shared factor and per-context residuals (Stockfish's feature factorisation);
the served table is their sum, and quantisation and the range constraint act
on the sum."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .features import BUCKETS, CLOCK_BUCKETS, CONTEXTS, LAYOUTS, PIECE_ROWS, SLOTS

QA = 255  # scale of the feature table and the accumulator clamp
QB = 64  # scale of the readout, dense and residual weights
EVAL_SCALE = 600.0  # search score per unit of raw value
DENSE = 32
ROW_STD = 0.04  # initial scale of a feature row


def fake_quant(t: torch.Tensor, scale: int) -> torch.Tensor:
    """Round to the integer grid in the forward pass, identity in the backward."""
    return t + (torch.round(t * scale) / scale - t).detach()


def accumulate(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Row sums of `weight` over the ids of every perspective, `ids` (M, SLOTS)
    with the table's last (all-zero) row as padding. On the CPU the sum is a
    sparse CSR matrix product, five times faster than `embedding_bag` forward
    and backward (measured 18 ms against 95 ms for 4,096 perspectives); on
    the GPU the embedding bag is used. Both give identical sums."""
    pad = weight.shape[0] - 1
    if ids.device.type != "cpu":
        return F.embedding_bag(ids, weight, mode="sum", padding_idx=pad)
    m = ids.shape[0]
    col = ids.reshape(-1)
    crow = torch.arange(0, m * SLOTS + 1, SLOTS, dtype=torch.int64)
    values = (col != pad).to(weight.dtype)
    return torch.sparse.mm(torch.sparse_csr_tensor(crow, col, values, (m, weight.shape[0])), weight)


class NNUE(nn.Module):
    """Two perspectives share the feature table and `bias`; the concatenated
    squared clipped accumulators feed a per-bucket linear readout and a shared
    32-wide dense layer whose clipped output feeds a per-bucket residual row.
    The table is `piece` (486 x H, the shared piece-square factor) plus, in
    format 8, `context` (27 x 486 x H residuals, zero at the start), then
    `attack` (486 x H) and `clock` (32 x H). `buckets` is 1 (one head) or
    `features.BUCKETS` (one head per piece-count bucket, item 7); `version`
    is the file format, 6 or 8."""

    def __init__(self, hidden: int = 512, buckets: int = 1, version: int = 8):
        super().__init__()
        if hidden % 32 or not 32 <= hidden <= 1024:
            raise ValueError("hidden must be a multiple of 32 from 32 to 1024")
        if buckets not in (1, BUCKETS):
            raise ValueError(f"buckets must be 1 or {BUCKETS}")
        if version not in LAYOUTS:
            raise ValueError(f"version must be one of {sorted(LAYOUTS)}")
        self.hidden = hidden
        self.buckets = buckets
        self.layout = LAYOUTS[version]
        self.piece = nn.Parameter(torch.empty(PIECE_ROWS, hidden).normal_(std=ROW_STD))
        self.context = (
            nn.Parameter(torch.zeros(CONTEXTS, PIECE_ROWS, hidden)) if self.layout.contexts == CONTEXTS else None
        )
        self.attack = nn.Parameter(torch.empty(PIECE_ROWS, hidden).normal_(std=ROW_STD))
        self.clock = nn.Parameter(torch.empty(2 * CLOCK_BUCKETS, hidden).normal_(std=ROW_STD))
        self.bias = nn.Parameter(torch.full((hidden,), 0.15))
        self.output = nn.Linear(2 * hidden, buckets)
        self.dense = nn.Linear(2 * hidden, DENSE)
        self.delta = nn.Linear(DENSE, buckets, bias=False)
        nn.init.normal_(self.output.weight, std=0.025)
        nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.dense.weight, std=0.03)
        nn.init.constant_(self.dense.bias, 0.2)
        nn.init.zeros_(self.delta.weight)

    @property
    def version(self) -> int:
        return self.layout.version

    def rows(self, qat: bool = False) -> torch.Tensor:
        """The served feature table (features x H) in the layout's row order."""
        piece = self.piece if self.context is None else (self.piece + self.context).reshape(-1, self.hidden)
        table = torch.cat([piece, self.attack, self.clock])
        return fake_quant(table, QA) if qat else table

    def forward(self, ids: torch.Tensor, bucket: torch.Tensor, qat: bool = False) -> torch.Tensor:
        """`ids` (N, 2, SLOTS) feature ids in the model's layout, `bucket` (N,)
        the head of every position (all zero with one head); returns the raw
        value (N,)."""
        n = ids.shape[0]
        rows = self.rows(qat)
        weight = torch.cat([rows, rows.new_zeros(1, self.hidden)])
        bias = fake_quant(self.bias, QA) if qat else self.bias
        acc = accumulate(ids.reshape(-1, SLOTS), weight) + bias
        act = acc.clamp(0, 1).square().reshape(n, 2 * self.hidden)
        out_w = fake_quant(self.output.weight, QB) if qat else self.output.weight
        out_b = fake_quant(self.output.bias, QB) if qat else self.output.bias
        raw = F.linear(act, out_w, out_b)
        inputs = fake_quant(act, QA) if qat else act
        dw = fake_quant(self.dense.weight, QB) if qat else self.dense.weight
        db = fake_quant(self.dense.bias, QA * QB) if qat else self.dense.bias
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
        delta_w = fake_quant(self.delta.weight, QB) if qat else self.delta.weight
        raw = raw + F.linear(hidden, delta_w)
        if self.buckets == 1:
            return raw.squeeze(-1)
        return raw.gather(1, bucket.reshape(-1, 1).long()).squeeze(-1)

    @torch.no_grad()
    def constrain(self) -> None:
        """Keep every parameter inside its integer range; a context row's sum
        is brought back by moving its residual."""
        for table in (self.piece, self.attack, self.clock):
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

    @torch.no_grad()
    def widen_hidden(self, hidden: int, seed: int = 0) -> NNUE:
        """A wider copy that evaluates identically at first: the existing
        columns are kept; every new column gets its own small seeded feature
        rows (so the columns differ from the first step on), the initial bias
        and zero readout and dense columns (DESIGN item 33)."""
        if hidden <= self.hidden or hidden % 32:
            raise ValueError(f"hidden must be a larger multiple of 32 than {self.hidden}")
        wide = NNUE(hidden, self.buckets, self.version)
        old, new = self.hidden, hidden
        generator = torch.Generator().manual_seed(seed)
        for name in ("piece", "attack", "clock"):
            table = getattr(wide, name)
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
        wide.delta.weight.copy_(self.delta.weight)
        return wide

    @torch.no_grad()
    def copy_tables(self, target: NNUE) -> None:
        """Copy the feature tables, the bias and the dense layer into `target`
        (its context residuals too when both networks have them)."""
        for name in ("piece", "attack", "clock", "bias"):
            getattr(target, name).copy_(getattr(self, name))
        if self.context is not None and target.context is not None:
            target.context.copy_(self.context)
        target.dense.load_state_dict(self.dense.state_dict())

    @torch.no_grad()
    def widen_buckets(self) -> NNUE:
        """A `BUCKETS`-head copy of a one-head network with every head equal
        to the single head, so the buckets start from the same evaluation."""
        if self.buckets != 1:
            raise ValueError("already bucketed")
        wide = NNUE(self.hidden, BUCKETS, self.version)
        self.copy_tables(wide)
        wide.output.weight.copy_(self.output.weight.expand(BUCKETS, -1))
        wide.output.bias.copy_(self.output.bias.expand(BUCKETS))
        wide.delta.weight.copy_(self.delta.weight.expand(BUCKETS, -1))
        return wide

    @torch.no_grad()
    def widen_contexts(self) -> NNUE:
        """The format 8 copy of a format 6 network: the piece rows become the
        shared factor, the 27 context residuals start at zero, everything
        else is kept, so it evaluates identically until the residuals train."""
        if self.context is not None:
            raise ValueError("already has the context rows")
        wide = NNUE(self.hidden, self.buckets, 8)
        self.copy_tables(wide)
        wide.context.zero_()
        wide.output.load_state_dict(self.output.state_dict())
        wide.delta.load_state_dict(self.delta.state_dict())
        return wide
