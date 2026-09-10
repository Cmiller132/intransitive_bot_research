"""The trainable network (DESIGN items 6, 7, 9, 10): the integer evaluator's
arithmetic in float with fake quantisation, so the exported file evaluates
exactly what was trained."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .features import BUCKETS, FEATURES, PAD, SLOTS

QA = 255  # scale of the feature table and the accumulator clamp
QB = 64  # scale of the readout, dense and residual weights
EVAL_SCALE = 600.0  # search score per unit of raw value
DENSE = 32


def fake_quant(t: torch.Tensor, scale: int) -> torch.Tensor:
    """Round to the integer grid in the forward pass, identity in the backward."""
    return t + (torch.round(t * scale) / scale - t).detach()


def accumulate(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Row sums of `weight` over the ids of every perspective, `ids` (M, SLOTS)
    with `PAD` for empty slots. On the CPU the sum is a sparse CSR matrix
    product, five times faster than `embedding_bag` forward and backward
    (measured 18 ms against 95 ms for 4,096 perspectives); on the GPU the
    embedding bag is used as before. Both give identical sums."""
    if ids.device.type != "cpu":
        return F.embedding_bag(ids, weight, mode="sum", padding_idx=PAD)
    m = ids.shape[0]
    col = ids.reshape(-1)
    crow = torch.arange(0, m * SLOTS + 1, SLOTS, dtype=torch.int64)
    values = (col != PAD).to(weight.dtype)
    return torch.sparse.mm(torch.sparse_csr_tensor(crow, col, values, (m, weight.shape[0])), weight)


class NNUE(nn.Module):
    """Two perspectives share `embedding` (F x H) and `bias`; the concatenated
    squared clipped accumulators feed a per-bucket linear readout and a shared
    32-wide dense layer whose clipped output feeds a per-bucket residual row.
    `buckets` is 1 (one head) or `features.BUCKETS` (one head per piece-count
    bucket, item 7)."""

    def __init__(self, hidden: int = 512, buckets: int = 1):
        super().__init__()
        if hidden % 32 or not 32 <= hidden <= 1024:
            raise ValueError("hidden must be a multiple of 32 from 32 to 1024")
        if buckets not in (1, BUCKETS):
            raise ValueError(f"buckets must be 1 or {BUCKETS}")
        self.hidden = hidden
        self.buckets = buckets
        self.embedding = nn.EmbeddingBag(FEATURES + 1, hidden, mode="sum", padding_idx=PAD)
        self.bias = nn.Parameter(torch.full((hidden,), 0.15))
        self.output = nn.Linear(2 * hidden, buckets)
        self.dense = nn.Linear(2 * hidden, DENSE)
        self.delta = nn.Linear(DENSE, buckets, bias=False)
        nn.init.normal_(self.embedding.weight, std=0.04)
        nn.init.normal_(self.output.weight, std=0.025)
        nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.dense.weight, std=0.03)
        nn.init.constant_(self.dense.bias, 0.2)
        nn.init.zeros_(self.delta.weight)
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()

    def forward(self, ids: torch.Tensor, bucket: torch.Tensor, qat: bool = False) -> torch.Tensor:
        """`ids` (N, 2, SLOTS) feature ids, `bucket` (N,) the head of every
        position (all zero with one head); returns the raw value (N,)."""
        n = ids.shape[0]
        weight = fake_quant(self.embedding.weight, QA) if qat else self.embedding.weight
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
        """Keep every parameter inside its integer range."""
        self.embedding.weight.clamp_(-8, 8)
        self.embedding.weight[PAD].zero_()
        self.bias.clamp_(-8, 8)
        self.output.weight.clamp_(-64, 64)
        self.output.bias.clamp_(-64, 64)
        self.dense.weight.clamp_(-127 / QB, 127 / QB)
        self.dense.bias.clamp_(-64, 64)
        self.delta.weight.clamp_(-64, 64)

    @torch.no_grad()
    def widen_hidden(self, hidden: int) -> NNUE:
        """A wider copy that evaluates identically at first: the existing rows,
        bias and head columns are kept, the new rows start at zero with a
        small bias so their squared clipped ReLU is active and trainable, and
        the new head columns are zero (DESIGN item 33)."""
        if hidden <= self.hidden or hidden % 32:
            raise ValueError(f"hidden must be a larger multiple of 32 than {self.hidden}")
        wide = NNUE(hidden, self.buckets)
        old, new = self.hidden, hidden
        wide.embedding.weight.zero_()
        wide.embedding.weight[:, :old].copy_(self.embedding.weight)
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
    def widen_buckets(self) -> NNUE:
        """A `BUCKETS`-head copy of a one-head network with every head equal
        to the single head, so the buckets start from the same evaluation."""
        if self.buckets != 1:
            raise ValueError("already bucketed")
        wide = NNUE(self.hidden, BUCKETS)
        wide.embedding.weight.copy_(self.embedding.weight)
        wide.bias.copy_(self.bias)
        wide.dense.weight.copy_(self.dense.weight)
        wide.dense.bias.copy_(self.dense.bias)
        wide.output.weight.copy_(self.output.weight.expand(BUCKETS, -1))
        wide.output.bias.copy_(self.output.bias.expand(BUCKETS))
        wide.delta.weight.copy_(self.delta.weight.expand(BUCKETS, -1))
        return wide
