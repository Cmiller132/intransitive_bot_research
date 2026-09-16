"""The integer file the Rust crate loads (RPSNNUE1 version 6, 8 or 9), an
independent NumPy evaluator over its bytes and the conversions of a version 6
file to version 8 and of a version 8 file to version 9 (DESIGN items 4 and 10;
runs/nnue_plan/format8_contract.md and models/nnue/FORMAT9.md).

Layout, little endian: magic `RPSNNUE1`; u32 version (6, 8 or 9); u32 features
F (1,004, 13,640 or 13,648); u32 hidden H; u32 QA 255; u32 QB 64; f32 eval
scale 600; u32 heads (1, or 8 in version 9); then bias H i16; feature weights
F x H i16 feature-major; then one complete head after another: readout 2H i16
(mover then opponent); readout bias i32; u32 dense width 32 (versions 6 and 8
only, version 9 fixes the width at 32); dense weights 32 x 2H i8 row-major;
dense biases 32 i32; residual outputs 32 i16. Version 9 selects its head by
the pieces on the board. Every integer is validated on read: magic, a known
version with its feature count and head count, H a multiple of 32 in 32..1024,
the scales, the dense width and the exact file size.

    python -m nnue.export convert [--to 8|9] <source file> <target file>
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from .features import (
    CONTEXTS,
    FORMAT6,
    FORMAT8,
    FORMAT9,
    GOAL_ROWS,
    LAYOUTS,
    PIECE_ROWS,
    feature_ids,
    feature_ids9,
    piece_bucket,
)
from .model import DENSE, EVAL_SCALE, NNUE, QA, QB
from .paths import sha256

MAGIC = b"RPSNNUE1"
HEADER = struct.Struct("<8sIIIIIfI")


def head_size(hidden: int, version: int = 8) -> int:
    """One head: readout, readout bias, the dense width before version 9,
    dense weights, dense biases and the residual readout."""
    width = 0 if LAYOUTS[version].goal else 4
    return 4 * hidden + 4 + width + DENSE * 2 * hidden + 4 * DENSE + 2 * DENSE


def file_size(hidden: int, version: int = 8) -> int:
    layout = LAYOUTS[version]
    return HEADER.size + 2 * hidden + 2 * layout.features * hidden + layout.heads * head_size(hidden, version)


def write_file(path: Path, raw: bytes, info: dict) -> dict:
    """Write the bytes atomically and the `.json` sidecar beside them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    info = {"format": "RPSNNUE1", **info, "bytes": len(raw), "sha256": sha256(path)}
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


@torch.no_grad()
def export(model: NNUE, path: Path, metadata: dict | None = None) -> dict:
    """Write `path` and `path.json`; returns the sidecar's content."""

    def quant(t, scale, dtype):
        values = np.rint(t.detach().cpu().numpy() * scale)
        info = np.iinfo(dtype)
        if values.min() < info.min or values.max() > info.max:
            raise ValueError(f"quantised values leave {dtype}: {values.min()}..{values.max()}")
        return values.astype(dtype).tobytes()

    h, layout = model.hidden, model.layout
    raw = HEADER.pack(MAGIC, layout.version, layout.features, h, QA, QB, EVAL_SCALE, layout.heads)
    raw += quant(model.bias, QA, "<i2")
    raw += quant(model.rows(), QA, "<i2")
    for index in range(layout.heads):
        weight, bias, dense, dense_bias, delta = model.head(index)
        raw += quant(weight, QB, "<i2")
        raw += quant(bias, QB, "<i4")
        if not layout.goal:
            raw += struct.pack("<I", DENSE)
        raw += quant(dense, QB, "i1")
        raw += quant(dense_bias, QA * QB, "<i4")
        raw += quant(delta, QB, "<i2")
    assert len(raw) == file_size(h, layout.version)
    info = {
        "version": layout.version,
        "features": layout.features,
        "hidden": h,
        "buckets": layout.heads,
        "dense": DENSE,
        "qa": QA,
        "qb": QB,
        "eval_scale": EVAL_SCALE,
        **(metadata or {}),
    }
    return write_file(path, raw, info)


def read(path: Path) -> dict:
    """The file's integer arrays: bias (H), weights (F, H), dense (32, 2H),
    dense_bias (32), output (2H), output_bias (heads), residual (32), with
    `version` (6, 8 or 9) and `features` F. A version 9 file carries eight
    heads, so `output`, `dense`, `dense_bias` and `residual` gain a leading
    head axis; the arrays of a one-head file keep their shape."""
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("not an RPSNNUE1 file")
    magic, version, features, hidden, qa, qb, scale, buckets = HEADER.unpack_from(data, 0)
    layout = LAYOUTS.get(version)
    if (
        magic != MAGIC
        or layout is None
        or layout.features != features
        or (qa, qb, scale) != (QA, QB, EVAL_SCALE)
        or hidden % 32
        or not 32 <= hidden <= 1024
    ):
        raise ValueError("not an RPSNNUE1 version 6, 8 or 9 file with the expected constants")
    if buckets != layout.heads or len(data) != file_size(hidden, version):
        raise ValueError("the version's head count and the exact file size are required")
    at = HEADER.size

    def take(dtype, count, shape):
        nonlocal at
        out = np.frombuffer(data, dtype, count, at).reshape(shape).astype(np.int64)
        at += count * np.dtype(dtype).itemsize
        return out

    out = {
        "version": version,
        "features": features,
        "hidden": hidden,
        "eval_scale": scale,
        "bias": take("<i2", hidden, (hidden,)),
        "weights": take("<i2", features * hidden, (features, hidden)),
    }
    heads: dict[str, list] = {name: [] for name in ("output", "output_bias", "dense", "dense_bias", "residual")}
    for _ in range(layout.heads):
        heads["output"].append(take("<i2", 2 * hidden, (2 * hidden,)))
        heads["output_bias"].append(take("<i4", 1, ()))
        if not layout.goal and take("<u4", 1, (1,))[0] != DENSE:
            raise ValueError("unexpected dense width")
        heads["dense"].append(take("i1", DENSE * 2 * hidden, (DENSE, 2 * hidden)))
        heads["dense_bias"].append(take("<i4", DENSE, (DENSE,)))
        heads["residual"].append(take("<i2", DENSE, (DENSE,)))
    for name, values in heads.items():
        stacked = np.stack(values)
        # One head keeps the arrays of formats 6 and 8; the biases stay (heads,).
        out[name] = stacked if layout.heads > 1 or name == "output_bias" else stacked[0]
    return out


def integer_eval(
    path: Path, board: np.ndarray, since_capture: np.ndarray, clock: np.ndarray, chunk: int = 256
) -> np.ndarray:
    """Raw values from the file bytes alone, with the crate's integer
    arithmetic (i32 accumulators, i64 sums, ties to even). A version 9 file
    reads its head from the pieces on the board."""
    net = read(path)
    layout, hidden = LAYOUTS[net["version"]], net["hidden"]
    weights = np.concatenate([net["weights"], np.zeros((1, hidden), dtype=np.int64)])  # the padding row
    board = np.asarray(board, dtype=np.uint8).reshape(-1, 81)
    since = np.asarray(since_capture).reshape(-1)
    clock = np.asarray(clock).reshape(-1)

    def value(act: np.ndarray, head: int) -> np.ndarray:
        output, dense, dense_bias, residual = (
            net[name][head] if layout.heads > 1 else net[name] for name in ("output", "dense", "dense_bias", "residual")
        )
        raw = (act * act * output).sum(axis=1) / (QA * QA * QB) + net["output_bias"][head] / QB
        dense_in = np.rint(act * act / QA).astype(np.int64)
        h = np.clip(np.rint((dense_in @ dense.T + dense_bias) / QB), 0, QA).astype(np.int64)
        return raw + (h * residual).sum(axis=1) / (QA * QB)

    out = []
    for start in range(0, len(board), chunk):
        part = board[start : start + chunk]
        since_part, clock_part = since[start : start + chunk], clock[start : start + chunk]
        if layout.goal:
            ids = feature_ids9(part, since_part, clock_part)
        else:
            ids = layout.rows(feature_ids(part, since_part, clock_part))
        acc = weights[ids.reshape(-1, layout.slots)].sum(axis=1) + net["bias"]
        act = np.clip(acc, 0, QA).reshape(-1, 2 * hidden)
        if layout.heads == 1:
            out.append(value(act, 0))
            continue
        bucket = piece_bucket(np.count_nonzero(part, axis=1), layout.heads)
        values = np.zeros(len(act))
        for head in np.unique(bucket):
            rows = bucket == head
            values[rows] = value(act[rows], int(head))
        out.append(values)
    return np.concatenate(out) if out else np.zeros(0)


def load(path: Path) -> NNUE:
    """The file's weights as a float model (for fine-tuning or checks). A
    version 8 table is split into the shared factor (the mean over the
    contexts) and the residuals, so the served sum is the file's row; a
    version 9 file's eight heads are split the same way, the shared head being
    their mean."""
    net = read(path)
    layout = LAYOUTS[net["version"]]
    model = NNUE(net["hidden"], net["version"])
    rows = net["weights"] / QA
    piece = rows[: layout.attack_base]

    def share(values: np.ndarray, shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """The shared head and its residuals: `values` is (heads, ...)."""
        mean = np.asarray(values.mean(axis=0))
        shared = torch.from_numpy(mean).float().reshape(shape)
        return shared, torch.from_numpy(values - mean).float().reshape(layout.heads, *shape)

    with torch.no_grad():
        if model.context is None:
            model.piece.copy_(torch.from_numpy(piece).float())
        else:
            piece = piece.reshape(CONTEXTS, PIECE_ROWS, -1)
            mean = piece.mean(axis=0)
            model.piece.copy_(torch.from_numpy(mean).float())
            model.context.copy_(torch.from_numpy(piece - mean).float())
        model.attack.copy_(torch.from_numpy(rows[layout.attack_base : layout.elapsed_base]).float())
        model.clock.copy_(torch.from_numpy(rows[layout.elapsed_base : layout.goal_base]).float())
        if model.goal is not None:
            model.goal.copy_(torch.from_numpy(rows[layout.goal_base :]).float())
        model.bias.copy_(torch.from_numpy(net["bias"] / QA).float())
        if layout.heads == 1:
            model.dense.weight.copy_(torch.from_numpy(net["dense"] / QB).float())
            model.dense.bias.copy_(torch.from_numpy(net["dense_bias"] / (QA * QB)).float())
            model.output.weight.copy_(torch.from_numpy(net["output"] / QB).float().reshape(1, -1))
            model.output.bias.copy_(torch.from_numpy(net["output_bias"] / QB).float())
            model.delta.weight.copy_(torch.from_numpy(net["residual"] / QB).float().reshape(1, -1))
        else:
            hidden = net["hidden"]
            for values, scale, shape, layer, residual in (
                (net["output"], QB, (1, 2 * hidden), model.output.weight, model.output_head),
                (net["output_bias"], QB, (1,), model.output.bias, model.output_head_bias),
                (net["dense"], QB, (DENSE, 2 * hidden), model.dense.weight, model.dense_head),
                (net["dense_bias"], QA * QB, (DENSE,), model.dense.bias, model.dense_head_bias),
                (net["residual"], QB, (1, DENSE), model.delta.weight, model.delta_head),
            ):
                shared, parts = share(values / scale, shape)
                layer.copy_(shared)
                residual.copy_(parts)
    return model


def convert(source: Path, target: Path, metadata: dict | None = None, to: int = 8) -> dict:
    """A version 6 file as a version 8 file (`to` 8): every piece-square row is
    copied into all 27 contexts, the attacked and clock rows move to their
    bases, every other byte is kept. A version 8 file as a version 9 file
    (`to` 9): the shared table is kept, the eight goal rows are zero and the
    one head is repeated into all eight, its dense-width field dropped. Either
    way the target holds identical raw values on every position."""
    source, target = Path(source), Path(target)
    data = source.read_bytes()
    old = FORMAT8 if to == FORMAT9.version else FORMAT6
    new = FORMAT9 if to == FORMAT9.version else FORMAT8
    magic, version, features, hidden, qa, qb, scale, buckets = HEADER.unpack_from(data, 0)
    if (
        magic != MAGIC
        or version != old.version
        or features != old.features
        or (qa, qb) != (QA, QB)
        or buckets != old.heads
        or len(data) != file_size(hidden, old.version)
    ):
        raise ValueError(f"not a one-head RPSNNUE1 version {old.version} file")
    at = HEADER.size + 2 * hidden
    table_end = at + 2 * features * hidden
    raw = HEADER.pack(MAGIC, new.version, new.features, hidden, QA, QB, scale, new.heads)
    raw += data[HEADER.size : at]  # the bias
    if to == FORMAT9.version:
        head = data[table_end:]
        # Version 9 has no dense-width field; the rest of the head is copied.
        readout = 4 * hidden + 4
        raw += data[at:table_end] + np.zeros((GOAL_ROWS, hidden), "<i2").tobytes()
        raw += (head[:readout] + head[readout + 4 :]) * new.heads
    else:
        weights = np.frombuffer(data, "<i2", features * hidden, at).reshape(features, hidden)
        raw += np.tile(weights[:PIECE_ROWS], (CONTEXTS, 1)).tobytes() + weights[PIECE_ROWS:].tobytes()
        raw += data[table_end:]
    assert len(raw) == file_size(hidden, new.version)
    info = {
        "version": new.version,
        "features": new.features,
        "hidden": hidden,
        "buckets": new.heads,
        "dense": DENSE,
        "qa": QA,
        "qb": QB,
        "eval_scale": scale,
        "converted_from": {"path": str(source), "sha256": sha256(source), "version": old.version},
        **(metadata or {}),
    }
    return write_file(target, raw, info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["convert"])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--to", type=int, default=FORMAT8.version, choices=(FORMAT8.version, FORMAT9.version))
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.target, to=args.to), indent=2))
