"""The integer file the Rust crate loads (RPSNNUE1 version 6 or 8), an
independent NumPy evaluator over its bytes and the conversion of a version 6
file to version 8 (DESIGN items 4 and 10; runs/nnue_plan/format8_contract.md).

Layout, little endian: magic `RPSNNUE1`; u32 version (6 or 8); u32 features F
(1,004 or 13,640); u32 hidden H; u32 QA 255; u32 QB 64; f32 eval scale 600;
u32 buckets B; then bias H i16; feature weights F x H i16 feature-major;
readouts B x 2H i16 bucket-major (mover then opponent); readout biases B i32;
u32 dense width 32; dense weights 32 x 2H i8 row-major; dense biases 32 i32;
residual outputs B x 32 i16 bucket-major.

    python -m nnue.export convert <version 6 file> <version 8 file>
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from .features import BUCKETS, CONTEXTS, FORMAT6, FORMAT8, LAYOUTS, PIECE_ROWS, SLOTS, feature_ids, piece_bucket
from .model import DENSE, EVAL_SCALE, NNUE, QA, QB

MAGIC = b"RPSNNUE1"
HEADER = struct.Struct("<8sIIIIIfI")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size(hidden: int, buckets: int, version: int = 8) -> int:
    features = LAYOUTS[version].features
    return (
        HEADER.size
        + 2 * hidden
        + 2 * features * hidden
        + buckets * (2 * 2 * hidden + 4)
        + 4
        + DENSE * 2 * hidden
        + 4 * DENSE
        + buckets * 2 * DENSE
    )


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
    raw = HEADER.pack(MAGIC, layout.version, layout.features, h, QA, QB, EVAL_SCALE, model.buckets)
    raw += quant(model.bias, QA, "<i2")
    raw += quant(model.rows(), QA, "<i2")
    raw += quant(model.output.weight, QB, "<i2")
    raw += quant(model.output.bias, QB, "<i4")
    raw += struct.pack("<I", DENSE)
    raw += quant(model.dense.weight, QB, "i1")
    raw += quant(model.dense.bias, QA * QB, "<i4")
    raw += quant(model.delta.weight, QB, "<i2")
    assert len(raw) == file_size(h, model.buckets, layout.version)
    info = {
        "version": layout.version,
        "features": layout.features,
        "hidden": h,
        "buckets": model.buckets,
        "dense": DENSE,
        "qa": QA,
        "qb": QB,
        "eval_scale": EVAL_SCALE,
        **(metadata or {}),
    }
    return write_file(path, raw, info)


def read(path: Path) -> dict:
    """The file's integer arrays: bias (H), weights (F, H), dense (32, 2H),
    dense_bias (32), output (B, 2H), output_bias (B), residual (B, 32), with
    `version` (6 or 8) and `features` F."""
    data = Path(path).read_bytes()
    magic, version, features, hidden, qa, qb, scale, buckets = HEADER.unpack_from(data, 0)
    layout = LAYOUTS.get(version)
    if magic != MAGIC or layout is None or layout.features != features or (qa, qb) != (QA, QB):
        raise ValueError("not an RPSNNUE1 version 6 or 8 file with the expected constants")
    if buckets not in (1, BUCKETS) or len(data) != file_size(hidden, buckets, version):
        raise ValueError("unexpected bucket count or file size")
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
        "buckets": buckets,
        "eval_scale": scale,
        "bias": take("<i2", hidden, (hidden,)),
        "weights": take("<i2", features * hidden, (features, hidden)),
        "output": take("<i2", buckets * 2 * hidden, (buckets, 2 * hidden)),
        "output_bias": take("<i4", buckets, (buckets,)),
    }
    if take("<u4", 1, (1,))[0] != DENSE:
        raise ValueError("unexpected dense width")
    out["dense"] = take("i1", DENSE * 2 * hidden, (DENSE, 2 * hidden))
    out["dense_bias"] = take("<i4", DENSE, (DENSE,))
    out["residual"] = take("<i2", buckets * DENSE, (buckets, DENSE))
    return out


def integer_eval(
    path: Path, board: np.ndarray, since_capture: np.ndarray, clock: np.ndarray, chunk: int = 256
) -> np.ndarray:
    """Raw values from the file bytes alone, with the crate's integer
    arithmetic (i32 accumulators, i64 sums, ties to even)."""
    net = read(path)
    layout, hidden = LAYOUTS[net["version"]], net["hidden"]
    weights = np.concatenate([net["weights"], np.zeros((1, hidden), dtype=np.int64)])  # the padding row
    board = np.asarray(board, dtype=np.uint8).reshape(-1, 81)
    since = np.asarray(since_capture).reshape(-1)
    clock = np.asarray(clock).reshape(-1)
    out = []
    for start in range(0, len(board), chunk):
        part = board[start : start + chunk]
        ids = layout.rows(feature_ids(part, since[start : start + chunk], clock[start : start + chunk]))
        acc = weights[ids.reshape(-1, SLOTS)].sum(axis=1) + net["bias"]
        act = np.clip(acc, 0, QA).reshape(-1, 2 * hidden)
        bucket = piece_bucket(part) if net["buckets"] > 1 else np.zeros(len(act), dtype=np.int64)
        readout = net["output"][bucket]
        raw = (act * act * readout).sum(axis=1) / (QA * QA * QB) + net["output_bias"][bucket] / QB
        dense_in = np.rint(act * act / QA).astype(np.int64)
        h = np.clip(np.rint((dense_in @ net["dense"].T + net["dense_bias"]) / QB), 0, QA).astype(np.int64)
        out.append(raw + (h * net["residual"][bucket]).sum(axis=1) / (QA * QB))
    return np.concatenate(out) if out else np.zeros(0)


def load(path: Path) -> NNUE:
    """The file's weights as a float model (for fine-tuning or checks). A
    version 8 table is split into the shared factor (the mean over the
    contexts) and the residuals, so the served sum is the file's row."""
    net = read(path)
    layout = LAYOUTS[net["version"]]
    model = NNUE(net["hidden"], net["buckets"], net["version"])
    rows = net["weights"] / QA
    piece = rows[: layout.attack_base]
    with torch.no_grad():
        if model.context is None:
            model.piece.copy_(torch.from_numpy(piece).float())
        else:
            piece = piece.reshape(CONTEXTS, PIECE_ROWS, -1)
            mean = piece.mean(axis=0)
            model.piece.copy_(torch.from_numpy(mean).float())
            model.context.copy_(torch.from_numpy(piece - mean).float())
        model.attack.copy_(torch.from_numpy(rows[layout.attack_base : layout.elapsed_base]).float())
        model.clock.copy_(torch.from_numpy(rows[layout.elapsed_base :]).float())
        model.bias.copy_(torch.from_numpy(net["bias"] / QA).float())
        model.dense.weight.copy_(torch.from_numpy(net["dense"] / QB).float())
        model.dense.bias.copy_(torch.from_numpy(net["dense_bias"] / (QA * QB)).float())
        model.output.weight.copy_(torch.from_numpy(net["output"] / QB).float())
        model.output.bias.copy_(torch.from_numpy(net["output_bias"] / QB).float())
        model.delta.weight.copy_(torch.from_numpy(net["residual"] / QB).float())
    return model


def convert(source: Path, target: Path, metadata: dict | None = None) -> dict:
    """A version 6 file as a version 8 file: every piece-square row is copied
    into all 27 contexts, the attacked and clock rows move to their bases,
    every other byte is kept; identical raw values on every position."""
    data = Path(source).read_bytes()
    magic, version, features, hidden, qa, qb, scale, buckets = HEADER.unpack_from(data, 0)
    if (
        magic != MAGIC
        or version != FORMAT6.version
        or features != FORMAT6.features
        or (qa, qb) != (QA, QB)
        or buckets != 1
        or len(data) != file_size(hidden, buckets, FORMAT6.version)
    ):
        raise ValueError("not a one-head RPSNNUE1 version 6 file")
    at = HEADER.size + 2 * hidden
    weights = np.frombuffer(data, "<i2", features * hidden, at).reshape(features, hidden)
    raw = HEADER.pack(MAGIC, FORMAT8.version, FORMAT8.features, hidden, QA, QB, scale, buckets)
    raw += data[HEADER.size : at]
    raw += np.tile(weights[:PIECE_ROWS], (CONTEXTS, 1)).tobytes() + weights[PIECE_ROWS:].tobytes()
    raw += data[at + 2 * features * hidden :]
    assert len(raw) == file_size(hidden, buckets, FORMAT8.version)
    info = {
        "version": FORMAT8.version,
        "features": FORMAT8.features,
        "hidden": hidden,
        "buckets": buckets,
        "dense": DENSE,
        "qa": QA,
        "qb": QB,
        "eval_scale": scale,
        "converted_from": {"path": str(source), "sha256": sha256(source), "version": FORMAT6.version},
        **(metadata or {}),
    }
    return write_file(target, raw, info)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "convert":
        raise SystemExit(__doc__)
    print(json.dumps(convert(Path(sys.argv[2]), Path(sys.argv[3])), indent=2))
