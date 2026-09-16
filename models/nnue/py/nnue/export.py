"""The integer file the Rust crate loads (RPSNNUE1 version 6 or 8), an
independent NumPy evaluator over its bytes and the conversion of a version 6
file to version 8 (DESIGN items 4 and 10; runs/nnue_plan/format8_contract.md).

Layout, little endian: magic `RPSNNUE1`; u32 version (6 or 8); u32 features F
(1,004 or 13,640); u32 hidden H; u32 QA 255; u32 QB 64; f32 eval scale 600;
u32 buckets 1; then bias H i16; feature weights F x H i16 feature-major;
readout 2H i16 (mover then opponent); readout bias i32; u32 dense width 32;
dense weights 32 x 2H i8 row-major; dense biases 32 i32; residual outputs
32 i16. Every integer is validated on read: magic, a known version with its
feature count, H a multiple of 32 in 32..1024, the scales, one head, the
dense width and the exact file size.

    python -m nnue.export convert <version 6 file> <version 8 file>
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from .features import CONTEXTS, FORMAT6, FORMAT8, LAYOUTS, PIECE_ROWS, SLOTS, feature_ids
from .model import DENSE, EVAL_SCALE, NNUE, QA, QB
from .paths import sha256

MAGIC = b"RPSNNUE1"
HEADER = struct.Struct("<8sIIIIIfI")


def file_size(hidden: int, version: int = 8) -> int:
    features = LAYOUTS[version].features
    return (
        HEADER.size
        + 2 * hidden
        + 2 * features * hidden
        + (4 * hidden + 4)
        + 4
        + DENSE * 2 * hidden
        + 4 * DENSE
        + 2 * DENSE
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
    raw = HEADER.pack(MAGIC, layout.version, layout.features, h, QA, QB, EVAL_SCALE, 1)
    raw += quant(model.bias, QA, "<i2")
    raw += quant(model.rows(), QA, "<i2")
    raw += quant(model.output.weight, QB, "<i2")
    raw += quant(model.output.bias, QB, "<i4")
    raw += struct.pack("<I", DENSE)
    raw += quant(model.dense.weight, QB, "i1")
    raw += quant(model.dense.bias, QA * QB, "<i4")
    raw += quant(model.delta.weight, QB, "<i2")
    assert len(raw) == file_size(h, layout.version)
    info = {
        "version": layout.version,
        "features": layout.features,
        "hidden": h,
        "buckets": 1,
        "dense": DENSE,
        "qa": QA,
        "qb": QB,
        "eval_scale": EVAL_SCALE,
        **(metadata or {}),
    }
    return write_file(path, raw, info)


def read(path: Path) -> dict:
    """The file's integer arrays: bias (H), weights (F, H), dense (32, 2H),
    dense_bias (32), output (2H), output_bias (1), residual (32), with
    `version` (6 or 8) and `features` F."""
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
        raise ValueError("not an RPSNNUE1 version 6 or 8 file with the expected constants")
    if buckets != 1 or len(data) != file_size(hidden, version):
        raise ValueError("one output head and the exact file size are required")
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
        "output": take("<i2", 2 * hidden, (2 * hidden,)),
        "output_bias": take("<i4", 1, (1,)),
    }
    if take("<u4", 1, (1,))[0] != DENSE:
        raise ValueError("unexpected dense width")
    out["dense"] = take("i1", DENSE * 2 * hidden, (DENSE, 2 * hidden))
    out["dense_bias"] = take("<i4", DENSE, (DENSE,))
    out["residual"] = take("<i2", DENSE, (DENSE,))
    return out


def integer_eval(
    path: Path | dict, board: np.ndarray, since_capture: np.ndarray, clock: np.ndarray, chunk: int = 256
) -> np.ndarray:
    """Raw values from the file bytes alone, with the crate's integer
    arithmetic (i32 accumulators, i64 sums, ties to even). `path` is the file
    or an already-read one (`read`), so a caller evaluating position by
    position reads the file once."""
    net = path if isinstance(path, dict) else read(path)
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
        raw = (act * act * net["output"]).sum(axis=1) / (QA * QA * QB) + net["output_bias"][0] / QB
        dense_in = np.rint(act * act / QA).astype(np.int64)
        h = np.clip(np.rint((dense_in @ net["dense"].T + net["dense_bias"]) / QB), 0, QA).astype(np.int64)
        out.append(raw + (h * net["residual"]).sum(axis=1) / (QA * QB))
    return np.concatenate(out) if out else np.zeros(0)


def load(path: Path) -> NNUE:
    """The file's weights as a float model (for fine-tuning or checks). A
    version 8 table is split into the shared factor (the mean over the
    contexts) and the residuals, so the served sum is the file's row."""
    net = read(path)
    layout = LAYOUTS[net["version"]]
    model = NNUE(net["hidden"], net["version"])
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
        model.output.weight.copy_(torch.from_numpy(net["output"] / QB).float().reshape(1, -1))
        model.output.bias.copy_(torch.from_numpy(net["output_bias"] / QB).float())
        model.delta.weight.copy_(torch.from_numpy(net["residual"] / QB).float().reshape(1, -1))
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
        or len(data) != file_size(hidden, FORMAT6.version)
    ):
        raise ValueError("not a one-head RPSNNUE1 version 6 file")
    at = HEADER.size + 2 * hidden
    weights = np.frombuffer(data, "<i2", features * hidden, at).reshape(features, hidden)
    raw = HEADER.pack(MAGIC, FORMAT8.version, FORMAT8.features, hidden, QA, QB, scale, buckets)
    raw += data[HEADER.size : at]
    raw += np.tile(weights[:PIECE_ROWS], (CONTEXTS, 1)).tobytes() + weights[PIECE_ROWS:].tobytes()
    raw += data[at + 2 * features * hidden :]
    assert len(raw) == file_size(hidden, FORMAT8.version)
    info = {
        "version": FORMAT8.version,
        "features": FORMAT8.features,
        "hidden": hidden,
        "buckets": 1,
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
