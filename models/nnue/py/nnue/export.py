"""The integer file the Rust crate loads (RPSNNUE1 version 6 or 7), an
independent NumPy evaluator over its bytes, and the conversion of the
prototype's version 3 files (DESIGN items 4, 10 and 36).

Layout, little endian: magic `RPSNNUE1`; u32 version (6 or 7); u32 features F
(1004 for version 6, 1022 with the race rows for version 7);
u32 hidden H; u32 QA 255; u32 QB 64; f32 eval scale 600; u32 buckets B;
then bias H i16; feature weights F x H i16 feature-major; readouts B x 2H
i16 bucket-major (mover then opponent); readout biases B i32; u32 dense width
32; dense weights 32 x 2H i8 row-major; dense biases 32 i32; residual outputs
B x 32 i16 bucket-major. Version 3 files (the prototype: 972
features, one head, the same arithmetic) convert by inserting zero clock rows.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

from .features import (
    BUCKETS,
    CLOCK_BUCKETS,
    FEATURES,
    FORMAT6_FEATURES,
    PIECE_ROWS,
    SLOTS,
    feature_ids,
    piece_bucket,
)
from .model import DENSE, EVAL_SCALE, NNUE, QA, QB

MAGIC = b"RPSNNUE1"
VERSIONS = {FORMAT6_FEATURES: 6, FEATURES: 7}  # file version by feature count
HEADER = struct.Struct("<8sIIIIIfI")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size(hidden: int, buckets: int, features: int = FEATURES) -> int:
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


@torch.no_grad()
def export(model: NNUE, path: Path, metadata: dict | None = None) -> dict:
    """Write `path` and `path.json`; returns the sidecar's content."""

    def quant(t, scale, dtype):
        values = np.rint(t.detach().cpu().numpy() * scale)
        info = np.iinfo(dtype)
        if values.min() < info.min or values.max() > info.max:
            raise ValueError(f"quantised values leave {dtype}: {values.min()}..{values.max()}")
        return values.astype(dtype).tobytes()

    h, features = model.hidden, model.features
    raw = HEADER.pack(MAGIC, VERSIONS[features], features, h, QA, QB, EVAL_SCALE, model.buckets)
    raw += quant(model.bias, QA, "<i2")
    raw += quant(model.embedding.weight[:features], QA, "<i2")
    raw += quant(model.output.weight, QB, "<i2")
    raw += quant(model.output.bias, QB, "<i4")
    raw += struct.pack("<I", DENSE)
    raw += quant(model.dense.weight, QB, "i1")
    raw += quant(model.dense.bias, QA * QB, "<i4")
    raw += quant(model.delta.weight, QB, "<i2")
    assert len(raw) == file_size(h, model.buckets, features)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    info = {
        "format": "RPSNNUE1",
        "version": VERSIONS[features],
        "features": features,
        "hidden": h,
        "buckets": model.buckets,
        "dense": DENSE,
        "qa": QA,
        "qb": QB,
        "eval_scale": EVAL_SCALE,
        "bytes": len(raw),
        "sha256": sha256(path),
        **(metadata or {}),
    }
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def read(path: Path) -> dict:
    """The file's integer arrays: bias (H), weights (F, H), dense (32, 2H),
    dense_bias (32), output (B, 2H), output_bias (B), residual (B, 32);
    `features` is F (1004 or 1022)."""
    data = Path(path).read_bytes()
    magic, version, features, hidden, qa, qb, scale, buckets = HEADER.unpack_from(data, 0)
    if magic != MAGIC or VERSIONS.get(features) != version or (qa, qb) != (QA, QB):
        raise ValueError("not an RPSNNUE1 version 6 or 7 file with the expected constants")
    if buckets not in (1, BUCKETS) or len(data) != file_size(hidden, buckets, features):
        raise ValueError("unexpected bucket count or file size")
    at = HEADER.size

    def take(dtype, count, shape):
        nonlocal at
        out = np.frombuffer(data, dtype, count, at).reshape(shape).astype(np.int64)
        at += count * np.dtype(dtype).itemsize
        return out

    out = {
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


def integer_eval(path: Path, board: np.ndarray, since_capture: np.ndarray, clock: np.ndarray) -> np.ndarray:
    """Raw values from the file bytes alone, with the crate's integer
    arithmetic (i32 accumulators, i64 sums, ties to even)."""
    net = read(path)
    hidden, qa, qb = net["hidden"], QA, QB
    ids = feature_ids(board, since_capture, clock).reshape(-1, SLOTS)
    # Rows the file lacks (the race rows of a format 6 file) and the padding row are zero.
    weights = np.zeros((FEATURES + 1, hidden), dtype=np.int64)
    weights[: net["features"]] = net["weights"]
    acc = weights[ids].sum(axis=1) + net["bias"]
    act = np.clip(acc, 0, qa).reshape(-1, 2 * hidden)
    bucket = piece_bucket(board) if net["buckets"] > 1 else np.zeros(len(act), dtype=np.int64)
    out = net["output"][bucket]
    raw = (act * act * out).sum(axis=1) / (qa * qa * qb) + net["output_bias"][bucket] / qb
    dense_in = np.rint(act * act / qa).astype(np.int64)
    h = np.clip(np.rint((dense_in @ net["dense"].T + net["dense_bias"]) / qb), 0, qa).astype(np.int64)
    raw += (h * net["residual"][bucket]).sum(axis=1) / (qa * qb)
    return raw


def load(path: Path) -> NNUE:
    """The file's weights as a float model (for fine-tuning or checks)."""
    net = read(path)
    model = NNUE(net["hidden"], net["buckets"], net["features"])
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.bias.copy_(torch.from_numpy(net["bias"] / QA).float())
        model.embedding.weight[: net["features"]].copy_(torch.from_numpy(net["weights"] / QA).float())
        model.dense.weight.copy_(torch.from_numpy(net["dense"] / QB).float())
        model.dense.bias.copy_(torch.from_numpy(net["dense_bias"] / (QA * QB)).float())
        model.output.weight.copy_(torch.from_numpy(net["output"] / QB).float())
        model.output.bias.copy_(torch.from_numpy(net["output_bias"] / QB).float())
        model.delta.weight.copy_(torch.from_numpy(net["residual"] / QB).float())
    return model


def convert_v3(source: Path, target: Path, metadata: dict | None = None) -> dict:
    """The prototype's version 3 file as a version 6 file with zero clock rows;
    identical raw values on every position (DESIGN item 4)."""
    data = Path(source).read_bytes()
    version, features, hidden, qa, qb, scale = struct.unpack_from("<IIIIIf", data, 8)
    if data[:8] != MAGIC or version != 3 or features != 2 * PIECE_ROWS or (qa, qb) != (QA, QB):
        raise ValueError("not a prototype version 3 file")
    at = 32
    bias = np.frombuffer(data, "<i2", hidden, at)
    at += 2 * hidden
    weights = np.frombuffer(data, "<i2", features * hidden, at).reshape(features, hidden)
    at += 2 * features * hidden
    output = np.frombuffer(data, "<i2", 2 * hidden, at)
    at += 4 * hidden
    output_bias = struct.unpack_from("<i", data, at)[0]
    at += 4
    if struct.unpack_from("<I", data, at)[0] != DENSE:
        raise ValueError("unexpected dense width")
    at += 4
    dense = np.frombuffer(data, "i1", DENSE * 2 * hidden, at).reshape(DENSE, 2 * hidden)
    at += DENSE * 2 * hidden
    dense_bias = np.frombuffer(data, "<i4", DENSE, at)
    at += 4 * DENSE
    residual = np.frombuffer(data, "<i2", DENSE, at)
    at += 2 * DENSE
    if at != len(data):
        raise ValueError("trailing bytes in the version 3 file")
    raw = HEADER.pack(MAGIC, VERSIONS[FORMAT6_FEATURES], FORMAT6_FEATURES, hidden, QA, QB, scale, 1)
    raw += bias.tobytes()
    raw += weights.tobytes() + np.zeros((2 * CLOCK_BUCKETS, hidden), dtype="<i2").tobytes()
    raw += output.tobytes() + struct.pack("<i", output_bias) + struct.pack("<I", DENSE)
    raw += dense.tobytes() + dense_bias.tobytes() + residual.tobytes()
    assert len(raw) == file_size(hidden, 1, FORMAT6_FEATURES)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    info = {
        "format": "RPSNNUE1",
        "version": VERSIONS[FORMAT6_FEATURES],
        "features": FORMAT6_FEATURES,
        "hidden": hidden,
        "buckets": 1,
        "dense": DENSE,
        "qa": QA,
        "qb": QB,
        "eval_scale": scale,
        "bytes": len(raw),
        "sha256": sha256(target),
        "converted_from": {"path": str(source), "sha256": sha256(Path(source)), "version": 3},
        **(metadata or {}),
    }
    target.with_suffix(target.suffix + ".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
