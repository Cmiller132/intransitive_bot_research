"""Teacher labels in bulk on the GPU (DESIGN item 22 when the GPU is free):
conv's batched Gumbel search, run from a training checkpoint, values
thousands of positions per call.

    python -m nnue.gpu_label --input <set> --out <set> --ckpt runs/conv_g128/ckpt_000040.pt [--teacher conv|sq]
        [--sims 128] [--batch 4096] [--rows N] [--seed S]

`--teacher` names the model package whose checkpoint and batched search
label the rows (conv by default; sq with `--ckpt weights/sq_g128.pt`); the
teacher is chosen by arena rating at the time.

Measured on the RTX 4070 Ti with conv_g128@40: about 55 rows/s uncompiled at
128 simulations and batch 4,096; the compiled network (the run's own
`compile_net`) is several times faster after its first-call compilation.

The output has the same shape as `nnue.label`: exact proofs from the engine
(kind 2), otherwise the search's root value (kind 3), under the site clock.
The teacher is the checkpoint's EMA weights through the same evaluator
self-play uses; the conv package is imported from models/conv/py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import data
from .games import SITE_CLOCK
from .label import proofs
from .paths import data_dir, workspace_root


def teacher(kind: str, ckpt: Path, device: torch.device, sims: int, batch: int, seed: int):
    """`(search_values, config)`: a function from canonical boards, counters and
    plies (tensors on `device`, `batch` rows) to root values, built from a
    conv or sq training checkpoint with the package's own search and evaluator."""
    sys.path.insert(0, str(workspace_root() / "models" / kind / "py"))
    if kind == "conv":
        from conv import kernels
        from conv.model import build, config_of, load_checkpoint
        from conv.search import GumbelSearch
        from conv.train import compile_net, student_evaluator

        checkpoint = load_checkpoint(str(ckpt), device)
        cfg = config_of(checkpoint)
        cfg = replace(cfg, search=replace(cfg.search, sims=sims, cheap_sims=min(cfg.search.cheap_sims, sims)))
        net = build(cfg, checkpoint, device, "ema").eval()
        evaluate = student_evaluator(compile_net(net, cfg), cfg.play.draw_kernel_width)
        clock = torch.full((batch,), SITE_CLOCK, dtype=torch.int32, device=device)
        search = GumbelSearch(
            cfg.search, batch, device, clock, cfg.rules.clock_penalty, cfg.rules.max_plies, seed, None
        )

        def values_of(b, s, p):
            _, legal, _ = kernels.derive_batch(b, s, p, clock)
            search.forget()
            return search(b, s, p, legal, evaluate, True).value

    elif kind == "sq":
        from sq import kernels
        from sq.model import build, config_of, load_checkpoint
        from sq.search import GumbelSearch
        from sq.train import acting, compile_net

        checkpoint = load_checkpoint(str(ckpt), device)
        cfg = config_of(checkpoint)
        cfg = replace(cfg, search=replace(cfg.search, sims=sims, cheap_sims=min(cfg.search.cheap_sims, sims)))
        net = build(cfg, checkpoint, device, "ema").eval()
        forward = acting(compile_net(net, cfg))
        clock = torch.tensor(SITE_CLOCK, dtype=torch.int32, device=device)
        search = GumbelSearch(
            cfg.search, batch, device, clock, cfg.rules.clock_penalty, cfg.rules.max_plies, seed, None
        )

        def values_of(b, s, p):
            planes, legal, _ = kernels.derive_batch(b, s, p)
            search.forget()
            return search(b, s, p, legal, planes, forward, True)[1]

    else:
        raise ValueError(f"unknown teacher {kind!r}; conv or sq")
    return values_of, cfg


@torch.no_grad()
def values(
    values_of, device: torch.device, board: np.ndarray, since: np.ndarray, ply: np.ndarray, batch: int
) -> np.ndarray:
    """Root values of every row, searched `batch` rows at a time (the last
    batch padded with the first row)."""
    n = len(board)
    out = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch):
        idx = np.arange(start, min(start + batch, n))
        pad = np.concatenate([idx, np.zeros(batch - len(idx), dtype=np.int64)])
        b = torch.from_numpy(board[pad].astype(np.int8)).to(device)
        s = torch.from_numpy(since[pad].astype(np.int32)).to(device)
        p = torch.from_numpy(ply[pad].astype(np.int32)).to(device)
        out[idx] = values_of(b, s, p)[: len(idx)].float().cpu().numpy()
    return out


def label(
    input_set: str, out: str, ckpt: Path, sims: int, batch: int, sample: int, seed: int, teacher_kind: str = "conv"
) -> Path:
    started = time.perf_counter()
    source = Path(data_dir(input_set))
    rows = {name: np.load(source / f"{name}.npy") for name in data.FIELDS}
    n = len(rows["board"])
    if sample and sample < n:
        chosen = np.sort(np.random.default_rng(seed).choice(n, sample, replace=False))
        rows = {name: value[chosen] for name, value in rows.items()}
        n = sample
    device = torch.device("cuda")
    values_of, cfg = teacher(teacher_kind, ckpt, device, sims, batch, seed)
    target = proofs(rows["board"])
    kind = np.where(target != 0, data.KIND_PROOF, data.KIND_TEACHER).astype(np.uint8)
    open_rows = np.flatnonzero(kind == data.KIND_TEACHER)
    searched = values(
        values_of, device, rows["board"][open_rows], rows["since_capture"][open_rows], rows["ply"][open_rows], batch
    )
    target[open_rows] = np.clip(searched, -1, 1)
    labelled = dict(rows)
    labelled.update(
        target=target,
        weight=np.ones(n, dtype=np.float32),
        kind=kind,
        capture_clock=np.full(n, SITE_CLOCK, dtype=np.uint16),
    )
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    target_dir = data_dir(out)
    data.write(
        target_dir,
        labelled,
        {
            "producer": "teacher-gpu",
            "teacher": teacher_kind,
            "checkpoint": {"path": str(ckpt), "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()},
            "weights": "ema",
            "sims": sims,
            "search": {k: getattr(cfg.search, k) for k in ("sims", "candidates", "cheap_sims", "cheap_candidates")},
            "rules": {"capture_clock": SITE_CLOCK, "repetition_draw": False},
            "input": {"dataset": str(source), "provenance": provenance, "rows": n, "sample": sample, "seed": seed},
            "proofs": int((kind == data.KIND_PROOF).sum()),
            "seconds": time.perf_counter() - started,
        },
    )
    print(
        json.dumps(
            {"event": "written", "dataset": str(target_dir), "rows": n, "seconds": time.perf_counter() - started}
        )
    )
    return target_dir


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ckpt", type=Path, required=True, help="a conv training checkpoint")
    parser.add_argument("--teacher", choices=("conv", "sq"), default="conv")
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--rows", type=int, default=0, help="label a random sample of this many rows")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    label(args.input, args.out, args.ckpt, args.sims, args.batch, args.rows, args.seed, args.teacher)


if __name__ == "__main__":
    main(sys.argv[1:])
