"""Teacher labels in bulk on the GPU (DESIGN item 22 when the GPU is free):
conv's batched Gumbel search, run from a training checkpoint, values
thousands of positions per call.

    python -m nnue.gpu_label --input <set> --out <set> [--input <set> --out <set> ...] --ckpt <checkpoint.pt>
        [--teacher conv|sq] [--sims 128] [--batch 4096] [--rows N] [--seed S] [--reuse-nodes K]

`--teacher` names the model package whose checkpoint and batched search
label the rows (conv by default; sq with `--ckpt weights/sq_g128.pt`); the
teacher is chosen by arena rating at the time. Several `--input`/`--out`
pairs share one loaded teacher (the checkpoint load, the compile and the
graph capture cost about a minute per process), and the exact proofs of the
next set are computed on the CPU while the GPU searches the current one; a
pair whose output exists is skipped (datasets are immutable).

Measured on the RTX 4070 Ti with conv_g128@150 at 128 simulations: the GPU
is saturated (SM and memory controller at 100 %), about 250 rows/s at batch
4,096; a `progress` line every 16 batches reports the running rate.

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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import data
from .games import SITE_CLOCK
from .label import proofs
from .paths import data_dir, workspace_root

SEARCH_FIELDS = ("sims", "candidates", "cheap_sims", "cheap_candidates", "reuse_nodes")


def load_teacher(kind: str, ckpt: Path, device: torch.device):
    """`(kind, actor, config, kernels)`: the network of a conv or sq training
    checkpoint (EMA weights) as the package's own compiled actor."""
    sys.path.insert(0, str(workspace_root() / "models" / kind / "py"))
    if kind == "conv":
        from conv import kernels
        from conv.model import build, config_of, load_checkpoint
        from conv.train import compile_net, student_evaluator

        checkpoint = load_checkpoint(str(ckpt), device)
        cfg = config_of(checkpoint)
        net = build(cfg, checkpoint, device, "ema").eval()
        actor = student_evaluator(compile_net(net, cfg), cfg.play.draw_kernel_width)
    elif kind == "sq":
        from sq import kernels
        from sq.model import build, config_of, load_checkpoint
        from sq.train import acting, compile_net

        checkpoint = load_checkpoint(str(ckpt), device)
        cfg = config_of(checkpoint)
        net = build(cfg, checkpoint, device, "ema").eval()
        actor = acting(compile_net(net, cfg))
    else:
        raise ValueError(f"unknown teacher {kind!r}; conv or sq")
    return kind, actor, cfg, kernels


def searcher(loaded, device: torch.device, sims: int, batch: int, seed: int, reuse_nodes: int | None = None):
    """`(values_of, search config)`: a function from canonical boards, counters
    and plies (tensors on `device`, `batch` rows) to root values through the
    package's batched search; `reuse_nodes` overrides the tree capacity kept
    between calls (unused here, every call starts afresh)."""
    kind, actor, cfg, kernels = loaded
    search_cfg = replace(cfg.search, sims=sims, cheap_sims=min(cfg.search.cheap_sims, sims))
    if reuse_nodes is not None:
        search_cfg = replace(search_cfg, reuse_nodes=reuse_nodes)
    if kind == "conv":
        from conv.search import GumbelSearch

        clock = torch.full((batch,), SITE_CLOCK, dtype=torch.int32, device=device)
        search = GumbelSearch(
            search_cfg, batch, device, clock, cfg.rules.clock_penalty, cfg.rules.max_plies, seed, None
        )

        def values_of(b, s, p):
            _, legal, _ = kernels.derive_batch(b, s, p, clock)
            search.forget()
            return search(b, s, p, legal, actor, True).value

    else:
        from sq.search import GumbelSearch

        clock = torch.tensor(SITE_CLOCK, dtype=torch.int32, device=device)
        search = GumbelSearch(
            search_cfg, batch, device, clock, cfg.rules.clock_penalty, cfg.rules.max_plies, seed, None
        )

        def values_of(b, s, p):
            planes, legal, _ = kernels.derive_batch(b, s, p)
            search.forget()
            return search(b, s, p, legal, planes, actor, True)[1]

    return values_of, search_cfg


def teacher(
    kind: str, ckpt: Path, device: torch.device, sims: int, batch: int, seed: int, reuse_nodes: int | None = None
):
    """`(values_of, config)` for one checkpoint: `load_teacher` then `searcher`;
    the config carries the search settings in use."""
    loaded = load_teacher(kind, ckpt, device)
    values_of, search_cfg = searcher(loaded, device, sims, batch, seed, reuse_nodes)
    return values_of, replace(loaded[2], search=search_cfg)


@torch.no_grad()
def values(
    values_of,
    device: torch.device,
    board: np.ndarray,
    since: np.ndarray,
    ply: np.ndarray,
    batch: int,
    name: str = "",
    report: int = 16,
) -> np.ndarray:
    """Root values of every row, searched `batch` rows at a time (the last
    batch padded with the first row); a progress line every `report` batches."""
    n = len(board)
    out = np.zeros(n, dtype=np.float32)
    started = time.perf_counter()
    for k, start in enumerate(range(0, n, batch)):
        idx = np.arange(start, min(start + batch, n))
        pad = np.concatenate([idx, np.zeros(batch - len(idx), dtype=np.int64)])
        b = torch.from_numpy(board[pad].astype(np.int8)).to(device)
        s = torch.from_numpy(since[pad].astype(np.int32)).to(device)
        p = torch.from_numpy(ply[pad].astype(np.int32)).to(device)
        out[idx] = values_of(b, s, p)[: len(idx)].float().cpu().numpy()
        done = int(idx[-1]) + 1
        if report and (k + 1) % report == 0 and done < n:
            rate = done / (time.perf_counter() - started)
            print(
                json.dumps({"event": "progress", "set": name, "rows": done, "of": n, "rows_per_s": round(rate, 1)}),
                flush=True,
            )
    return out


def prepare(input_set: str, sample: int, seed: int) -> dict:
    """One set's rows (a random sample when asked), their exact proofs and
    the rows left for the search; CPU only, so it runs ahead of the GPU."""
    source = Path(data_dir(input_set))
    rows = {name: np.load(source / f"{name}.npy") for name in data.FIELDS}
    n = len(rows["board"])
    if sample and sample < n:
        chosen = np.sort(np.random.default_rng(seed).choice(n, sample, replace=False))
        rows = {name: value[chosen] for name, value in rows.items()}
        n = sample
    target = proofs(rows["board"])
    kind = np.where(target != 0, data.KIND_PROOF, data.KIND_TEACHER).astype(np.uint8)
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    return {
        "source": source,
        "rows": rows,
        "n": n,
        "target": target,
        "kind": kind,
        "open": np.flatnonzero(kind == data.KIND_TEACHER),
        "provenance": provenance,
    }


def label_sets(
    pairs: list[tuple[str, str]],
    ckpt: Path,
    sims: int,
    batch: int,
    sample: int,
    seed: int,
    teacher_kind: str = "conv",
    reuse_nodes: int | None = None,
) -> list[Path]:
    """Label every `(input, out)` pair with one loaded teacher, in order; a
    pair whose output exists is skipped. Returns the datasets written."""
    pending = []
    for input_set, out in pairs:
        if data_dir(out).exists():
            print(json.dumps({"event": "skipped", "dataset": str(data_dir(out)), "reason": "exists"}), flush=True)
        else:
            pending.append((input_set, out))
    if not pending:
        return []
    device = torch.device("cuda")
    loading = time.perf_counter()
    values_of, cfg = teacher(teacher_kind, ckpt, device, sims, batch, seed, reuse_nodes)
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "event": "teacher",
                "kind": teacher_kind,
                "checkpoint": str(ckpt),
                "seconds": round(time.perf_counter() - loading, 1),
            }
        ),
        flush=True,
    )
    written = []
    with ThreadPoolExecutor(1) as pool:
        ahead = pool.submit(prepare, pending[0][0], sample, seed)
        for k, (_input_set, out) in enumerate(pending):
            prepared = ahead.result()
            if k + 1 < len(pending):
                ahead = pool.submit(prepare, pending[k + 1][0], sample, seed)
            started = time.perf_counter()
            rows, n, target, kind = prepared["rows"], prepared["n"], prepared["target"], prepared["kind"]
            open_rows = prepared["open"]
            searched = values(
                values_of,
                device,
                rows["board"][open_rows],
                rows["since_capture"][open_rows],
                rows["ply"][open_rows],
                batch,
                out,
            )
            target[open_rows] = np.clip(searched, -1, 1)
            labelled = dict(rows)
            labelled.update(
                target=target,
                weight=np.ones(n, dtype=np.float32),
                kind=kind,
                capture_clock=np.full(n, SITE_CLOCK, dtype=np.uint16),
            )
            target_dir = data_dir(out)
            data.write(
                target_dir,
                labelled,
                {
                    "producer": "teacher-gpu",
                    "teacher": teacher_kind,
                    "checkpoint": {"path": str(ckpt), "sha256": sha},
                    "weights": "ema",
                    "sims": sims,
                    "search": {
                        field: getattr(cfg.search, field) for field in SEARCH_FIELDS if hasattr(cfg.search, field)
                    },
                    "rules": {"capture_clock": SITE_CLOCK, "repetition_draw": False},
                    "input": {
                        "dataset": str(prepared["source"]),
                        "provenance": prepared["provenance"],
                        "rows": n,
                        "sample": sample,
                        "seed": seed,
                    },
                    "proofs": int((kind == data.KIND_PROOF).sum()),
                    "seconds": time.perf_counter() - started,
                },
            )
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "event": "written",
                        "dataset": str(target_dir),
                        "rows": n,
                        "seconds": elapsed,
                        "rows_per_s": round(len(open_rows) / elapsed, 1),
                    }
                ),
                flush=True,
            )
            written.append(target_dir)
    return written


def label(
    input_set: str, out: str, ckpt: Path, sims: int, batch: int, sample: int, seed: int, teacher_kind: str = "conv"
) -> Path:
    """One set (the original interface)."""
    written = label_sets([(input_set, out)], ckpt, sims, batch, sample, seed, teacher_kind)
    return written[0] if written else data_dir(out)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", required=True, help="input set; repeat with --out for several sets")
    parser.add_argument("--out", action="append", required=True, help="output set, one per --input")
    parser.add_argument("--ckpt", type=Path, required=True, help="a conv (or sq) training checkpoint")
    parser.add_argument("--teacher", choices=("conv", "sq"), default="conv")
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--rows", type=int, default=0, help="label a random sample of this many rows per set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reuse-nodes", type=int, default=None, help="tree capacity kept between calls (config default)"
    )
    args = parser.parse_args(argv)
    if len(args.input) != len(args.out):
        parser.error("--input and --out must be given the same number of times")
    label_sets(
        list(zip(args.input, args.out, strict=True)),
        args.ckpt,
        args.sims,
        args.batch,
        args.rows,
        args.seed,
        args.teacher,
        args.reuse_nodes,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
