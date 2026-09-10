"""Teacher labels (DESIGN item 22): every row of a dataset gets a fresh
search value from a model engine through `bot analyse`, or an exact proof
from the engine's tactics, under the site rules.

    python -m nnue.label --input <set> --out <set> --engine sq:weights/sq_g128.onnx [--sims 256] [--workers 8]
        [--rows N] [--seed S]

Rows that win at once are worth 1 and rows whose every move loses in two are
worth -1 (kind 2); every other row is worth the teacher's root value after
`sims` simulations (kind 3), with the chosen line's Q kept in the provenance
statistics. The output keeps the input's boards, counters, games, orbits and
splits, and records the site clock, because the teacher searches under it.
`--rows N` labels a seeded random sample of N rows of the input (a large set
of positions whose teacher labels are wanted a slice at a time).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import data
from .gauntlet import bot_binary
from .paths import data_dir, workspace_root

SITE_CLOCK = 200


def proofs(board: np.ndarray) -> np.ndarray:
    """+1 where the mover wins at once, -1 where every legal move loses in
    two, 0 otherwise (NaN-free: exact values only)."""
    import engine

    out = np.zeros(len(board), dtype=np.float32)
    for i, cells in enumerate(board.tolist()):
        legal = np.array(engine.legal_mask(cells))
        if not legal.any():
            out[i] = -1
            continue
        wins, loses, _ = engine.tactics(cells)
        if any(wins):
            out[i] = 1
        elif all(lost for lost, ok in zip(loses, legal, strict=True) if ok):
            out[i] = -1
    return out


def analyse(engine_spec: str, rows: list[dict], sims: int) -> list[dict]:
    """The responses for `rows`, in order. A process that dies (an analyser
    panic on one position) is retried on each half of its rows until the
    single offending row is isolated and answered with an error."""
    try:
        return analyse_once(engine_spec, rows, sims)
    except RuntimeError as failure:
        if len(rows) == 1:
            return [{"error": str(failure)}]
        half = len(rows) // 2
        return analyse(engine_spec, rows[:half], sims) + analyse(engine_spec, rows[half:], sims)


def analyse_once(engine_spec: str, rows: list[dict], sims: int) -> list[dict]:
    """One `bot analyse` process over `rows`; returns its responses in order."""
    command = [str(bot_binary()), "analyse", "--engine", engine_spec, "--threads", "1"]
    requests = "".join(
        json.dumps(
            {
                "id": i,
                "setup": row["board"],
                "to_move": "blue",
                "psc": int(row["since_capture"]),
                "ply": int(row["ply"]),
                "sims": sims,
            }
        )
        + "\n"
        for i, row in enumerate(rows)
    )
    result = subprocess.run(command, input=requests, capture_output=True, text=True, cwd=workspace_root(), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"bot analyse failed: {result.stderr.strip()}")
    responses = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(responses) != len(rows):
        raise RuntimeError(f"{len(responses)} responses for {len(rows)} requests")
    return responses


def label(input_set: str, out: str, engine_spec: str, sims: int, workers: int, sample: int = 0, seed: int = 0) -> Path:
    started = time.perf_counter()
    source = Path(data_dir(input_set))
    rows = {name: np.load(source / f"{name}.npy") for name in data.FIELDS}
    n = len(rows["board"])
    if sample and sample < n:
        chosen = np.sort(np.random.default_rng(seed).choice(n, sample, replace=False))
        rows = {name: value[chosen] for name, value in rows.items()}
        n = sample
    target = proofs(rows["board"])
    kind = np.where(target != 0, data.KIND_PROOF, data.KIND_TEACHER).astype(np.uint8)
    open_rows = np.flatnonzero(kind == data.KIND_TEACHER)
    requests = [
        {"board": rows["board"][i].tolist(), "since_capture": rows["since_capture"][i], "ply": rows["ply"][i]}
        for i in open_rows
    ]
    chunks = [requests[k::workers] for k in range(workers)]
    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(lambda chunk: analyse(engine_spec, chunk, sims) if chunk else [], chunks))
    chosen_q, root_value, errors = [], [], 0
    for k, chunk_answers in enumerate(answers):
        for j, response in enumerate(chunk_answers):
            i = open_rows[k + j * workers]
            if response.get("error") or not response.get("search"):
                errors += 1
                kind[i] = 255
                continue
            search = response["search"]
            target[i] = float(np.clip(search["root_value"], -1, 1))
            root_value.append(search["root_value"])
            chosen_q.append(search["lines"][0]["q"] if search["lines"] else float("nan"))
    if errors and not root_value:
        raise SystemExit(f"every search request failed ({errors} errors); is {engine_spec!r} an analysable engine?")
    keep = kind != 255
    labelled = {name: rows[name][keep] for name in data.FIELDS}
    labelled.update(
        target=target[keep],
        weight=np.ones(int(keep.sum()), dtype=np.float32),
        kind=kind[keep],
        capture_clock=np.full(int(keep.sum()), SITE_CLOCK, dtype=np.uint16),
    )
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    chosen = np.array(chosen_q, dtype=np.float64)
    root = np.array(root_value, dtype=np.float64)
    target_dir = data_dir(out)
    data.write(
        target_dir,
        labelled,
        {
            "producer": "teacher",
            "engine": engine_spec,
            "sims": sims,
            "rules": {"capture_clock": SITE_CLOCK, "repetition_draw": False},
            "input": {"dataset": str(source), "provenance": provenance, "rows": n, "sample": sample, "seed": seed},
            "proofs": int((kind == data.KIND_PROOF).sum()),
            "errors": errors,
            "chosen_q_vs_root": {
                "mean_abs_difference": float(np.nanmean(np.abs(chosen - root))) if len(root) else None,
                "rows": int(len(root)),
            },
            "seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps({"event": "written", "dataset": str(target_dir), "rows": int(keep.sum()), "errors": errors}))
    return target_dir


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="dataset to relabel")
    parser.add_argument("--out", required=True, help="dataset to write")
    parser.add_argument("--engine", default="sq:weights/sq_g128.onnx", help="analyser spec for `bot analyse`")
    parser.add_argument("--sims", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rows", type=int, default=0, help="label a random sample of this many rows")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    label(args.input, args.out, args.engine, args.sims, args.workers, args.rows, args.seed)


if __name__ == "__main__":
    main(sys.argv[1:])
