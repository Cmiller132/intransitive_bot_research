"""Teacher labels (DESIGN item 22): every row of a dataset gets a fresh
search value from a model engine through `bot analyse`, or an exact proof
from the engine's tactics, under the site rules.

    python -m nnue.label --input <set> --out <set> --engine sq:weights/sq_g128.onnx [--sims 256] [--workers 8]
        [--rows N] [--seed S | --select <shallow set>]

Rows that win at once are worth 1 and rows whose every move loses in two are
worth -1 (kind 2); every other row is worth the teacher's root value after
`sims` simulations (kind 3), with the chosen line's Q kept in the provenance
statistics. The output keeps the input's boards, counters, games, orbits and
splits, and records the site clock, because the teacher searches under it.
`--rows N` labels a seeded random sample of N rows of the input (a large set
of positions whose teacher labels are wanted a slice at a time); with
`--select <set>`, a `nnue.label` pass over a sample of the input at a shallow
budget, it labels instead the N rows whose input label differs most from
their shallow label (the positions the shallow search misjudges, the
selection of step_plan_final.md item D; its control is the seeded sample).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import engine
import numpy as np

from . import data
from .games import parse_action
from .paths import bot_binary, data_dir, sha256, workspace_root

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
    """The responses for `rows`, in order, from one `bot analyse` process,
    matched by request id. The answers a dying process gave are kept; a row it
    did not answer gets an error response, and its one retry is the caller's."""
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
    answered: dict[int, dict] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            response = json.loads(line)
        except ValueError:
            break  # the torn last line of a dying process
        if isinstance(response, dict) and isinstance(response.get("id"), int) and 0 <= response["id"] < len(rows):
            answered[response["id"]] = response
    if len(answered) < len(rows):
        tail = result.stderr.strip()[-300:]
        reason = f"bot analyse exited with {result.returncode}: {tail}" if result.returncode else "no response"
        return [answered.get(i, {"error": reason}) for i in range(len(rows))]
    return [answered[i] for i in range(len(rows))]


def captures(board: np.ndarray, since: int, ply: int, token: str) -> bool:
    """Whether the chosen move (an absolute token from the blue frame the
    request used) captures: the counter resets on a capture."""
    action = parse_action(token, False)
    _, since_after, _, outcome = engine.apply(list(board), int(since), int(ply), action, SITE_CLOCK)
    return since_after == 0 and outcome == 0


def disagreement(source: Path, shallow: Path, count: int) -> tuple[np.ndarray, dict]:
    """Indices of the `count` rows of `source` whose label differs most from
    the same position's label in `shallow`; positions absent from `shallow`
    are not eligible. Positions match by board and clock state."""
    from .importer import context_hash

    def keys(directory: Path) -> np.ndarray:
        return context_hash(
            np.load(directory / "board.npy", mmap_mode="r"),
            np.load(directory / "since_capture.npy"),
            np.load(directory / "capture_clock.npy"),
        )

    source_key, shallow_key = keys(source), keys(shallow)
    for name, key in ((source, source_key), (shallow, shallow_key)):
        if len(np.unique(key)) != len(key):
            raise ValueError(f"{name} holds duplicate positions; selection needs one row per board and clock state")
    order = np.argsort(source_key, kind="stable")
    index = order[np.minimum(np.searchsorted(source_key, shallow_key, sorter=order), len(order) - 1)]
    if not np.array_equal(source_key[index], shallow_key):
        raise ValueError(f"{shallow} holds positions that are not in {source}; it must be a labelling of the source")
    gap = np.abs(np.load(source / "target.npy")[index] - np.load(shallow / "target.npy"))
    if not np.isfinite(gap).all():
        raise ValueError("non-finite label gap")
    if len(gap) < count:
        raise ValueError(f"{len(gap)} eligible rows in {shallow}, {count} requested")
    top = np.argsort(-gap, kind="stable")[:count]
    stats = {
        "pool": int(len(gap)),
        "selected": int(len(top)),
        "gap_min": float(gap[top].min()) if len(top) else None,
        "gap_mean": float(gap[top].mean()) if len(top) else None,
        "pool_gap_mean": float(gap.mean()) if len(gap) else None,
    }
    return np.sort(index[top]), stats


def network_hash(engine_spec: str) -> str | None:
    """The hash of the network an `nnue:<file>[?...]` spec names; None for other engines."""
    kind, _, rest = engine_spec.partition(":")
    if kind != "nnue":
        return None
    file = Path(rest.partition("?")[0])
    return sha256(file if file.is_absolute() else workspace_root() / file)


def label(
    input_set: str,
    out: str,
    engine_spec: str,
    sims: int,
    workers: int,
    sample: int = 0,
    seed: int = 0,
    quiet_best: bool = False,
    select: str = "",
) -> Path:
    """`quiet_best` drops the rows whose chosen move is a capture, Stockfish's
    generation filter: the evaluator then learns positions the search
    would evaluate statically, not ones it resolves with a capture."""
    started = time.perf_counter()
    source = Path(data_dir(input_set))
    rows = {name: np.load(source / f"{name}.npy") for name in data.FIELDS}
    n = len(rows["board"])
    selection = None
    if select:
        chosen, selection = disagreement(source, Path(data_dir(select)), sample or n)
        rows = {name: value[chosen] for name, value in rows.items()}
        n = len(chosen)
    elif sample and sample < n:
        chosen = np.sort(np.random.default_rng(seed).choice(n, sample, replace=False))
        rows = {name: value[chosen] for name, value in rows.items()}
        n = sample
    target = proofs(rows["board"])
    kind = np.where(target != 0, data.KIND_PROOF, data.KIND_TEACHER).astype(np.uint8)
    open_rows = np.flatnonzero(kind == data.KIND_TEACHER)

    attempts = {"processes": 0, "seconds": 0.0}  # every process run, failed ones included, and their wall time

    def searched(chunk: list[dict]) -> list[dict]:
        if not chunk:
            return []
        began = time.perf_counter()
        try:
            return analyse(engine_spec, chunk, sims)
        finally:
            attempts["processes"] += 1
            attempts["seconds"] += time.perf_counter() - began

    def search_rows(indices: np.ndarray) -> dict[int, dict]:
        """`bot analyse` over the rows `indices`, `workers` processes at a time."""
        requests = [
            {"board": rows["board"][i].tolist(), "since_capture": rows["since_capture"][i], "ply": rows["ply"][i]}
            for i in indices
        ]
        chunks = [requests[k::workers] for k in range(workers)]
        with ThreadPoolExecutor(workers) as pool:
            answers = list(pool.map(searched, chunks))
        return {int(indices[k + j * workers]): r for k, chunk in enumerate(answers) for j, r in enumerate(chunk)}

    def failed(response: dict) -> bool:
        return bool(response.get("error")) or not response.get("search")

    responses = search_rows(open_rows)
    # A failed root is searched once more under the same budget; no other root ever replaces it.
    retry = np.array([i for i in open_rows if failed(responses[i])], dtype=np.int64)
    if len(retry):
        responses.update(search_rows(retry))
    chosen_q, root_value, errors, dropped = [], [], 0, 0
    nodes, unknown = 0, 0  # searched nodes the answers report; attempts whose cost no answer reports
    for i in open_rows:
        response = responses[i]
        if failed(response):
            errors += 1
            kind[i] = 255
            unknown += 1
            continue
        search = response["search"]
        nodes += int(search.get("nodes") or 0)
        unknown += "nodes" not in search
        if (
            quiet_best
            and search["lines"]
            and captures(rows["board"][i], rows["since_capture"][i], rows["ply"][i], search["lines"][0]["move"])
        ):
            dropped += 1
            kind[i] = 255
            continue
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
    chosen = np.array(chosen_q, dtype=np.float64)
    root = np.array(root_value, dtype=np.float64)
    target_dir = data_dir(out)
    data.write(
        target_dir,
        labelled,
        {
            "producer": "teacher",
            "engine": {
                "spec": engine_spec,
                "binary": str(bot_binary()),
                "binary_sha256": sha256(bot_binary()),
                "network_sha256": network_hash(engine_spec),
            },
            "sims": sims,
            "rules": {"capture_clock": SITE_CLOCK, "repetition_draw": False},
            "input": {
                "dataset": input_set,
                "provenance_sha256": sha256(source / "provenance.json"),
                "rows": n,
                "sample": sample,
                "seed": seed,
            },
            "select": {
                "dataset": select,
                "provenance_sha256": sha256(data_dir(select) / "provenance.json"),
                **selection,
            }
            if selection
            else None,
            "proofs": int((kind == data.KIND_PROOF).sum()),
            "retries": int(len(retry)),
            "errors": errors,
            "cost": {
                "submitted": int(len(open_rows) + len(retry)),  # request slots, not proven searches
                "processes": attempts["processes"],
                "process_seconds": attempts["seconds"],
                "reported_nodes": nodes,
                "attempts_without_node_counts": unknown + int(len(retry)),
            },
            "quiet_best": quiet_best,
            "dropped": dropped,
            "chosen_q_vs_root": {
                "mean_abs_difference": float(np.nanmean(np.abs(chosen - root))) if len(root) else None,
                "rows": int(len(root)),
            },
            "seconds": time.perf_counter() - started,
        },
    )
    print(
        json.dumps(
            {
                "event": "written",
                "dataset": str(target_dir),
                "rows": int(keep.sum()),
                "errors": errors,
                "dropped": dropped,
            }
        )
    )
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
    parser.add_argument("--quiet-best", action="store_true", help="drop rows whose best move is a capture")
    parser.add_argument("--select", default="", help="label the rows disagreeing most with this shallow labelling")
    args = parser.parse_args(argv)
    label(
        args.input, args.out, args.engine, args.sims, args.workers, args.rows, args.seed, args.quiet_best, args.select
    )


if __name__ == "__main__":
    main(sys.argv[1:])
