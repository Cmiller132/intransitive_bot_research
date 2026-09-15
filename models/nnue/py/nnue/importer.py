"""Datasets from other producers (DESIGN items 19 and 20): the replay windows
of a self-play run (`conv` schema 2 today), the site's human games and the
searched roots of `bot selfplay`.

    python -m nnue.importer conv --run runs/conv_g128 --out <set> [--iterations 13-25] [--children 16]
    python -m nnue.importer human --file <games_export.txt> --out <set>

A conv window holds, per row, the position, the played action's lambda return
and up to 16 visited root candidates with their completed Q. The importer
labels the played root with its return (kind 0, weight 0.25) and every visited
candidate's nonterminal child, applied through the engine, with minus that Q
(kind 1, weight 1); outcomes stay on played rows only. Episodes are recovered
from the ply sequence of every environment across contiguous windows, exact
(board, since_capture, clock) contexts are merged with the weight capped at 4,
and splits go by episode first and never let a symmetry orbit cross splits.

A human export (the site's compact lines `<b|r|d|u> <blue> <red> <moves>`)
gives every played root of a finished game its outcome as the label (kind 0,
weight 0.25, outcome recorded), merged over contexts like the windows; it is
the broad coverage of human play that a teacher can relabel later.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import data
from .features import orbit_hash
from .games import SITE_CLOCK, decode_export, replay
from .paths import data_dir

CONV_SCHEMA = 2
RETURN_WEIGHT = 0.25
MAX_WEIGHT = 4.0


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def window_files(run: Path, iterations: tuple[int, int] | None) -> list[tuple[int, Path]]:
    """`(iteration, path)` of the run's windows, the compressed form when both exist."""
    found: dict[int, Path] = {}
    for path in sorted(run.glob("window_*.pt*")):
        stem = path.name[len("window_") :].split(".")[0]
        if not stem.isdigit():
            continue
        it = int(stem)
        if iterations and not iterations[0] <= it <= iterations[1]:
            continue
        if it not in found or path.suffix == ".zst":
            found[it] = path
    return sorted(found.items())


def load_window(path: Path) -> tuple[dict[str, np.ndarray], str]:
    """The window's arrays and the sha256 of the file's bytes."""
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if path.suffix == ".zst":
        from compression import zstd

        raw = zstd.decompress(raw)
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if payload.get("schema") != CONV_SCHEMA:
        raise ValueError(f"{path}: window schema {payload.get('schema')}, expected {CONV_SCHEMA}")
    arrays = {
        k: (v.float() if v.dtype == torch.bfloat16 else v).numpy()
        for k, v in payload.items()
        if isinstance(v, torch.Tensor)
    }
    arrays["envs"], arrays["steps"], arrays["iteration"] = payload["envs"], payload["steps"], payload["iteration"]
    n = arrays["steps"] * arrays["envs"]
    # Windows written before the outcome labels exist carry none.
    arrays.setdefault("outcome", np.zeros(n, dtype=np.int8))
    arrays.setdefault("outcome_ok", np.zeros(n, dtype=bool))
    for name in ("board", "since_capture", "ply", "clock", "action", "ret", "candidates", "candidate_q"):
        if name not in arrays or arrays[name].shape[0] != n:
            raise ValueError(f"{path}: field {name} missing or not {n} rows")
    return arrays, digest


class Episodes:
    """Episode ids per environment, carried across contiguous windows: a new
    episode starts wherever the ply does not follow the previous step's ply."""

    def __init__(self, envs: int):
        self.envs = envs
        self.current = np.arange(envs, dtype=np.int64)
        self.next_id = envs
        self.last_ply: np.ndarray | None = None
        self.last_iteration = -2

    def label(self, ply: np.ndarray, iteration: int) -> np.ndarray:
        """(steps, envs) episode ids for a window's ply array of that shape."""
        steps = ply.shape[0]
        out = np.empty_like(ply, dtype=np.int64)
        if self.last_ply is None:
            previous = ply[0] - 1  # the first window keeps its initial ids
        elif iteration == self.last_iteration + 1:
            previous = self.last_ply
        else:
            previous = np.full(self.envs, -2)  # a gap: every environment starts a new episode
        for t in range(steps):
            fresh = ply[t] != previous + 1
            count = int(fresh.sum())
            self.current[fresh] = np.arange(self.next_id, self.next_id + count)
            self.next_id += count
            out[t] = self.current
            previous = ply[t]
        self.last_ply = ply[-1].copy()
        self.last_iteration = iteration
        return out


def children_of(window: dict[str, np.ndarray], cap: int) -> dict[str, np.ndarray]:
    """Nonterminal children of the visited candidates through the engine."""
    import engine

    board, since, ply, clock = window["board"], window["since_capture"], window["ply"], window["clock"]
    visited = window["candidate_visited"].copy()
    if cap < visited.shape[1]:
        # Keep the most valuable candidates by Q so a cap does not drop the best line.
        order = np.argsort(-np.where(visited, window["candidate_q"], -np.inf), axis=1)
        keep = np.zeros_like(visited)
        np.put_along_axis(keep, order[:, :cap], True, axis=1)
        visited &= keep
    rows, slots = np.nonzero(visited)
    boards = np.empty((len(rows), 81), dtype=np.uint8)
    child_since = np.empty(len(rows), dtype=np.int64)
    child_ply = np.empty(len(rows), dtype=np.int64)
    ongoing = np.zeros(len(rows), dtype=bool)
    actions = window["candidates"][rows, slots].astype(int)
    q = window["candidate_q"][rows, slots].astype(np.float32)
    for k, (i, action) in enumerate(zip(rows.tolist(), actions.tolist(), strict=True)):
        child, s, p, outcome = engine.apply(board[i].tolist(), int(since[i]), int(ply[i]), action, int(clock[i]))
        if outcome == 0:
            boards[k] = np.frombuffer(child, dtype=np.uint8)
            child_since[k], child_ply[k] = s, p
            ongoing[k] = True
    return {
        "row": rows[ongoing],
        "board": boards[ongoing],
        "since_capture": child_since[ongoing],
        "ply": child_ply[ongoing],
        "target": np.clip(-q[ongoing], -1, 1),
    }


def context_hash(board: np.ndarray, since_capture: np.ndarray, clock: np.ndarray, seed: int = 7) -> np.ndarray:
    """A 64-bit key of the exact (board, since_capture, clock) context."""
    rng = np.random.default_rng(seed)
    table = rng.integers(0, np.iinfo(np.uint64).max, size=(81, 7), dtype=np.uint64)
    key = np.bitwise_xor.reduce(table[np.arange(81), np.asarray(board, dtype=np.uint8)], axis=1)
    key ^= np.asarray(since_capture, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    key ^= np.asarray(clock, dtype=np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
    return key


def merge_contexts(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One row per exact (board, since_capture, clock): weighted mean target,
    weight capped, the lowest kind, the outcome of any row that knows one."""
    key = context_hash(rows["board"], rows["since_capture"], rows["capture_clock"])
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    count = len(first)
    weight = rows["weight"].astype(np.float64)
    mass = np.bincount(inverse, weights=weight, minlength=count)
    target = np.bincount(inverse, weights=rows["target"] * weight, minlength=count) / mass
    outcome_ok = np.zeros(count, dtype=bool)
    outcome = np.zeros(count, dtype=np.int8)
    known = np.flatnonzero(rows["outcome_ok"])[::-1]
    outcome_ok[inverse[known]] = True
    outcome[inverse[known]] = rows["outcome"][known]
    kind = np.full(count, 255, dtype=np.uint8)
    np.minimum.at(kind, inverse, rows["kind"])  # a played root outranks its other labels
    merged = {name: rows[name][first] for name in ("board", "since_capture", "ply", "capture_clock", "source", "game")}
    merged.update(
        target=target.astype(np.float32),
        weight=np.minimum(mass, MAX_WEIGHT).astype(np.float32),
        kind=kind,
        outcome=outcome,
        outcome_ok=outcome_ok,
    )
    return merged


def assign_splits(game: np.ndarray, orbit: np.ndarray, seed: int) -> np.ndarray:
    """90/5/5 by episode, then every orbit seen in more than one split goes to train."""
    rng = np.random.default_rng(seed)
    unique_games, inverse = np.unique(game, return_inverse=True)
    bucket = rng.integers(0, 100, len(unique_games))
    split = np.where(bucket < 90, data.TRAIN, np.where(bucket < 95, data.VALIDATION, data.TEST)).astype(np.uint8)
    split = split[inverse.ravel()]
    for held in (data.VALIDATION, data.TEST):
        leaked = np.isin(orbit, orbit[split != held]) & (split == held)
        split[leaked] = data.TRAIN
    return split


def import_conv(run: Path, out: str, iterations: tuple[int, int] | None, cap: int, seed: int) -> Path:
    files = window_files(run, iterations)
    if not files:
        raise FileNotFoundError(f"no windows in {run}")
    parts: list[dict[str, np.ndarray]] = []
    episodes: Episodes | None = None
    manifest = []
    for it, path in files:
        window, digest = load_window(path)
        manifest.append({"iteration": it, "file": path.name, "sha256": digest})
        envs, steps = window["envs"], window["steps"]
        episodes = episodes or Episodes(envs)
        game = episodes.label(window["ply"].reshape(steps, envs).astype(np.int64), it).reshape(-1)
        n = steps * envs
        root = {
            "board": window["board"].astype(np.uint8),
            "since_capture": window["since_capture"].astype(np.int64),
            "ply": window["ply"].astype(np.int64),
            "capture_clock": window["clock"].astype(np.int64),
            "target": np.clip(window["ret"], -1, 1).astype(np.float32),
            "weight": np.full(n, RETURN_WEIGHT, dtype=np.float32),
            "kind": np.full(n, data.KIND_RETURN, dtype=np.uint8),
            "outcome": window["outcome"].astype(np.int8),
            "outcome_ok": window["outcome_ok"].astype(bool),
            "source": np.full(n, data.SOURCE_CONV, dtype=np.uint8),
            "game": game,
        }
        child = children_of(window, cap)
        m = len(child["row"])
        parts.append(root)
        parts.append(
            {
                "board": child["board"],
                "since_capture": child["since_capture"],
                "ply": child["ply"],
                "capture_clock": root["capture_clock"][child["row"]],
                "target": child["target"],
                "weight": np.ones(m, dtype=np.float32),
                "kind": np.full(m, data.KIND_CHILD, dtype=np.uint8),
                "outcome": np.zeros(m, dtype=np.int8),
                "outcome_ok": np.zeros(m, dtype=bool),
                "source": np.full(m, data.SOURCE_CONV, dtype=np.uint8),
                "game": game[child["row"]],
            }
        )
        print(json.dumps({"event": "window", "iteration": it, "roots": n, "children": m}), flush=True)
    rows = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    del parts
    raw_rows = len(rows["board"])
    rows = merge_contexts(rows)
    rows["orbit"] = orbit_hash(rows["board"])
    rows["split"] = assign_splits(rows["game"], rows["orbit"], seed)
    config_path = run / "config.json"
    provenance = {
        "producer": "conv",
        "run": str(run.resolve()),
        "windows": manifest,
        "config": json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None,
        "sources": {data.SOURCE_CONV: "conv self-play windows"},
        "labels": {
            "kind_0": "played root: lambda return (search-value bootstrap) at weight 0.25",
            "kind_1": "visited candidate's nonterminal child: minus its completed Q at weight 1",
            "outcomes": "played rows only, from the window's back-filled outcome",
            "domain": "source rules: the run's per-game clocks and its search shaping",
        },
        "children_per_root": cap,
        "raw_rows": raw_rows,
        "seed": seed,
    }
    target = data_dir(out)
    data.write(target, rows, provenance)
    counts = {int(s): int((rows["split"] == s).sum()) for s in range(3)}
    print(json.dumps({"event": "written", "dataset": str(target), "rows": len(rows["board"]), "splits": counts}))
    return target


def import_human(file: Path, out: str, seed: int) -> Path:
    """Every played root of the finished games in the site's export."""
    raw = file.read_bytes()
    results = {"b": 1, "r": -1, "d": 0}
    counts = {"games": 0, "unfinished": 0, "without_moves": 0, "finished": 0}
    boards, since, ply, outcome, game = [], [], [], [], []
    for line in raw.decode("ascii").splitlines():
        parts = line.split()
        if not parts:
            continue
        counts["games"] += 1
        if len(parts) != 4:
            counts["without_moves"] += 1
            continue
        if parts[0] not in results:
            counts["unfinished"] += 1
            continue
        roots, _ = replay(decode_export(parts[3]))
        blue = results[parts[0]]
        for board, s, p, _ in roots:
            boards.append(board)
            since.append(s)
            ply.append(p)
            outcome.append(blue if p % 2 == 0 else -blue)
            game.append(counts["finished"])
        counts["finished"] += 1
    n = len(boards)
    rows = {
        "board": np.stack(boards).astype(np.uint8),
        "since_capture": np.array(since, dtype=np.int64),
        "ply": np.array(ply, dtype=np.int64),
        "capture_clock": np.full(n, SITE_CLOCK),
        "target": np.array(outcome, dtype=np.float32),
        "weight": np.full(n, RETURN_WEIGHT, dtype=np.float32),
        "kind": np.full(n, data.KIND_RETURN, dtype=np.uint8),
        "outcome": np.array(outcome, dtype=np.int8),
        "outcome_ok": np.ones(n, dtype=bool),
        "source": np.full(n, data.SOURCE_HUMAN, dtype=np.uint8),
        "game": np.array(game, dtype=np.int64),
    }
    rows = merge_contexts(rows)
    rows["orbit"] = orbit_hash(rows["board"])
    rows["split"] = assign_splits(rows["game"], rows["orbit"], seed)
    target = data_dir(out)
    data.write(
        target,
        rows,
        {
            "producer": "human",
            "sources": {data.SOURCE_HUMAN: "site export of human games"},
            "file": {"name": file.name, "sha256": sha256_bytes(raw), "bytes": len(raw)},
            "counts": counts,
            "positions": n,
            "rules": {"capture_clock": SITE_CLOCK, "frame": "Blue's home a1, Blue first"},
            "label": "game outcome from the mover's view, weight 0.25",
            "seed": seed,
        },
    )
    print(json.dumps({"event": "written", "dataset": str(target), "rows": len(rows["board"]), "counts": counts}))
    return target


SELFPLAY_KINDS = {"search": data.KIND_TEACHER, "engine_proof": data.KIND_PROOF}


def import_selfplay(records: list[Path], out: str, min_ply: int, seed: int) -> Path:
    """The searched roots of `bot selfplay` games (the published shards listed
    in each directory's manifest) as labelled rows: target = tanh(root_score /
    score_scale) from the mover's view, the score of the last completed
    iteration (kind TEACHER) or an engine proof (kind PROOF); the outcome from
    the mover's view where the game's result is real (not censored) and the
    root lies after the last random deviation; rows from ply >= min_ply;
    duplicate (board, clock state) rows dropped, the first kept. Game ids are
    unique across directories (directory index in the high byte)."""
    boards: list[np.ndarray] = []
    since_all: list[int] = []
    plies: list[int] = []
    games: list[int] = []
    outcomes: list[int] = []
    outcome_ok: list[bool] = []
    targets: list[float] = []
    kinds: list[int] = []
    clocks: list[int] = []
    counts = {"games": 0, "censored": 0, "roots": 0, "labelled": 0, "eligible": 0, "mismatched": 0}
    ends: dict[str, int] = {}
    manifests = []
    for index, directory in enumerate(records):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        scale = float(manifest.get("score_scale", 600))
        manifests.append(
            {
                "directory": str(directory),
                "settings": manifest.get("settings"),
                "model_sha256": manifest.get("model_sha256"),
                "binary_sha256": manifest.get("binary_sha256"),
                "rules_sha256": manifest.get("rules_sha256"),
                "score_scale": scale,
                "shards": len(manifest.get("shards", [])),
            }
        )
        for shard in manifest.get("shards", []):
            with gzip.open(directory / shard["file"], "rt", encoding="utf-8") as lines:
                for line in lines:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if "moves" not in record or "roots" not in record:
                        continue
                    counts["games"] += 1
                    end = record.get("end") or "none"
                    ends[end] = ends.get(end, 0) + 1
                    censored = bool(record.get("censored")) or record.get("outcome") is None
                    counts["censored"] += censored
                    clock = int(record.get("capture_clock", SITE_CLOCK))
                    roots, _ = replay(record["moves"], clock)
                    game = (index << 24) | int(record["game_id"])
                    # None for a game the generator recorded as interrupted (censored: no outcome, 2026-09-15)
                    after = int(record.get("outcome_after_ply") or 0)
                    for root in record["roots"]:
                        counts["roots"] += 1
                        kind = SELFPLAY_KINDS.get(root.get("score_kind"))
                        if kind is None or root.get("root_score") is None:
                            continue
                        counts["labelled"] += 1
                        ply = int(root["ply"])
                        if ply < min_ply:
                            continue
                        if ply >= len(roots) or roots[ply][1] != int(root["since_capture"]) or roots[ply][2] != ply:
                            counts["mismatched"] += 1
                            continue
                        counts["eligible"] += 1
                        board, since, _, _ = roots[ply]
                        mover = int(root["mover"])
                        boards.append(board)
                        since_all.append(since)
                        plies.append(ply)
                        games.append(game)
                        clocks.append(clock)
                        targets.append(float(np.tanh(float(root["root_score"]) / scale)))
                        kinds.append(kind)
                        if censored:
                            outcomes.append(0)
                            outcome_ok.append(False)
                        else:
                            result = int(record["outcome"])
                            outcomes.append(result if mover == 0 else -result)
                            outcome_ok.append(ply >= after)
    if not boards:
        raise SystemExit("no labelled roots at or after the minimum ply")
    rows = {
        "board": np.stack(boards).astype(np.uint8),
        "since_capture": np.array(since_all, dtype=np.uint16),
        "ply": np.array(plies, dtype=np.uint16),
        "capture_clock": np.array(clocks, dtype=np.uint16),
        "target": np.clip(np.array(targets, dtype=np.float32), -1, 1),
        "kind": np.array(kinds, dtype=np.uint8),
        "game": np.array(games, dtype=np.uint32),
        "outcome": np.array(outcomes, dtype=np.int8),
        "outcome_ok": np.array(outcome_ok, dtype=np.bool_),
    }
    _, first = np.unique(context_hash(rows["board"], rows["since_capture"], rows["capture_clock"]), return_index=True)
    first.sort()
    rows = {k: v[first] for k, v in rows.items()}
    n = len(first)
    rows.update(
        weight=np.ones(n, dtype=np.float32),
        source=np.full(n, data.SOURCE_SELFPLAY, dtype=np.uint8),
        orbit=orbit_hash(rows["board"]),
    )
    rows["split"] = assign_splits(rows["game"], rows["orbit"], seed)
    counts["unique"] = n
    target = data_dir(out)
    data.write(
        target,
        rows,
        {
            "producer": "selfplay",
            "sources": {data.SOURCE_SELFPLAY: "searched roots of bot selfplay games"},
            "records": manifests,
            "label": "tanh(root_score / score_scale) of the last completed iteration, mover's view; proofs as PROOF",
            "outcome": "mover's view; outcome_ok only for real results at or after the last random deviation",
            "min_ply": min_ply,
            "rules": {"capture_clock": int(rows["capture_clock"][0]), "repetition_draw": False},
            "counts": counts,
            "ends": ends,
            "seed": seed,
        },
    )
    print(json.dumps({"event": "written", "dataset": str(target), "rows": n, "counts": counts, "ends": ends}))
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    conv = sub.add_parser("conv", help="import the windows of a conv run")
    conv.add_argument("--run", type=Path, required=True)
    conv.add_argument("--out", required=True, help="dataset name under runs/nnue_data")
    conv.add_argument("--iterations", help="first-last, inclusive")
    conv.add_argument("--children", type=int, default=16, help="candidates per root at most, best Q first")
    conv.add_argument("--seed", type=int, default=20260909)
    human = sub.add_parser("human", help="import the site's export of human games")
    human.add_argument("--file", type=Path, required=True)
    human.add_argument("--out", required=True, help="dataset name under runs/nnue_data")
    human.add_argument("--seed", type=int, default=20260909)
    selfplay = sub.add_parser("selfplay", help="import the searched roots of bot selfplay records")
    selfplay.add_argument("--records", type=Path, action="append", required=True, help="records directory; repeatable")
    selfplay.add_argument("--out", required=True, help="dataset name under runs/nnue_data")
    selfplay.add_argument("--min-ply", type=int, default=16, help="first ply whose roots become training rows")
    selfplay.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args(argv)
    if args.command == "conv":
        span = tuple(int(x) for x in args.iterations.split("-")) if args.iterations else None
        import_conv(args.run, args.out, span, args.children, args.seed)
    elif args.command == "human":
        import_human(args.file, args.out, args.seed)
    else:
        import_selfplay(args.records, args.out, args.min_ply, args.seed)


if __name__ == "__main__":
    main(sys.argv[1:])
