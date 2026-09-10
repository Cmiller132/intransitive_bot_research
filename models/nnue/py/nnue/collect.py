"""Student collection (DESIGN item 21): positions from the student's own
games, for the teacher to label.

    python -m nnue.collect --round <name> --student <a.nnue> [--previous <b.nnue> | --opponent <spec> --move-ms N]
        [--families 64] [--states 10000]

Each opening family is one seeded random opening; the student plays it as
the first and as the second player against the previous network (or itself),
at node budgets that rotate through 2.5k, 7.5k and 20k nodes (`Clock::Sims`,
n x 2,500 nodes), or against any `bot` player spec (`--opponent
sq:weights/sq_g128.onnx`, the teacher itself) under a wall clock
(`--move-ms`), where the positions are those a stronger opponent steers the
student into. The first player's evaluated leaves are traced on the
cheapest games (with a wall clock, on every fourth family). The round samples `states` positions: 40 % played roots,
30 % evaluated leaves, 20 % legal alternatives at roots where the mover's
value dropped over its next move, 10 % targeted (sparse, long clock, or
tactical). Games go to `runs/nnue_collect/<round>/games.jsonl`; the dataset
`runs/nnue_data/<round>` holds raw rows (kind 4, target 0) with the game
outcome on played roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data, features
from .games import SITE_CLOCK, replay
from .gauntlet import bot_binary
from .importer import assign_splits, context_hash
from .paths import data_dir, run_dir, workspace_root

BUDGETS = (1, 3, 8)  # Clock::Sims(n): 2.5k, 7.5k and 20k nodes
SHARES = {"root": 0.4, "leaf": 0.3, "alternative": 0.2, "targeted": 0.1}
VALUE_DROP = 0.25
SPARSE_PIECES = 8
LONG_CLOCK = 24
LEAF_RESERVOIR = 4


@dataclass(frozen=True)
class Game:
    family: int
    first: str  # a `bot` player spec
    second: str
    budget: int  # Clock::Sims budget, or 0 under a wall clock
    move_ms: int  # wall clock per move, or 0 under a node budget
    leaves: bool


def spec_of(network: Path | str) -> str:
    """A `bot` player spec: a bare path means an NNUE file."""
    text = str(network)
    return text if text.startswith(("sq:", "conv:", "nnue:", "rpsi:")) else f"nnue:{text}"


def schedule(student: Path, previous: Path | str | None, families: int, move_ms: int = 0) -> list[Game]:
    """Two games per family; the student's own leaves are traced on the
    cheapest budget only, on every fourth family."""
    me, other_spec = spec_of(student), spec_of(previous) if previous else spec_of(student)
    games = []
    for family in range(families):
        if move_ms:
            traced = family % 4 == 0
            games.append(Game(family, me, other_spec, 0, move_ms, traced))
            games.append(Game(family, other_spec, me, 0, move_ms, False))
            continue
        budget = BUDGETS[family % len(BUDGETS)]
        other = budget if previous else BUDGETS[(family + 1) % len(BUDGETS)]
        traced = budget == BUDGETS[0] and family % 4 == 0
        games.append(Game(family, me, other_spec, budget, 0, traced))
        games.append(Game(family, other_spec, me, other, 0, False))
    return games


def alternatives(board: np.ndarray, since: int, ply: int, played: int) -> list[tuple[np.ndarray, int, int]]:
    """The children of every legal move except the one played (ongoing only)."""
    import engine

    cells = board.tolist()
    out = []
    for action, legal in enumerate(engine.legal_mask(cells)):
        if not legal or action == played:
            continue
        child, s, p, outcome = engine.apply(cells, since, ply, action, SITE_CLOCK)
        if outcome == 0:
            out.append((np.frombuffer(child, dtype=np.uint8).copy(), s, p))
    return out


def tactical(board: np.ndarray) -> bool:
    import engine

    wins, _, threes = engine.tactics(board.tolist())
    return any(wins) or any(threes)


def play(game: Game, seed: int, out: Path) -> tuple[dict, Path | None]:
    """One `bot play`; returns the record and the leaves file, if traced."""
    leaves = out / f"leaves_{game.family}_{seed}.jsonl" if game.leaves else None
    command = [
        str(bot_binary()),
        "play",
        "--first",
        game.first,
        "--second",
        game.second,
        *(["--move-ms", str(game.move_ms)] if game.move_ms else ["--sims", str(game.budget)]),
        "--player-threads",
        "1",
        "--seed",
        str(seed),
    ]
    if leaves:
        leaves.unlink(missing_ok=True)
        command += ["--leaves", str(leaves)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=workspace_root(), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"bot play failed: {result.stderr.strip()}")
    record = json.loads(result.stdout)
    record.update(family=game.family, seed=seed, budget=game.budget, move_ms=game.move_ms, first_spec=game.first)
    return record, leaves


def sample_leaves(path: Path, keep: int, rng: np.random.Generator) -> list[tuple[np.ndarray, int, int]]:
    """Reservoir-sample `keep` leaf records from one traced game, then delete the file."""
    chosen: list[tuple[np.ndarray, int, int]] = []
    seen = 0
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            record = json.loads(line)
            item = (np.array(record["board"], dtype=np.uint8), int(record["since_capture"]), int(record["ply"]))
            seen += 1
            if len(chosen) < keep:
                chosen.append(item)
            else:
                slot = int(rng.integers(0, seen))
                if slot < keep:
                    chosen[slot] = item
    path.unlink()
    return chosen


def value_drops(record: dict) -> set[int]:
    """Plies whose mover's search value fell by VALUE_DROP by its next move."""
    values = [None if s is None else float(s["value"]) for s in record["stats"]]
    offset = len(record["moves"]) - len(values)
    drops = set()
    for i in range(len(values) - 2):
        a, b = values[i], values[i + 2]
        if a is not None and b is not None and a - b >= VALUE_DROP:
            drops.add(offset + i)
    return drops


def collect(
    round_name: str,
    student: Path,
    previous: Path | str | None,
    families: int,
    states: int,
    workers: int,
    seed: int,
    move_ms: int = 0,
) -> Path:
    started = time.perf_counter()
    out = run_dir("nnue_collect") / round_name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    games = schedule(student, previous, families, move_ms)
    with ThreadPoolExecutor(workers) as pool:
        played = list(pool.map(lambda g: play(g, seed + g.family, out), games))
    with (out / "games.jsonl").open("w", encoding="utf-8") as sink:
        for record, _ in played:
            sink.write(json.dumps(record) + "\n")

    pools: dict[str, list[tuple[np.ndarray, int, int, int, int, bool]]] = {k: [] for k in SHARES}
    # Every pool item: board, since_capture, ply, game id, outcome (mover's view), outcome known.
    leaf_keep = int(SHARES["leaf"] * states * LEAF_RESERVOIR / max(1, sum(g.leaves for g in games)))
    for index, (record, leaves) in enumerate(played):
        roots, _ = replay(record["moves"])
        winner = record["winner"]
        drops = value_drops(record)
        for board, since, ply, action in roots:
            mover_first = ply % 2 == 0
            outcome = 0 if winner is None else (1 if (winner == 0) == mover_first else -1)
            pools["root"].append((board, since, ply, index, outcome, True))
            if ply in drops:
                for child, s, p in alternatives(board, since, ply, action):
                    pools["alternative"].append((child, s, p, index, 0, False))
        if leaves:
            for board, since, ply in sample_leaves(leaves, leaf_keep, rng):
                pools["leaf"].append((board, since, ply, index, 0, False))
    candidates = pools["root"] + pools["leaf"]
    pools["targeted"] = [
        item
        for item in candidates
        if np.count_nonzero(item[0]) <= SPARSE_PIECES or item[1] >= LONG_CLOCK or tactical(item[0])
    ]

    chosen: list[tuple[np.ndarray, int, int, int, int, bool]] = []
    for name, share in SHARES.items():
        pool = pools[name]
        want = min(len(pool), int(round(share * states)))
        picks = rng.choice(len(pool), want, replace=False) if want else []
        chosen.extend(pool[i] for i in picks)
    rows = {
        "board": np.stack([c[0] for c in chosen]).astype(np.uint8),
        "since_capture": np.array([c[1] for c in chosen], dtype=np.uint16),
        "ply": np.array([c[2] for c in chosen], dtype=np.uint16),
        "game": np.array([c[3] for c in chosen], dtype=np.uint32),
        "outcome": np.array([c[4] for c in chosen], dtype=np.int8),
        "outcome_ok": np.array([c[5] for c in chosen], dtype=np.bool_),
    }
    n = len(chosen)
    clock = np.full(n, SITE_CLOCK, dtype=np.uint16)
    _, first = np.unique(context_hash(rows["board"], rows["since_capture"], clock), return_index=True)
    first.sort()
    rows = {k: v[first] for k, v in rows.items()}
    n = len(first)
    rows.update(
        capture_clock=clock[:n],
        target=np.zeros(n, dtype=np.float32),
        weight=np.ones(n, dtype=np.float32),
        kind=np.full(n, data.KIND_RAW, dtype=np.uint8),
        source=np.full(n, data.SOURCE_STUDENT, dtype=np.uint8),
        orbit=features.orbit_hash(rows["board"]),
    )
    rows["split"] = assign_splits(rows["game"], rows["orbit"], seed)
    target = data_dir(round_name)
    results = [r["winner"] for r, _ in played]
    data.write(
        target,
        rows,
        {
            "producer": "student",
            "sources": {data.SOURCE_STUDENT: "student games (nnue.collect)"},
            "student": {"path": str(student), "sha256": hashlib.sha256(student.read_bytes()).hexdigest()},
            "previous": str(previous) if previous else None,
            "move_ms": move_ms,
            "rules": {"capture_clock": SITE_CLOCK, "repetition_draw": False},
            "seed": seed,
            "games": len(played),
            "first_wins": results.count(0),
            "second_wins": results.count(1),
            "draws": results.count(None),
            "pools": {k: len(v) for k, v in pools.items()},
            "sampled": n,
            "seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps({"event": "written", "dataset": str(target), "rows": n, "games": len(played)}))
    return target


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", required=True, help="dataset and game directory name")
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=None, help="the previous network as the opponent")
    parser.add_argument("--opponent", default=None, help="any `bot` player spec as the opponent (sq:..., conv:...)")
    parser.add_argument("--move-ms", type=int, default=0, help="wall clock per move instead of node budgets")
    parser.add_argument("--families", type=int, default=64)
    parser.add_argument("--states", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args(argv)
    previous = args.previous.resolve() if args.previous else args.opponent
    collect(
        args.round, args.student.resolve(), previous, args.families, args.states, args.workers, args.seed, args.move_ms
    )


if __name__ == "__main__":
    main(sys.argv[1:])
