"""Self-play games for the dashboard: batch statistics, single games move by move, and the games still being played.

Standard library only. The rules needed for a replay are small: a piece steps to one of the eight neighbouring squares,
onto an empty square or onto an enemy piece it beats (rock beats scissors, scissors beats paper, paper beats rock), and
takes its place. A recorded move is legal, so replaying it is moving the piece.

Frames. Move tokens (`b4-a4`) are absolute: Blue's home is a1, Blue moves first. A journal's events and a root's actions
are canonical, in the mover's frame (direction * 81 + origin); for Red, the mover at odd plies, the board is reflected
across the anti-diagonal. Boards sent to the page are absolute: 0 empty, 1-3 Blue rock/paper/scissors, 4-6 Red.

A published batch is read once per shard, in a background thread, and its per-game summaries are cached under
runs/nnue_dashboard/cache/games/, so a 51,200-game batch costs its parse once.
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
import threading
import time
from pathlib import Path

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
SETUP = {
    1: (28, 20, 12),
    2: (37, 29, 21, 13),
    3: (38, 30, 22),
}  # rock, paper, scissors of the side at a1
SCORE_SCALE = 600.0
DECIDED = 0.9
ENDS = [
    "Goal",
    "Elimination",
    "Stalemate",
    "CaptureClock",
    "PlyCap",
    "Forfeit",
    "Interrupted",
]


def mirror_anti(square: int) -> int:
    rank, file = divmod(square, 9)
    return (8 - file) * 9 + (8 - rank)


def initial_board() -> list[int]:
    board = [0] * 81
    for piece, squares in SETUP.items():
        for square in squares:
            board[square] = piece
            board[mirror_anti(square)] = piece + 3
    return board


def square_of(name: str) -> int:
    return (int(name[1]) - 1) * 9 + ord(name[0]) - ord("a")


def name_of(square: int) -> str:
    return "abcdefghi"[square % 9] + str(square // 9 + 1)


def parse_token(token: str) -> tuple[int, int]:
    text = token.lower().lstrip("rps")
    origin, target = (text[:2], text[3:]) if len(text) == 5 else (text[:2], text[2:])
    return square_of(origin), square_of(target)


def canonical_move(action: int, ply: int) -> tuple[int, int]:
    """A canonical action of the mover at `ply` as absolute (from, to)."""
    direction, origin = divmod(int(action), 81)
    d_rank, d_file = DIRS[direction]
    rank, file = divmod(origin, 9)
    target = (rank + d_rank) * 9 + file + d_file
    return (mirror_anti(origin), mirror_anti(target)) if ply % 2 else (origin, target)


def replay(moves: list[tuple[int, int]]) -> tuple[list[dict], list[int]]:
    """Each move with whether it captured and which piece moved; and the final board."""
    board = initial_board()
    out = []
    for origin, target in moves:
        piece, taken = board[origin], board[target]
        board[target], board[origin] = piece, 0
        out.append({"from": origin, "to": target, "piece": piece, "taken": taken})
    return out, board


def root_value(root: dict) -> float | None:
    """The root's score as Blue's expected result in [-1, 1]."""
    score = root.get("root_score")
    if score is None:
        return None
    value = math.tanh(float(score) / SCORE_SCALE)
    return value if int(root.get("mover", root.get("ply", 0) % 2)) == 0 else -value


def summarise(record: dict) -> dict:
    moves = [parse_token(t) for t in record.get("moves", [])]
    played, board = replay(moves)
    roots = sorted(record.get("roots", []), key=lambda r: r.get("ply", 0))
    winner = record.get("winner")
    result = 1 if winner == 0 else -1 if winner == 1 else 0
    decided = None
    if result:
        last_unsure = -1
        for root in roots:
            value = root_value(root)
            if value is not None and value * result < DECIDED:
                last_unsure = root["ply"]
        later = [r["ply"] for r in roots if r["ply"] > last_unsure]
        decided = later[0] if later else None
    depths = [
        r.get("completed_depth") for r in roots if r.get("completed_depth") is not None
    ]
    return {
        "id": record.get("game_id"),
        "plies": record.get("plies") or len(moves),
        "end": record.get("end") or "none",
        "winner": winner,
        "censored": bool(record.get("censored")),
        "captures": sum(1 for m in played if m["taken"]),
        "blue_left": sum(1 for c in board if 1 <= c <= 3),
        "red_left": sum(1 for c in board if c >= 4),
        "random": len(record.get("random_plies") or []),
        "alternatives": sum(
            1
            for r in roots
            if r.get("alternative") and r.get("played_action") != r.get("searched_best")
        ),
        "decided": decided,
        "depth": round(sum(depths) / len(depths), 2) if depths else None,
    }


def detail(record: dict, live: bool = False) -> dict:
    """One game for the viewer: the moves, and per ply what the search saw."""
    moves = [parse_token(t) for t in record.get("moves", [])]
    played, _ = replay(moves)
    randoms = set(record.get("random_plies") or [])
    by_ply = {r["ply"]: r for r in record.get("roots", [])}
    plies = []
    for ply, move in enumerate(played):
        root = by_ply.get(ply)
        entry = {
            **move,
            "token": f"{name_of(move['from'])}-{name_of(move['to'])}",
            "random": ply in randoms or ply < record.get("opening_plies", 0),
        }
        if root:
            entry.update(
                value=root_value(root),
                depth=root.get("completed_depth"),
                nodes=root.get("nodes"),
                since=root.get("since_capture"),
                kind=root.get("score_kind"),
            )
            best = root.get("searched_best")
            if (
                best is not None
                and root.get("played_action") is not None
                and best != root["played_action"]
            ):
                origin, target = canonical_move(best, ply)
                entry["best"] = {
                    "from": origin,
                    "to": target,
                    "token": f"{name_of(origin)}-{name_of(target)}",
                }
            if root.get("alternative"):
                entry["alternative"] = True
        plies.append(entry)
    return {
        "id": record.get("game_id"),
        "initial": initial_board(),
        "plies": plies,
        "end": record.get("end"),
        "winner": record.get("winner"),
        "censored": record.get("censored"),
        "live": live,
        "capture_clock": record.get("capture_clock"),
        "opening_plies": record.get("opening_plies", 0),
        "summary": summarise(record) if not live else None,
    }


def journal_record(path: Path) -> dict | None:
    """An in-flight game: the journal's header record with its committed events applied."""
    try:
        lines = path.read_bytes().split(b"\n")
    except OSError:
        return None
    complete = lines[:-1]  # the last element is empty or a torn write
    if not complete:
        return None
    try:
        record = json.loads(complete[0])
    except ValueError:
        return None
    moves = list(record.get("moves", []))
    roots = list(record.get("roots", []))
    randoms = list(record.get("random_plies") or [])
    for raw in complete[1:]:
        try:
            event = json.loads(raw)
        except ValueError:
            break
        if event.get("action") is None:
            continue
        ply = int(event["ply"])
        origin, target = canonical_move(event["action"], ply)
        moves.append(f"{name_of(origin)}-{name_of(target)}")
        if event.get("random"):
            randoms.append(ply)
        if event.get("root"):
            roots.append(event["root"])
    record.update(
        moves=moves,
        roots=roots,
        random_plies=randoms,
        plies=len(moves),
        end=None,
        winner=None,
    )
    return record


def aggregate(games: list[dict]) -> dict:
    n = len(games)
    if not n:
        return {"games": 0}
    lengths = sorted(g["plies"] for g in games)
    ends: dict[str, int] = {}
    for g in games:
        ends[g["end"]] = ends.get(g["end"], 0) + 1
    real = [g for g in games if not g["censored"]]
    blue = sum(
        1.0 if g["winner"] == 0 else 0.5 if g["winner"] is None else 0.0 for g in real
    )
    decided = sorted(g["decided"] for g in games if g["decided"] is not None)
    top = max(lengths)
    width = 25 if top <= 600 else 50
    bins = [0] * (top // width + 1)
    for length in lengths:
        bins[length // width] += 1
    q = lambda p: lengths[min(n - 1, int(p * n))]
    depths = [g["depth"] for g in games if g["depth"] is not None]
    return {
        "games": n,
        "length_mean": sum(lengths) / n,
        "length_median": statistics.median(lengths),
        "length_p10": q(0.1),
        "length_p90": q(0.9),
        "length_max": top,
        "histogram": {"width": width, "counts": bins},
        "ends": ends,
        "blue_score": blue / len(real) if real else None,
        "draw_share": sum(1 for g in real if g["winner"] is None) / len(real)
        if real
        else None,
        "censored_share": (n - len(real)) / n,
        "captures_mean": sum(g["captures"] for g in games) / n,
        "decided_median": statistics.median(decided) if decided else None,
        "decided_share": len(decided) / n,
        "random_mean": sum(g["random"] for g in games) / n,
        "alternatives_mean": sum(g["alternatives"] for g in games) / n,
        "depth_mean": sum(depths) / len(depths) if depths else None,
        "pieces_left_mean": sum(g["blue_left"] + g["red_left"] for g in games) / n,
    }


class GameIndex:
    """Per-game summaries of every published shard, read once in the background and cached on disk."""

    def __init__(self, runs_dir: Path):
        self.runs = Path(runs_dir)
        self.cache = self.runs / "nnue_dashboard" / "cache" / "games"
        self.lock = threading.Lock()
        self.memory: dict[tuple[str, str], list[dict]] = {}
        self.wanted: list[str] = []
        self.busy: dict[str, tuple[int, int]] = {}
        self.failed: dict[
            tuple[str, str], float
        ] = {}  # unreadable shards, retried after a minute
        self.wake = threading.Event()
        threading.Thread(target=self.work, name="games", daemon=True).start()

    def manifest(self, name: str) -> dict | None:
        try:
            return json.loads(
                (self.runs / "nnue_selfplay" / name / "manifest.json").read_text()
            )
        except (OSError, ValueError):
            return None

    def cache_file(self, name: str, shard: dict) -> Path:
        return (
            self.cache / name / f"{shard['file']}.{shard.get('sha256', 'x')[:12]}.json"
        )

    def shard_summaries(self, name: str, shard: dict) -> list[dict] | None:
        key = (name, shard["file"] + shard.get("sha256", ""))
        with self.lock:
            if key in self.memory:
                return self.memory[key]
        path = self.cache_file(name, shard)
        try:
            games = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        with self.lock:
            self.memory[key] = games
        return games

    def read_shard(self, name: str, shard: dict) -> list[dict]:
        games = []
        with gzip.open(
            self.runs / "nnue_selfplay" / name / shard["file"], "rt", encoding="utf-8"
        ) as lines:
            for line in lines:
                if line.strip():
                    record = json.loads(line)
                    if "moves" in record:
                        games.append(summarise(record))
        path = self.cache_file(name, shard)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(games, separators=(",", ":")))
        tmp.replace(path)
        with self.lock:
            self.memory[(name, shard["file"] + shard.get("sha256", ""))] = games
        return games

    def request(self, names: list[str]) -> None:
        with self.lock:
            for name in names:
                if name not in self.wanted:
                    self.wanted.append(name)
        self.wake.set()

    def work(self) -> None:
        while True:
            self.wake.wait(timeout=30)
            self.wake.clear()
            while True:
                with self.lock:
                    queue = list(self.wanted)
                todo = None
                for name in queue:
                    manifest = self.manifest(name)
                    if not manifest:
                        continue
                    shards = [s for s in manifest.get("shards", []) if s.get("file")]
                    now = time.time()
                    missing = [
                        s
                        for s in shards
                        if self.shard_summaries(name, s) is None
                        and now - self.failed.get((name, s["file"]), 0) > 60
                    ]
                    with self.lock:
                        self.busy[name] = (len(shards) - len(missing), len(shards))
                    if missing:
                        todo = (name, missing[0])
                        break
                if not todo:
                    break
                try:
                    self.read_shard(*todo)
                except (OSError, ValueError, EOFError):
                    with (
                        self.lock
                    ):  # a shard being replaced, missing or torn: try again later
                        self.failed[(todo[0], todo[1]["file"])] = time.time()
                time.sleep(0.01)

    def games(self, name: str) -> tuple[list[dict], dict]:
        manifest = self.manifest(name) or {}
        shards = [s for s in manifest.get("shards", []) if s.get("file")]
        out, ready = [], 0
        for shard in shards:
            got = self.shard_summaries(name, shard)
            if got is not None:
                out.extend(got)
                ready += 1
        if ready < len(shards):
            self.request([name])
        return out, {
            "shards_ready": ready,
            "shards": len(shards),
            "complete_batch": bool(manifest.get("complete")),
        }

    def stats(self, name: str) -> dict:
        games, progress = self.games(name)
        return {"run": name, **progress, **aggregate(games)}

    def listing(
        self,
        name: str,
        end: str = "",
        sort: str = "id",
        offset: int = 0,
        limit: int = 50,
    ) -> dict:
        games, progress = self.games(name)
        if end:
            games = [
                g
                for g in games
                if g["end"] == end
                or (end == "blue" and g["winner"] == 0)
                or (end == "red" and g["winner"] == 1)
                or (end == "draw" and g["winner"] is None and not g["censored"])
            ]
        keys = {
            "id": lambda g: g["id"],
            "longest": lambda g: -g["plies"],
            "shortest": lambda g: g["plies"],
            "captures": lambda g: -g["captures"],
            "earliest_decided": lambda g: (g["decided"] is None, g["decided"] or 0),
        }
        games = sorted(games, key=keys.get(sort, keys["id"]))
        return {
            "total": len(games),
            "games": games[offset : offset + limit],
            **progress,
        }

    def live(self, name: str) -> list[dict]:
        folder = self.runs / "nnue_selfplay" / name / "active"
        out = []
        for path in sorted(folder.glob("*.jsonl")) if folder.is_dir() else []:
            record = journal_record(path)
            if not record:
                continue
            played, board = replay([parse_token(t) for t in record["moves"]])
            values = [
                root_value(r)
                for r in record["roots"]
                if r.get("root_score") is not None
            ]
            out.append(
                {
                    "id": record.get("game_id"),
                    "plies": len(played),
                    "board": board,
                    "last": played[-1] if played else None,
                    "value": values[-1] if values else None,
                }
            )
        return out

    def game(self, name: str, game_id: int) -> dict | None:
        folder = self.runs / "nnue_selfplay" / name
        journal = folder / "active" / f"{game_id:012}.jsonl"
        if journal.is_file():
            record = journal_record(journal)
            return detail(record, live=True) if record else None
        pending = folder / "pending" / f"{game_id:012}.json"
        if pending.is_file():
            try:
                return detail(json.loads(pending.read_text()))
            except (OSError, ValueError):
                pass
        shards = [
            s for s in (self.manifest(name) or {}).get("shards", []) if s.get("file")
        ]
        # the shard whose id range holds the game first, then the rest in case ids are not contiguous
        likely = [
            s
            for s in shards
            if s.get("first_game", 0)
            <= game_id
            < s.get("first_game", 0) + (s.get("counts") or {}).get("games", 0)
        ]
        for shard in likely + [s for s in shards if s not in likely]:
            try:
                with gzip.open(folder / shard["file"], "rt", encoding="utf-8") as lines:
                    for line in lines:
                        if line.strip() and f'"game_id":{game_id},' in line.replace(
                            " ", ""
                        ):
                            record = json.loads(line)
                            if record.get("game_id") == game_id:
                                return detail(record)
            except (OSError, EOFError):
                continue
        return None
