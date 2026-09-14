"""The `bot analyse` process pool, position resolution and the eval response."""

from __future__ import annotations

import itertools
import json
import logging
import os
import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from arena.core.game import (
    GameRecord,
    MoveRecord,
    PositionState,
    Result,
    position_key,
    replay,
    start_side,
    terminal_at,
)
from arena.core.notation import move_token, parse_token
from arena.core.pgn import parse_fen, parse_pieces
from arena.core.symmetry import symmetry_key
from arena.players import resolve_spec, split_spec
from arena.settings import Settings

LOG = logging.getLogger("arena.engines")
QUICK_SIMS = 0
STANDARD_SIMS = 128
DEEP_SIMS = 800
MAX_SIMS = 2000
BUDGETS = {"quick": QUICK_SIMS, "standard": STANDARD_SIMS, "deep": DEEP_SIMS}
SIMS_PER_100MS = 40
REQUEST_TIMEOUT = 60.0
HEAD_SPECS: list[dict[str, Any]] = [
    {
        "id": "policy",
        "label": "Policy",
        "group": "policy",
        "kind": "move_scalar",
        "description": "Network policy probability for every legal move.",
        "default_on": True,
        "order": 10,
        "params": {},
    },
    {
        "id": "search",
        "label": "Search",
        "group": "policy",
        "kind": "move_scalar",
        "description": "Share of root visits each move received.",
        "default_on": True,
        "order": 20,
        "params": {},
    },
    {
        "id": "q",
        "label": "Q",
        "group": "value",
        "kind": "move_scalar",
        "description": "Network action value for each legal move from the mover's view.",
        "default_on": False,
        "order": 30,
        "params": {},
    },
    {
        "id": "draw",
        "label": "Draw",
        "group": "value",
        "kind": "move_scalar",
        "description": "Network draw probability for each legal move.",
        "default_on": False,
        "order": 40,
        "params": {},
    },
    {
        "id": "value",
        "label": "Value",
        "group": "value",
        "kind": "scalar",
        "description": "Position value from Blue's view, beside the searched value.",
        "default_on": False,
        "order": 50,
        "params": {},
    },
    {
        "id": "plies_to_end",
        "label": "Game length",
        "group": "forecast",
        "kind": "histogram",
        "description": "Predicted distribution and expectation of plies remaining.",
        "default_on": False,
        "order": 60,
        "params": {},
    },
]


class EngineError(RuntimeError):
    """The analysis pool could not answer a request."""


def resolve_sims(budget: str | dict[str, Any] | int | None) -> tuple[str, int]:
    """Normalise a public budget into its cache key and simulation count."""
    if budget is None:
        budget = "standard"
    if isinstance(budget, bool):
        raise TypeError("budget must be a name or an object with sims or ms")
    if isinstance(budget, int):
        budget = {"sims": budget}
    if isinstance(budget, str):
        key = budget.lower()
        if key not in BUDGETS:
            raise ValueError(f"unknown analysis budget {budget!r}")
        return key, BUDGETS[key]
    if not isinstance(budget, dict):
        raise TypeError("budget must be quick/standard/deep or an object with sims or ms")
    supplied = [name for name in ("sims", "ms") if name in budget]
    if len(supplied) != 1:
        raise ValueError("custom budget must contain exactly one of sims or ms")
    name = supplied[0]
    value = budget[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"budget {name} must be an integer")
    if name == "sims":
        if not 0 <= value <= MAX_SIMS:
            raise ValueError(f"budget sims must be between 0 and {MAX_SIMS}")
        return f"sims:{value}", value
    if value <= 0:
        raise ValueError("budget ms must be positive")
    return f"ms:{value}", min(MAX_SIMS, value * SIMS_PER_100MS // 100)


@dataclass
class ResolvedPosition:
    """A position plus the request that reproduces it for the bot."""

    board: list[int]
    to_move: str
    psc: int
    ply: int
    setup: list[int] | None
    setup_to_move: str = "blue"
    setup_psc: int = 0
    setup_ply: int = 0
    moves: list[str] = field(default_factory=list)
    terminal: Result | None = None
    history: list[dict[str, int]] = field(default_factory=list)

    def state(self) -> PositionState:
        """The position as a replay state, for keys and symmetry."""
        return PositionState(self.board, self.to_move, self.ply, self.psc, 1, self.terminal)

    def request(self, sims: int, multipv: int) -> dict[str, Any]:
        """The bot analyse request for this position."""
        return {
            "setup": self.setup,
            "to_move": self.setup_to_move,
            "psc": self.setup_psc,
            "ply": self.setup_ply,
            "moves": list(self.moves),
            "sims": sims,
            "multipv": multipv,
        }


def _square(value: Any) -> int:
    """Coerce a JSON move square into an absolute index."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("move squares must be integers")
    if not 0 <= value < 81:
        raise ValueError("move square is off the board")
    return value


def _moves_from(history: Any) -> list[dict[str, int]]:
    """Read a list of {from, to} moves."""
    moves = []
    for item in history or []:
        if not isinstance(item, dict):
            raise ValueError("history moves must be objects")
        moves.append({"from": _square(item.get("from")), "to": _square(item.get("to"))})
    return moves


def resolve_position(ref: Any, conn: Any = None) -> ResolvedPosition:
    """Resolve a public PositionRef into a position and its bot request."""
    if not isinstance(ref, dict):
        raise ValueError("position must be an object")
    variants = [name for name in ("game_id", "fen", "board") if ref.get(name) is not None]
    if len(variants) != 1:
        raise ValueError("position must contain exactly one of game_id, fen, or board")
    if variants[0] == "game_id":
        return _game_position(ref, conn)
    if variants[0] == "fen":
        board, side = _fen_board(ref)
        return _board_position(board, side, ref)
    board = list(ref["board"])
    if len(board) != 81 or any(isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= 6 for c in board):
        raise ValueError("board must be 81 cells encoded 0..6")
    side = str(ref.get("to_move", "blue")).lower()
    if side not in ("blue", "red"):
        raise ValueError("to_move must be blue or red")
    return _board_position(board, side, ref)


def _fen_board(ref: dict[str, Any]) -> tuple[list[int], str]:
    """Parse a FEN position reference into an absolute board and side to move."""
    fen = str(ref.get("fen", "")).strip()
    if not fen:
        raise ValueError("fen may not be empty")
    frame = str(ref.get("frame") or "meaf").lower()
    if frame not in ("meaf", "henhen"):
        raise ValueError("frame must be meaf or henhen")
    if frame == "henhen":
        board, side, _territory = parse_fen(fen)
        return board, side
    fields = fen.split()
    if len(fields) < 2:
        raise ValueError("FEN needs piece and side-to-move fields")
    field_side = fields[1].lower()
    if field_side not in ("b", "blue", "r", "red", "-"):
        raise ValueError(f"unknown FEN side to move {fields[1]!r}")
    return parse_pieces(fields[0]), "red" if field_side in ("r", "red") else "blue"


def _board_position(board: list[int], side: str, ref: dict[str, Any]) -> ResolvedPosition:
    """Resolve a setup board plus an optional continuation played from it."""
    psc = int(ref.get("psc") or 0)
    ply = int(ref.get("ply") or 0)
    if psc < 0 or ply < 0:
        raise ValueError("psc and ply must be non-negative")
    history = _moves_from(ref.get("history"))
    record = GameRecord(
        setup=list(board),
        meta={"start_to_move": side},
        moves=[MoveRecord(move["from"], move["to"]) for move in history],
    )
    final = replay(record)[-1]
    captured = any(move.capture for move in record.moves)
    quiet = final.psc if captured else psc + final.psc
    return ResolvedPosition(
        board=final.board,
        to_move=final.to_move,
        psc=quiet,
        ply=ply + len(history),
        setup=list(board),
        setup_to_move=side,
        setup_psc=psc,
        setup_ply=ply,
        moves=[move_token(move) for move in record.moves],
        terminal=final.result or terminal_at(final.board, final.to_move, quiet),
        history=history,
    )


def _game_position(ref: dict[str, Any], conn: Any) -> ResolvedPosition:
    """Resolve a stored game's position at one ply."""
    from arena.db import store

    if conn is None:
        raise ValueError("a game position needs a database connection")
    game_id = str(ref.get("game_id") or "")
    record = store.get_game(conn, game_id)
    if record is None:
        raise LookupError(f"unknown game {game_id}")
    states = replay(record)
    if ref.get("ply") is None:
        raise ValueError("a game position needs a ply")
    at = int(ref["ply"])
    if not 0 <= at < len(states):
        raise ValueError(f"game ply {at} is outside 0..{len(states) - 1}")
    state = states[at]
    setup = list(record.setup) if record.setup is not None else None
    return ResolvedPosition(
        board=state.board,
        to_move=state.to_move,
        psc=state.psc,
        ply=state.ply,
        setup=setup,
        setup_to_move=start_side(record) if setup is not None else "blue",
        setup_psc=0,
        setup_ply=0,
        moves=[move_token(move) for move in record.moves[:at]],
        terminal=state.result,
        history=[{"from": move.from_, "to": move.to} for move in record.moves[:at]],
    )


class BotProcess:
    """One long-lived `bot analyse` process serving requests in order."""

    def __init__(self, engine: str, command: list[str], cwd: str, env: dict[str, str]):
        self.engine = engine
        self.command = command
        self.cwd = cwd
        self.env = env
        self.lock = threading.Lock()
        self.counter = itertools.count(1)
        self.proc: subprocess.Popen[str] | None = None

    def _spawn(self) -> subprocess.Popen[str]:
        """Start the process if it is not already running."""
        proc = self.proc
        if proc is not None and proc.poll() is None:
            return proc
        self.proc = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
        )
        return self.proc

    def ask(self, payload: dict[str, Any], timeout: float = REQUEST_TIMEOUT) -> dict[str, Any]:
        """Send one request and return its response, restarting after a failure."""
        with self.lock:
            for attempt in (0, 1):
                try:
                    return self._exchange(payload, timeout)
                except EngineError:
                    self.close()
                    if attempt:
                        raise
        raise EngineError("engine did not answer")

    def _exchange(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Write one request and read its response line."""
        proc = self._spawn()
        request = dict(payload)
        request["id"] = f"r{next(self.counter)}"
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EngineError(f"engine {self.engine} is not accepting requests") from exc
        result: dict[str, Any] = {}
        error: list[str] = []

        def read() -> None:
            """Read one response line from the process."""
            line = proc.stdout.readline()
            if not line:
                error.append(f"engine {self.engine} closed its output")
                return
            try:
                result.update(json.loads(line))
            except ValueError:
                error.append(f"engine {self.engine} wrote an unparseable response")

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reader.join(timeout)
        if reader.is_alive():
            raise EngineError(f"engine {self.engine} timed out after {timeout:.0f}s")
        if error:
            raise EngineError(error[0])
        if result.get("error"):
            raise EngineError(str(result["error"]))
        return result

    def close(self) -> None:
        """Stop the process."""
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()

    def alive(self) -> bool:
        """Whether the process is currently running."""
        return self.proc is not None and self.proc.poll() is None


class AnalysisPool:
    """A least-recently-used pool of `bot analyse` processes, one per engine."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.processes: OrderedDict[str, BotProcess] = OrderedDict()

    def analysable(self, spec: str) -> bool:
        """Whether an engine spec can be analysed by the pool."""
        parts = split_spec(spec)
        return parts is not None and parts[0] != "rpsi"

    def process(self, engine: str, spec: str) -> BotProcess:
        """Return the process for an engine, evicting the least recently used."""
        if not self.analysable(spec):
            raise EngineError(f"engine {engine} does not support analysis")
        with self.lock:
            existing = self.processes.get(engine)
            if existing is not None:
                self.processes.move_to_end(engine)
                return existing
            resolved = resolve_spec(spec, self.settings.players_dir / engine)
            command = [*self.settings.bot_command(), "analyse", "--engine", resolved, "--threads", "1"]
            env = dict(os.environ, LD_LIBRARY_PATH=str(self.settings.root))
            created = BotProcess(engine, command, str(self.settings.root), env)
            self.processes[engine] = created
            while len(self.processes) > max(1, self.settings.analysis_procs):
                _name, evicted = self.processes.popitem(last=False)
                evicted.close()
            return created

    def ask(self, engine: str, spec: str, payload: dict[str, Any], timeout: float = REQUEST_TIMEOUT) -> dict[str, Any]:
        """Send one analyse request to an engine."""
        return self.process(engine, spec).ask(payload, timeout)

    def alive(self) -> bool:
        """Whether any pooled process is running."""
        with self.lock:
            return any(item.alive() for item in self.processes.values())

    def close(self) -> None:
        """Stop every pooled process."""
        with self.lock:
            processes = list(self.processes.values())
            self.processes.clear()
        for item in processes:
            item.close()


def head_manifest(spec: str) -> list[dict[str, Any]]:
    """The head manifest an engine's responses can fill."""
    parts = split_spec(spec)
    if parts is None or parts[0] == "rpsi":
        return []
    return [dict(head) for head in HEAD_SPECS]


def _move_values(mapping: Any, scale: str, render: str, low: float, high: float) -> dict[str, Any] | None:
    """Build a move_scalar overlay from a token-keyed mapping."""
    if not isinstance(mapping, dict):
        return None
    moves = []
    for token, value in mapping.items():
        origin, target = parse_token(str(token))
        moves.append({"from": origin, "to": target, "value": round(float(value), 4)})
    moves.sort(key=lambda move: -move["value"])
    return {"kind": "move_scalar", "moves": moves, "range": [low, high], "render": render, "scale": scale}


def build_heads(response: dict[str, Any], to_move: str) -> dict[str, Any]:
    """Build the six overlays from one bot analyse response."""
    network = response.get("network") or {}
    search = response.get("search") or {}
    heads: dict[str, Any] = {}
    policy = _move_values(network.get("policy"), "sequential", "arrows", 0.0, 1.0)
    if policy is not None:
        heads["policy"] = policy
    q = _move_values(network.get("q"), "diverging", "tint", -1.0, 1.0)
    if q is not None:
        heads["q"] = q
    draw = _move_values(network.get("draw"), "sequential", "tint", 0.0, 1.0)
    if draw is not None:
        heads["draw"] = draw
    lines = search.get("lines") or []
    total = sum(int(line.get("visits", 0)) for line in lines)
    if lines and total > 0:
        shares = {str(line["move"]): int(line.get("visits", 0)) / total for line in lines}
        heads["search"] = _move_values(shares, "sequential", "arrows", 0.0, 1.0)
    sign = 1.0 if to_move == "blue" else -1.0
    items = []
    if network:
        label = "State value head" if network.get("value_source") == "state_head" else "Policy-weighted Q"
        items.append({"label": label, "value": round(sign * float(network.get("value", 0.0)), 4), "format": "signed"})
        if network.get("wdl"):
            for name, p in zip(("Win", "Draw", "Loss"), network["wdl"], strict=True):
                items.append({"label": f"{name} (outcome head)", "value": round(float(p), 4), "format": "percent"})
    if search:
        items.append(
            {"label": "Root value", "value": round(sign * float(search.get("root_value", 0.0)), 4), "format": "signed"}
        )
    plies_left = search.get("plies_left") if search else network.get("plies_left")
    if plies_left is not None:
        items.append({"label": "Plies left", "value": round(float(plies_left), 2), "unit": "plies", "format": "plies"})
    if items:
        heads["value"] = {"kind": "scalar", "items": items}
    forecast = network.get("plies_to_end")
    if isinstance(forecast, dict) and forecast.get("centers"):
        bins = [
            {"label": f"{float(center):g}", "p": round(float(p), 4)}
            for center, p in zip(forecast["centers"], forecast.get("p", []), strict=False)
        ]
        histogram: dict[str, Any] = {"kind": "histogram", "bins": bins, "unit": "plies"}
        if network.get("plies_left") is not None:
            histogram["expectation"] = round(float(network["plies_left"]), 2)
        heads["plies_to_end"] = histogram
    return heads


def network_lines(network: dict[str, Any]) -> list[dict[str, Any]]:
    """Lines for a network-only evaluation: every legal move by policy, unvisited."""
    policy = network.get("policy") or {}
    q = network.get("q") or {}
    return [
        {"move": token, "pv": [token], "q": q.get(token, 0.0), "pi": pi, "visits": 0}
        for token, pi in sorted(policy.items(), key=lambda item: -item[1])
    ]


def build_response(engine: str, resolved: ResolvedPosition, response: dict[str, Any], sims: int) -> dict[str, Any]:
    """Turn one bot analyse response into the public EvalResponse."""
    state = resolved.state()
    keys = {"position_key": position_key(state), "symmetry_key": symmetry_key(state)}
    to_move = str(response.get("to_move") or resolved.to_move)
    terminal = response.get("terminal")
    legal = [{"from": parse_token(token)[0], "to": parse_token(token)[1]} for token in response.get("legal") or []]
    if terminal is not None:
        winner = terminal.get("winner")
        blue = 1.0 if winner == "blue" else (-1.0 if winner == "red" else 0.0)
        mover = blue if to_move == "blue" else -blue
        return {
            "engine": engine,
            **keys,
            "to_move": to_move,
            "legal": [],
            "value": {"blue": blue, "mover": mover, "source": "network"},
            "search": {"sims": 0, "nodes": 0, "evaluations": 0, "ms": 0, "lines": []},
            "heads": {},
            "terminal": terminal,
        }
    network = response.get("network") or {}
    search = response.get("search") or {}
    mover = float(search.get("root_value", 0.0)) if search else float(network.get("value", 0.0))
    blue = mover if to_move == "blue" else -mover
    lines = []
    for rank, line in enumerate(search.get("lines") or network_lines(network), 1):
        origin, target = parse_token(str(line["move"]))
        lines.append(
            {
                "move": {"from": origin, "to": target},
                "pv": [{"from": parse_token(token)[0], "to": parse_token(token)[1]} for token in line.get("pv") or []],
                "q": round(float(line.get("q", 0.0)), 4),
                "pi": round(float(line.get("pi", 0.0)), 4),
                "visits": int(line.get("visits", 0)),
                "rank": rank,
            }
        )
    return {
        "engine": engine,
        **keys,
        "to_move": to_move,
        "legal": legal,
        "value": {"blue": round(blue, 4), "mover": round(mover, 4), "source": "search" if search else "network"},
        "search": {
            "sims": int(search.get("sims", sims)),
            "nodes": int(search.get("nodes", 0)),
            "evaluations": int(search.get("evaluations", 0)),
            "ms": int(search.get("ms", 0)),
            "lines": lines,
        },
        "heads": build_heads(response, to_move),
        "terminal": None,
    }


def truncate_lines(response: dict[str, Any], multipv: int) -> dict[str, Any]:
    """Return a copy of an eval response with its lines cut to multipv."""
    trimmed = dict(response)
    search = dict(trimmed.get("search") or {})
    search["lines"] = list(search.get("lines") or [])[: max(1, multipv)]
    trimmed["search"] = search
    return trimmed


def terminal_response(engine: str, resolved: ResolvedPosition) -> dict[str, Any]:
    """The eval response of a position the rules have already decided."""
    result = resolved.terminal
    assert result is not None
    return build_response(
        engine,
        resolved,
        {
            "to_move": resolved.to_move,
            "legal": [],
            "terminal": {"winner": result.winner, "reason": result.reason},
            "network": None,
            "search": None,
        },
        0,
    )


def evaluate(
    pool: AnalysisPool, conn: Any, engine: str, spec: str, ref: Any, budget: Any = "standard", multipv: int = 4
) -> dict[str, Any]:
    """Evaluate one position, serving and filling the analyses cache."""
    from arena.db import store

    _key, sims = resolve_sims(budget)
    resolved = resolve_position(ref, conn)
    if resolved.terminal is not None:
        return truncate_lines(terminal_response(engine, resolved), multipv)
    key = position_key(resolved.state())
    cached = store.cached_analysis(conn, key, engine, sims)
    if cached is not None:
        return truncate_lines(cached, multipv)
    raw = pool.ask(engine, spec, resolved.request(sims, multipv))
    response = build_response(engine, resolved, raw, sims)
    store.save_analysis(conn, key, engine, sims, multipv, response)
    return truncate_lines(response, multipv)


def played_evidence(response: dict[str, Any], move: dict[str, int]) -> dict[str, Any] | None:
    """Describe the evaluated line for one played move."""
    for line in (response.get("search") or {}).get("lines") or []:
        if line["move"] == move:
            source = "search" if int(line.get("visits", 0)) > 0 else "network"
            return {
                "move": dict(move),
                "q": line["q"],
                "pi": line["pi"],
                "visits": int(line.get("visits", 0)),
                "source": source,
            }
    return None
