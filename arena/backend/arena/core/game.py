"""Game records and the absolute-frame rules replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias

from .frames import Frame

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
STAGNATION_PLIES = 200
Side: TypeAlias = Literal["blue", "red"]


def beats(piece: int, victim: int) -> bool:
    """Whether a piece type (1..3) captures a victim type (1..3)."""
    return (piece - victim) % 3 == 1


def mirror_anti(sq: int) -> int:
    """Reflect a square across the anti-diagonal, mapping a1 to i9."""
    rank, file = divmod(sq, 9)
    return (8 - file) * 9 + (8 - rank)


def initial_board_list() -> list[int]:
    """The frozen opening position in the absolute frame."""
    board = [0] * 81
    groups = {1: ("b4", "c3", "d2"), 2: ("b5", "c4", "d3", "e2"), 3: ("c5", "d4", "e3")}
    for piece, squares in groups.items():
        for name in squares:
            sq = (int(name[1]) - 1) * 9 + ord(name[0]) - ord("a")
            board[sq] = piece
            board[mirror_anti(sq)] = piece + 3
    return board


@dataclass
class Result:
    """The outcome of a game: winner, rules reason and the source's own text."""

    winner: Side | None
    reason: str
    reported: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> Result | None:
        """Rebuild a result from JSON, passing None through."""
        return None if value is None else cls(**value)


@dataclass
class PlayerRef:
    """One side's identity in a game record."""

    name: str
    id: str | None = None
    rating_before: float | None = None
    rating_after: float | None = None
    kind: Literal["human", "bot", "engine"] = "human"
    spec: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form, omitting unset optional fields."""
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PlayerRef:
        """Rebuild a player reference from JSON."""
        return cls(**value)


@dataclass
class MoveRecord:
    """One played move plus the annotations a source or the arena attached."""

    from_: int
    to: int
    capture: bool = False
    emt_ms: int | None = None
    clock_ms: dict[str, int] | None = None
    comment: str | None = None
    nags: list[str] | None = None
    stats: dict[str, Any] | None = None

    @property
    def from_sq(self) -> int:
        """The origin square, named without the trailing underscore."""
        return self.from_

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form with the contract's `from` key."""
        out: dict[str, Any] = {"from": self.from_, "to": self.to, "capture": self.capture}
        for key in ("emt_ms", "clock_ms", "comment", "nags", "stats"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MoveRecord:
        """Rebuild a move from JSON, dropping any rendered notation."""
        data = dict(value)
        data.pop("san", None)
        data["from_"] = data.pop("from")
        return cls(**data)


@dataclass
class PositionState:
    """A position reached during a replay."""

    board: list[int]
    to_move: Side
    ply: int
    psc: int
    repetition: int
    result: Result | None = None


@dataclass
class GameRecord:
    """A complete game: players, setup, moves, result and provenance."""

    id: str | None = None
    source: Literal["henhen", "meaf", "local", "arena"] = "local"
    source_id: str | None = None
    source_url: str | None = None
    frame: Frame = "meaf"
    players: dict[str, PlayerRef] = field(default_factory=lambda: {"blue": PlayerRef("Blue"), "red": PlayerRef("Red")})
    setup: list[int] | None = None
    moves: list[MoveRecord] = field(default_factory=list)
    result: Result = field(default_factory=lambda: Result(None, "unknown", ""))
    time_control: dict[str, int] | None = None
    tags: dict[str, str] = field(default_factory=dict)
    started_at: str | None = None
    ended_at: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    live: bool = False
    sims: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form stored in `record_json` and served by the API."""
        return {
            "id": self.id,
            "source": self.source,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "frame": self.frame,
            "players": {k: v.to_dict() for k, v in self.players.items()},
            "setup": self.setup,
            "moves": [m.to_dict() for m in self.moves],
            "result": asdict(self.result),
            "time_control": self.time_control,
            "tags": self.tags,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "meta": self.meta,
            "live": self.live,
            "sims": self.sims,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GameRecord:
        """Rebuild a game record from JSON."""
        data = dict(value)
        data["players"] = {k: PlayerRef.from_dict(v) for k, v in data["players"].items()}
        data["moves"] = [MoveRecord.from_dict(v) for v in data.get("moves", [])]
        data["result"] = Result.from_dict(data.get("result")) or Result(None, "unknown", "")
        data.pop("positions", None)
        data.pop("review_status", None)
        data.pop("review_job", None)
        return cls(**data)


@dataclass
class StudyNode:
    """One node of a MEAF workshop tree."""

    parent: int | str | None
    move: MoveRecord | None
    comment: str | None = None
    nags: list[str] = field(default_factory=list)
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        return {
            "parent": self.parent,
            "move": self.move.to_dict() if self.move else None,
            "comment": self.comment,
            "nags": self.nags,
            "label": self.label,
        }


@dataclass
class Study:
    """A move tree with a setup position, used by the workshop codecs."""

    root_setup: list[int]
    nodes: dict[int | str, StudyNode]
    title: str = "Untitled study"
    source: str = "local"
    id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        out: dict[str, Any] = {
            "root_setup": self.root_setup,
            "nodes": {str(k): v.to_dict() for k, v in self.nodes.items()},
            "title": self.title,
            "source": self.source,
        }
        if self.id is not None:
            out["id"] = self.id
        if self.meta:
            out["meta"] = self.meta
        return out


class IllegalMoveError(ValueError):
    """A recorded move is not legal in the position it was played from."""

    def __init__(self, ply: int, move: MoveRecord):
        self.ply = ply
        self.move = move
        super().__init__(f"illegal move at ply {ply}: {move.from_}->{move.to}")


def start_side(record: GameRecord) -> Side:
    """The side to move in a record's setup position."""
    explicit = record.meta.get("start_to_move")
    if explicit in ("blue", "red"):
        return explicit
    fen = record.tags.get("FEN")
    if fen:
        fields = fen.split()
        if len(fields) > 1 and fields[1].lower() == "r":
            return "red"
    return "blue"


def legal_moves(board: list[int], side: Side) -> list[tuple[int, int]]:
    """Every legal (from, to) pair for the side to move."""
    lo, hi = (1, 3) if side == "blue" else (4, 6)
    enemy_lo = 4 if side == "blue" else 1
    result: list[tuple[int, int]] = []
    for sq, cell in enumerate(board):
        if not lo <= cell <= hi:
            continue
        piece_type = cell if side == "blue" else cell - 3
        rank, file = divmod(sq, 9)
        for dr, df in DIRS:
            nr, nf = rank + dr, file + df
            if not (0 <= nr < 9 and 0 <= nf < 9):
                continue
            target = nr * 9 + nf
            victim = board[target]
            if victim == 0:
                result.append((sq, target))
            elif enemy_lo <= victim <= enemy_lo + 2:
                victim_type = victim if side == "red" else victim - 3
                if beats(piece_type, victim_type):
                    result.append((sq, target))
    return result


def terminal_after(board: list[int], mover: Side, psc: int, reported: str = "") -> Result | None:
    """The result of a position just reached by `mover`, or None if play continues."""
    other: Side = "red" if mover == "blue" else "blue"
    goal = 80 if mover == "blue" else 0
    own = range(1, 4) if mover == "blue" else range(4, 7)
    opponent = range(4, 7) if mover == "blue" else range(1, 4)
    if board[goal] in own:
        return Result(mover, "corner", reported)
    if not any(cell in opponent for cell in board):
        return Result(mover, "no_pieces", reported)
    if not legal_moves(board, other):
        return Result(mover, "no_moves", reported)
    if psc >= STAGNATION_PLIES:
        return Result(None, "stagnation", reported)
    return None


def terminal_at(board: list[int], to_move: Side, psc: int) -> Result | None:
    """The result of a standalone position, judged without knowing the last mover."""
    other: Side = "red" if to_move == "blue" else "blue"
    if board[80] in range(1, 4):
        return Result("blue", "corner", "1-0")
    if board[0] in range(4, 7):
        return Result("red", "corner", "0-1")
    if not any(1 <= cell <= 3 for cell in board):
        return Result("red", "no_pieces", "0-1")
    if not any(cell >= 4 for cell in board):
        return Result("blue", "no_pieces", "1-0")
    if not legal_moves(board, to_move):
        return Result(other, "no_moves", "0-1" if other == "red" else "1-0")
    if psc >= STAGNATION_PLIES:
        return Result(None, "stagnation", "1/2-1/2")
    return None


def _rep_key(board: list[int], side: Side) -> tuple[bytes, Side]:
    return bytes(board), side


def replay(record: GameRecord) -> list[PositionState]:
    """Replay a record under the game's rules, returning every position."""
    board = list(record.setup if record.setup is not None else initial_board_list())
    if len(board) != 81 or any(not isinstance(v, int) or not 0 <= v <= 6 for v in board):
        raise ValueError("setup must be 81 cells encoded 0..6")
    side = start_side(record)
    counts = {_rep_key(board, side): 1}
    states = [PositionState(list(board), side, 0, 0, 1, None)]
    terminal: Result | None = None
    for ply, move in enumerate(record.moves):
        if terminal is not None or (move.from_, move.to) not in legal_moves(board, side):
            raise IllegalMoveError(ply, move)
        capture = board[move.to] != 0
        move.capture = capture
        board[move.to], board[move.from_] = board[move.from_], 0
        mover = side
        side = "red" if side == "blue" else "blue"
        psc = 0 if capture else states[-1].psc + 1
        terminal = terminal_after(board, mover, psc, record.result.reported)
        key = _rep_key(board, side)
        counts[key] = counts.get(key, 0) + 1
        states.append(PositionState(list(board), side, ply + 1, psc, counts[key], terminal))
    if terminal and (record.result.winner != terminal.winner or record.result.reason != terminal.reason):
        record.meta.setdefault("rule_notes", []).append(
            f"source reports {record.result.winner}/{record.result.reason}; "
            f"rules find {terminal.winner}/{terminal.reason}"
        )
    return states


def position_key(state: PositionState) -> str:
    """The cache and explorer key of a position."""
    bucket = 0 if state.psc == 0 else (2 if state.psc >= 150 else 1)
    payload = bytes(state.board) + state.to_move.encode("ascii") + bytes([bucket])
    return hashlib.sha1(payload).hexdigest()


def dumps_record(record: GameRecord) -> str:
    """Serialise a game record compactly."""
    return json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"))


def record_id(record: GameRecord) -> str:
    """The content-addressed 12-hex identifier of an imported record."""
    payload = dict(record.to_dict())
    payload["id"] = None
    payload["live"] = False
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def game_to_study(record: GameRecord, title: str | None = None) -> Study:
    """Represent a game as a linear study tree."""
    nodes: dict[int | str, StudyNode] = {}
    parent: int | None = None
    for index, move in enumerate(record.moves, 1):
        nodes[index] = StudyNode(
            parent, MoveRecord.from_dict(move.to_dict()), comment=move.comment, nags=list(move.nags or [])
        )
        parent = index
    return Study(
        list(record.setup or initial_board_list()),
        nodes,
        title or record.tags.get("Event") or record.source_id or "Imported game",
        f"{record.source}_game",
    )
