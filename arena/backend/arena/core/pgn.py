"""Reader/writer for the henhen rps-strategy-pgn/1 interchange format."""

from __future__ import annotations

import re
import textwrap
from collections import OrderedDict
from datetime import UTC, datetime

from .frames import mirror_file
from .game import GameRecord, MoveRecord, PlayerRef, Result, replay
from .notation import format_move, parse_move

TAG_RE = re.compile(r'^\[([^\s]+)\s+"((?:\\.|[^"])*)"\]\s*$', re.MULTILINE)
TOKEN_RE = re.compile(r"\{[^}]*\}|\$\d+|\d+\.{1,3}|1/2-1/2|1-0|0-1|[^\s{}]+")
EMT_RE = re.compile(r"\[%emt\s+([0-9.]+)\]")
CLK_RE = re.compile(r"\[%clk\s+([^\s\]]+)\s+([^\s\]]+)\]")
END_RE = re.compile(r"\[%end\s+([^\s\]]+)\s+([^\]]+)\]", re.IGNORECASE)

TAG_ORDER = [
    "Event",
    "Site",
    "Date",
    "Round",
    "Red",
    "Blue",
    "Result",
    "GameId",
    "Variant",
    "ModeId",
    "BoardSize",
    "TimeControl",
    "SetUp",
    "FEN",
    "RedId",
    "BlueId",
    "RedElo",
    "BlueElo",
    "RedEloAfter",
    "RedRatingDiff",
    "BlueEloAfter",
    "BlueRatingDiff",
    "Ranked",
    "BookPlies",
    "OpeningSeed",
    "SeriesId",
    "Termination",
    "EndReason",
    "PlyCount",
    "MoveNumber",
    "UTCDate",
    "UTCTime",
    "StartTimeUnixMs",
    "EndTimeUnixMs",
    "FinalFEN",
    "Generator",
]

LETTER_TYPE = {"R": 1, "P": 2, "S": 3}


def parse_pieces(field: str) -> list[int]:
    """Parse a FEN piece field into 81 site-frame cells."""
    rows = field.split("/")
    if len(rows) != 9:
        raise ValueError("FEN piece field must have nine rows")
    board = [0] * 81
    for rank, row in enumerate(rows):
        file = 0
        for char in row:
            if char.isdigit():
                file += int(char)
            else:
                piece = LETTER_TYPE.get(char.upper())
                if piece is None or file >= 9:
                    raise ValueError(f"invalid FEN row {row!r}")
                board[rank * 9 + file] = piece if char.isupper() else piece + 3
                file += 1
        if file != 9:
            raise ValueError(f"FEN row covers {file} files")
    return board


def format_pieces(board: list[int]) -> str:
    """Render 81 site-frame cells as a FEN piece field."""
    rows: list[str] = []
    for rank in range(9):
        row, empty = "", 0
        for file in range(9):
            value = board[rank * 9 + file]
            if value == 0:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            letter = "RPS"[(value - 1) % 3]
            row += letter if value <= 3 else letter.lower()
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)


def _absolute_from_site_board(site_board: list[int]) -> list[int]:
    """Mirror a henhen board's files into the absolute frame."""
    absolute = [0] * 81
    for site_sq, value in enumerate(site_board):
        absolute[mirror_file(site_sq)] = value
    return absolute


def parse_fen(fen: str) -> tuple[list[int], str, str | None]:
    """Parse a henhen FEN into an absolute board, side to move and territory."""
    fields = fen.split()
    if len(fields) < 2:
        raise ValueError("henhen FEN needs piece and side-to-move fields")
    board = _absolute_from_site_board(parse_pieces(fields[0]))
    side = "red" if fields[1].lower() == "r" else "blue"
    territory = fields[2] if len(fields) > 2 else None
    return board, side, territory


def _clock_ms(text: str) -> int:
    """Parse a PGN h:mm:ss.mmm clock into milliseconds."""
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid PGN clock {text!r}")
    return round((int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])) * 1000)


def _format_clock(ms: int) -> str:
    """Render milliseconds as a PGN h:mm:ss.mmm clock."""
    hours, rest = divmod(max(0, ms), 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _result_from_tags(tags: dict[str, str]) -> Result:
    """Build a result from the PGN Result and EndReason tags."""
    token = tags.get("Result", "*")
    winner = "red" if token == "1-0" else ("blue" if token == "0-1" else None)
    reason = tags.get("EndReason", "unknown").lower().replace("zone", "corner")
    allowed = {"corner", "no_pieces", "no_moves", "stagnation", "repetition", "resign", "timeout", "agreed", "unknown"}
    return Result(winner, reason if reason in allowed else "unknown", token)


def loads(text: str) -> GameRecord:
    """Parse a henhen PGN into a game record."""
    tags: OrderedDict[str, str] = OrderedDict()
    for match in TAG_RE.finditer(text):
        tags[match.group(1)] = match.group(2).replace('\\"', '"').replace("\\\\", "\\")
    body_start = max((m.end() for m in TAG_RE.finditer(text)), default=0)
    body = text[body_start:]
    moves: list[MoveRecord] = []
    final_comment: str | None = None
    for token in TOKEN_RE.findall(body):
        if token.startswith("{"):
            comment = token[1:-1].strip()
            if moves:
                emt = EMT_RE.search(comment)
                clk = CLK_RE.search(comment)
                end = END_RE.search(comment)
                if emt:
                    moves[-1].emt_ms = round(float(emt.group(1)) * 1000)
                if clk:
                    moves[-1].clock_ms = {"red": _clock_ms(clk.group(1)), "blue": _clock_ms(clk.group(2))}
                if end:
                    final_comment = comment
                residual = EMT_RE.sub("", CLK_RE.sub("", END_RE.sub("", comment))).strip()
                if residual:
                    moves[-1].comment = residual
            continue
        if token.startswith("$"):
            if moves:
                if moves[-1].nags is None:
                    moves[-1].nags = []
                moves[-1].nags.append(token)
            continue
        if token in ("1-0", "0-1", "1/2-1/2", "*") or re.fullmatch(r"\d+\.{1,3}", token):
            continue
        moves.append(parse_move(token, "henhen", "henhen"))
    setup = None
    start_side = "blue"
    if "FEN" in tags:
        setup, start_side, territory = parse_fen(tags["FEN"])
    else:
        territory = None
    tc = None
    if "+" in tags.get("TimeControl", ""):
        initial, increment = tags["TimeControl"].split("+", 1)
        try:
            tc = {"initial_ms": int(float(initial) * 1000), "increment_ms": int(float(increment) * 1000)}
        except ValueError:
            pass

    def rating(name: str) -> float | None:
        try:
            return float(tags[name])
        except (KeyError, ValueError):
            return None

    def timestamp(name: str) -> str | None:
        try:
            return datetime.fromtimestamp(int(tags[name]) / 1000, UTC).isoformat()
        except (KeyError, ValueError, OSError):
            return None

    record = GameRecord(
        source="henhen",
        source_id=tags.get("GameId"),
        source_url=(f"https://rps.henhen1227.com/review?gameId={tags['GameId']}" if tags.get("GameId") else None),
        frame="henhen",
        players={
            "blue": PlayerRef(
                tags.get("Blue", "Blue"),
                tags.get("BlueId"),
                rating("BlueElo"),
                rating("BlueEloAfter"),
                "bot" if tags.get("BlueId", "").startswith("bot-") else "human",
            ),
            "red": PlayerRef(
                tags.get("Red", "Red"),
                tags.get("RedId"),
                rating("RedElo"),
                rating("RedEloAfter"),
                "bot" if tags.get("RedId", "").startswith("bot-") else "human",
            ),
        },
        setup=setup,
        moves=moves,
        result=_result_from_tags(tags),
        time_control=tc,
        tags=dict(tags),
        started_at=timestamp("StartTimeUnixMs"),
        ended_at=timestamp("EndTimeUnixMs"),
        meta={
            "start_to_move": start_side,
            "territory": territory,
            "end_comment": final_comment,
            "series": tags.get("SeriesId"),
            "opening_seed": tags.get("OpeningSeed"),
            "book_plies": tags.get("BookPlies"),
            "tournament": tags.get("Event"),
        },
    )
    # Legality is authoritative for capture flags, including malformed source separators.
    replay(record)
    return record


def _territory(board: list[int]) -> str:
    """Render the territory field of a henhen FEN."""
    cells = ["b" if 1 <= value <= 3 else ("r" if value >= 4 else "") for value in board]
    rows: list[str] = []
    for rank in range(9):
        row, empty = "", 0
        for file in range(9):
            value = cells[rank * 9 + (8 - file)]
            if not value:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += value
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)


def format_fen(board: list[int], to_move: str, territory: str | None = None) -> str:
    """Render an absolute board as a henhen FEN."""
    site_board = [0] * 81
    for absolute_sq, value in enumerate(board):
        site_board[mirror_file(absolute_sq)] = value
    pieces = format_pieces(site_board)
    side = "r" if to_move == "red" else "b"
    return f"{pieces} {side} {territory or _territory(board)}"


def _result_token(result: Result) -> str:
    """Render a result as its PGN token."""
    return "1-0" if result.winner == "red" else ("0-1" if result.winner == "blue" else "1/2-1/2")


def dumps(record: GameRecord) -> str:
    """Render a game record as a henhen PGN."""
    states = replay(record)
    tags = OrderedDict(record.tags)
    tags["Red"] = record.players["red"].name
    tags["Blue"] = record.players["blue"].name
    tags["Result"] = _result_token(record.result)
    tags["PlyCount"] = str(len(record.moves))
    tags["MoveNumber"] = str(len(record.moves))
    tags.setdefault("Generator", "rps-strategy-pgn/1")
    if record.setup is not None:
        tags.setdefault("SetUp", "1")
        tags.setdefault(
            "FEN", format_fen(record.setup, record.meta.get("start_to_move", "blue"), record.meta.get("territory"))
        )
    ordered = [key for key in TAG_ORDER if key in tags]
    ordered.extend(key for key in tags if key not in ordered)
    tag_text = "\n".join(f'[{key} "{tags[key].replace(chr(34), chr(92) + chr(34))}"]' for key in ordered)
    tokens: list[str] = []
    starts_red = record.meta.get("start_to_move") == "red"
    for ply, move in enumerate(record.moves):
        if ply % 2 == 0:
            tokens.append(f"{ply // 2 + 1}.")
        elif starts_red or ply % 2 == 1:
            tokens.append(f"{ply // 2 + 1}...")
        ending = (
            ply == len(record.moves) - 1
            and record.result.winner is not None
            and record.result.reason in {"corner", "no_moves", "no_pieces"}
        )
        tokens.append(format_move(move, states[ply].board, "henhen", "henhen", ending))
        comments: list[str] = []
        if move.emt_ms is not None:
            comments.append(f"[%emt {move.emt_ms / 1000:g}]")
        if move.clock_ms:
            comments.append(f"[%clk {_format_clock(move.clock_ms['red'])} {_format_clock(move.clock_ms['blue'])}]")
        if move.comment:
            comments.append(move.comment)
        if comments:
            tokens.append("{" + " ".join(comments) + "}")
    if record.moves and record.result.reason:
        winner = record.result.winner.title() if record.result.winner else "Draw"
        final_clock = record.moves[-1].clock_ms
        end_parts = [f"[%end {record.result.reason} {winner}]", "[%emt 0]"]
        if final_clock:
            end_parts.append(f"[%clk {_format_clock(final_clock['red'])} {_format_clock(final_clock['blue'])}]")
        tokens.append("{" + " ".join(end_parts) + "}")
    tokens.append(_result_token(record.result))
    wrapped = textwrap.fill(" ".join(tokens), width=80, break_long_words=False, break_on_hyphens=False)
    return f"{tag_text}\n\n{wrapped}\n"
