"""MEAF game-history, line, and compressed workshop codecs."""

from __future__ import annotations

import base64
import json
import re
import zlib
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from .frames import parse_square, square_name
from .game import (
    GameRecord,
    MoveRecord,
    PlayerRef,
    Result,
    Study,
    StudyNode,
    initial_board_list,
    replay,
)
from .notation import format_move, parse_move

LINE_MOVE_RE = re.compile(r"^([a-i][1-9])([-x])([a-i][1-9])$", re.IGNORECASE)
NUMBER_RE = re.compile(r"^(\d+)\.$")
RESULT_RE = re.compile(r"^(?:0-1|0-0|1-0)\.?$")


def _reason(raw: str | None) -> str:
    """Map a MEAF win reason onto the site's rules reasons."""
    text = (raw or "").lower()
    if text == "base capture":
        return "corner"
    if "resign" in text:
        return "resign"
    if "timeout" in text:
        return "timeout"
    if "stagnation" in text:
        return "stagnation"
    if "repetition" in text:
        return "repetition"
    if "draw" in text:
        return "agreed"
    return "unknown"


def game_history(data: str | dict[str, Any]) -> GameRecord:
    """Parse a MEAF game_history payload into a game record."""
    raw = json.loads(data) if isinstance(data, str) else data
    blue_name = raw.get("firstPlayer") or raw.get("playernames", ["Blue"])[0]
    names = list(raw.get("playernames", []))
    red_name = next((name for name in names if name != blue_name), "Red")
    ratings_before = raw.get("ratingsBefore", {})
    ratings_after = raw.get("ratingsAfter", {})
    moves: list[MoveRecord] = []
    for item in raw.get("moveHistory", []):
        token = item["move"]
        if not re.fullmatch(r"[A-I][1-9][A-I][1-9]", token, re.IGNORECASE):
            raise ValueError(f"invalid MEAF history move {token!r}")
        clock = item.get("clocksAfter")
        moves.append(
            MoveRecord(
                parse_square(token[:2], "meaf"),
                parse_square(token[2:], "meaf"),
                clock_ms=(
                    {"blue": round(float(clock[blue_name]) * 1000), "red": round(float(clock[red_name]) * 1000)}
                    if clock
                    else None
                ),
            )
        )
    winner_name = raw.get("winner")
    result = Result(
        "blue" if winner_name == blue_name else ("red" if winner_name == red_name else None),
        _reason(raw.get("winReason")),
        raw.get("winReason") or "",
    )
    options = raw.get("options", {})
    tc = None
    if options.get("timeControlEnabled", "initialTimeMinutes" in options):
        tc = {
            "initial_ms": round(float(options.get("initialTimeMinutes", 0)) * 60_000),
            "increment_ms": round(float(options.get("incrementSeconds", 0)) * 1000),
        }
    record = GameRecord(
        source="meaf",
        source_id=raw.get("gameID"),
        source_url=(f"https://meaf.us/rps2/?gameID={raw['gameID']}" if raw.get("gameID") else None),
        frame="meaf",
        players={
            "blue": PlayerRef(
                blue_name, rating_before=ratings_before.get(blue_name), rating_after=ratings_after.get(blue_name)
            ),
            "red": PlayerRef(
                red_name, rating_before=ratings_before.get(red_name), rating_after=ratings_after.get(red_name)
            ),
        },
        moves=moves,
        result=result,
        time_control=tc,
        started_at=raw.get("startTime"),
        ended_at=raw.get("endTime"),
        tags={},
        meta={"tournament": raw.get("tournamentName"), "meaf_raw": raw},
    )
    replay(record)
    return record


def parse_line(text: str) -> tuple[list[MoveRecord], int, Result | None]:
    """Parse a MEAF text line into moves, a leading-ply offset and a result."""
    tokens = text.split()
    if tokens and tokens[-1] == ".":
        tokens.pop()
    if not tokens:
        return [], 0, None
    offset = 0 if NUMBER_RE.fullmatch(tokens[0]) else 1
    moves: list[MoveRecord] = []
    result: Result | None = None
    index = 0
    expected_number = 1
    if offset:
        if not LINE_MOVE_RE.fullmatch(tokens[0]):
            raise ValueError(f"invalid MEAF line token {tokens[0]!r}")
        moves.append(parse_move(tokens[0], "meaf", "meaf"))
        index = 1
        expected_number = 2
    while index < len(tokens):
        token = tokens[index]
        if RESULT_RE.fullmatch(token):
            normalized = token.rstrip(".")
            result = Result(
                "red" if normalized == "0-1" else ("blue" if normalized == "1-0" else None),
                "unknown" if normalized != "0-0" else "agreed",
                normalized,
            )
            if index != len(tokens) - 1:
                raise ValueError("MEAF result token must end the line")
            break
        numbered = NUMBER_RE.fullmatch(token)
        if not numbered or int(numbered.group(1)) != expected_number:
            raise ValueError(f"expected move number {expected_number}., got {token!r}")
        index += 1
        if index >= len(tokens) or not LINE_MOVE_RE.fullmatch(tokens[index]):
            raise ValueError(f"move {expected_number}. has no Blue half-move")
        moves.append(parse_move(tokens[index], "meaf", "meaf"))
        index += 1
        if index < len(tokens) and LINE_MOVE_RE.fullmatch(tokens[index]):
            moves.append(parse_move(tokens[index], "meaf", "meaf"))
            index += 1
        expected_number += 1
    return moves, offset, result


def format_line(
    value: GameRecord | list[MoveRecord], leading_ply_offset: int | None = None, include_result: bool = False
) -> str:
    """Render moves or a game record as a MEAF text line."""
    if isinstance(value, GameRecord):
        moves = value.moves
        offset = int(value.meta.get("leading_ply_offset", 0) if leading_ply_offset is None else leading_ply_offset)
        result = value.result
        board = list(value.setup or initial_board_list())
    else:
        moves, offset, result = value, int(leading_ply_offset or 0), None
        board = list(initial_board_list())
    tokens: list[str] = []
    for index, move in enumerate(moves):
        absolute_ply = offset + index
        if absolute_ply % 2 == 0:
            tokens.append(f"{absolute_ply // 2 + 1}.")
        tokens.append(format_move(move, board, "meaf", "meaf"))
        board[move.to], board[move.from_] = board[move.from_], 0
    if include_result and result:
        tokens.append("1-0" if result.winner == "blue" else ("0-1" if result.winner == "red" else "0-0"))
    return " ".join(tokens)


def _decoded_payload(value: str | dict[str, Any]) -> dict[str, Any]:
    """Decode a workshop link, blob or already-decoded payload."""
    if isinstance(value, dict):
        if value.get("v") == 1 and "nodes" in value:
            return value
        if "decoded" in value:
            return value["decoded"]
        value = value.get("data") or value.get("data_b64") or ""
    text = value.strip()
    if text.startswith("http"):
        query = parse_qs(urlparse(text).query)
        text = query.get("workshop", [""])[0]
    padding = "=" * (-len(text) % 4)
    packed = base64.urlsafe_b64decode(text + padding)
    try:
        unpacked = zlib.decompress(packed)
    except zlib.error:
        unpacked = zlib.decompress(packed, -zlib.MAX_WBITS)
    return json.loads(unpacked)


def workshop(value: str | dict[str, Any]) -> Study:
    """Decode a MEAF workshop payload into a study."""
    data = _decoded_payload(value)
    setup = [0] * 81
    for square, side, letter in data.get("setup", []):
        cell = "RPS".index(letter.upper()) + 1
        setup[parse_square(square)] = cell + (3 if side == "red" else 0)
    nodes: dict[int, StudyNode] = {}
    for node_id, parent, origin, target, _side, label in data.get("nodes", []):
        nodes[int(node_id)] = StudyNode(
            parent, MoveRecord(parse_square(origin), parse_square(target)), comment=label or None, label=label or None
        )
    meta = {
        "workshop": {key: data.get(key) for key in ("v", "rootLabel", "cursor", "nextId", "collapsed", "rootCollapsed")}
    }
    return Study(setup, nodes, data.get("title", "Untitled study"), "meaf_workshop", meta=meta)


def workshop_json(study: Study) -> dict[str, Any]:
    """Render a study back into the workshop payload shape."""
    preserved = dict(study.meta.get("workshop", {}))
    data: dict[str, Any] = {
        "v": preserved.get("v", 1),
        "title": study.title,
        "rootLabel": preserved.get("rootLabel"),
        "setup": [],
        "cursor": preserved.get("cursor"),
        "nextId": preserved.get("nextId", max([int(k) for k in study.nodes] + [0]) + 1),
        "nodes": [],
        "collapsed": preserved.get("collapsed", []),
        "rootCollapsed": preserved.get("rootCollapsed", False),
    }
    for sq, cell in enumerate(study.root_setup):
        if cell:
            data["setup"].append([square_name(sq), "blue" if cell <= 3 else "red", "RPS"[(cell - 1) % 3]])
    for node_id, node in study.nodes.items():
        if node.move is None:
            continue
        # Workshop stores the mover redundantly; tree depth determines it.
        depth, parent = 0, node.parent
        while parent is not None and parent in study.nodes:
            depth += 1
            parent = study.nodes[parent].parent
        side = "blue" if depth % 2 == 0 else "red"
        data["nodes"].append(
            [
                int(node_id),
                node.parent,
                square_name(node.move.from_),
                square_name(node.move.to),
                side,
                node.label or node.comment or "",
            ]
        )
    return data


def encode_workshop(study: Study) -> tuple[str, dict[str, Any]]:
    """Encode a study as a workshop link plus its raw payload."""
    raw = workshop_json(study)
    packed = zlib.compress(json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode())
    encoded = base64.b64encode(packed).decode().rstrip("=")
    return f"https://meaf.us/rps2/?workshop={quote(encoded, safe='')}", raw
