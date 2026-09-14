"""Parsing and display of source-specific move dialects."""

from __future__ import annotations

import re

from .frames import Frame, parse_square, square_name
from .game import MoveRecord

MOVE_RE = re.compile(r"^(?:[RPS])?([A-I][1-9])([-X])([A-I][1-9])#?$", re.IGNORECASE)


def parse_move(text: str, dialect: str = "meaf", frame: Frame | None = None) -> MoveRecord:
    """Parse one move token written in a source's dialect."""
    if dialect not in ("meaf", "henhen"):
        raise ValueError(f"unknown notation dialect {dialect!r}")
    actual_frame: Frame = frame or ("henhen" if dialect == "henhen" else "meaf")
    stripped = text.strip()
    if dialect == "meaf" and stripped.endswith("#"):
        raise ValueError("MEAF line moves do not use the henhen # suffix")
    match = MOVE_RE.fullmatch(stripped)
    if not match:
        raise ValueError(f"invalid {dialect} move {text!r}")
    return MoveRecord(
        parse_square(match.group(1), actual_frame),
        parse_square(match.group(3), actual_frame),
        match.group(2).lower() == "x",
    )


def format_move(
    move: MoveRecord,
    board: list[int] | None = None,
    dialect: str = "meaf",
    frame: Frame | None = None,
    game_ending: bool = False,
) -> str:
    """Render one move in a source's dialect."""
    if dialect not in ("meaf", "henhen"):
        raise ValueError(f"unknown notation dialect {dialect!r}")
    actual_frame: Frame = frame or ("henhen" if dialect == "henhen" else "meaf")
    capture = move.capture or (board is not None and board[move.to] != 0)
    separator = "x" if capture else "-"
    origin = square_name(move.from_, actual_frame)
    target = square_name(move.to, actual_frame)
    if dialect == "meaf":
        return f"{origin.upper()}{separator}{target.upper()}"
    letter = ""
    if board is not None:
        cell = board[move.from_]
        if cell:
            letter = "RPS"[(cell - 1) % 3]
    return f"{letter}{origin}{separator}{target}{'#' if game_ending else ''}"


def move_token(move: MoveRecord | dict[str, int] | tuple[int, int]) -> str:
    """Render a move as the bot binary's absolute `a1-b2` token."""
    if isinstance(move, MoveRecord):
        origin, target = move.from_, move.to
    elif isinstance(move, dict):
        origin, target = int(move["from"]), int(move["to"])
    else:
        origin, target = move
    return f"{square_name(origin)}-{square_name(target)}"


def parse_token(token: str) -> tuple[int, int]:
    """Parse a bot `a1-b2` token into absolute (from, to) squares."""
    text = token.strip().lower()
    if len(text) != 5 or text[2] != "-":
        raise ValueError(f"invalid move token {token!r}")
    return parse_square(text[:2]), parse_square(text[3:])
