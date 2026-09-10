"""Game records as the engine sees them: absolute move tokens (`b4-a4`, Blue's
home a1, Blue first) replayed into canonical mover-frame roots, and the
site's compact export (10 bits per move: origin square times 8 plus the
direction index, base64url over the packed big-endian bits; the layout is
the site owner's `decodegames.py`)."""

from __future__ import annotations

import base64

import numpy as np

from . import features

SITE_CLOCK = 200
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
FILES = "abcdefghi"


def parse_action(token: str, red_to_move: bool) -> int:
    """An absolute move token as the mover's canonical action."""
    text = token.lower().lstrip("rps")
    from_name, to_name = (text[:2], text[3:]) if len(text) == 5 else (text[:2], text[2:])
    squares = []
    for name in (from_name, to_name):
        square = (ord(name[1]) - ord("1")) * 9 + ord(name[0]) - ord("a")
        squares.append(int(features.ANTI[square]) if red_to_move else square)
    start, end = squares
    delta = (end // 9 - start // 9, end % 9 - start % 9)
    return DIRS.index(delta) * 81 + start


def replay(moves: list[str], clock: int = SITE_CLOCK) -> tuple[list[tuple[np.ndarray, int, int, int]], int]:
    """Every root of a game as (board, since_capture, ply, action played) in
    the mover's frame, and the engine's outcome after the last move (0
    ongoing, 1 the last mover won, 2 draw)."""
    import engine

    board, since, ply, outcome = list(engine.initial_board()), 0, 0, 0
    roots = []
    for token in moves:
        action = parse_action(token, red_to_move=ply % 2 == 1)
        roots.append((np.array(board, dtype=np.uint8), since, ply, action))
        child, since, ply, outcome = engine.apply(board, since, ply, action, clock)
        board = list(child)
        if outcome != 0:
            break
    return roots, outcome


def decode_export(encoded: str) -> list[str]:
    """The site's packed move string as absolute tokens."""
    data = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    bits = len(data) * 8
    packed = int.from_bytes(data, "big")
    moves = []
    for index in range(bits // 10):
        value = (packed >> (bits - (index + 1) * 10)) & 0x3FF
        origin, direction = value >> 3, value & 7
        rank, file = divmod(origin, 9)
        d_file, d_rank = DIRS[direction]
        to_file, to_rank = file + d_file, rank + d_rank
        if origin > 80 or not (0 <= to_file < 9 and 0 <= to_rank < 9):
            raise ValueError(f"invalid packed move at index {index}")
        moves.append(f"{FILES[file]}{rank + 1}-{FILES[to_file]}{to_rank + 1}")
    return moves
