"""The game's diagonal and simultaneous R/P/S relabelling symmetries."""

from __future__ import annotations

from .game import PositionState, position_key


def cycle_cell(value: int, cycle: int) -> int:
    """Relabel one cell's piece type by a rotation of rock/paper/scissors."""
    if value == 0:
        return 0
    side, piece = divmod(value - 1, 3)
    return side * 3 + (piece + cycle) % 3 + 1


def transform_square(sq: int, diagonal: bool) -> int:
    """Reflect a square across the main diagonal when asked."""
    if not diagonal:
        return sq
    rank, file = divmod(sq, 9)
    return file * 9 + rank


def transform_board(board: list[int], diagonal: bool = False, cycle: int = 0) -> list[int]:
    """Apply a diagonal reflection and a piece-type rotation to a board."""
    out = [0] * 81
    for sq, cell in enumerate(board):
        out[transform_square(sq, diagonal)] = cycle_cell(cell, cycle % 3)
    return out


def images(state: PositionState) -> list[PositionState]:
    """The six symmetric images of a position."""
    return [
        PositionState(
            transform_board(state.board, diagonal, cycle),
            state.to_move,
            state.ply,
            state.psc,
            state.repetition,
            state.result,
        )
        for diagonal in (False, True)
        for cycle in range(3)
    ]


def symmetry_key(state: PositionState) -> str:
    """The smallest position key among a position's symmetric images."""
    return min(position_key(image) for image in images(state))
