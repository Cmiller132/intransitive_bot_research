"""Sparse features of a position (DESIGN item 5) and the board symmetries.

A position is a board of 81 cell codes in the mover's frame (0 empty, 1-3 own
rock/paper/scissors, 4-6 enemy), the plies since the last capture and the
game's capture clock. Both perspectives (the mover; the opponent after the
anti-diagonal reflection with colours swapped) index the same feature table:

- 0-485: piece-square, `(piece - 1) * 81 + square`;
- 486-971: the piece on the square is attacked by an adjacent enemy that
  beats it, `486 + (piece - 1) * 81 + square`;
- 972-987: elapsed-clock bucket of `since_capture`;
- 988-1003: remaining-clock bucket of `clock - since_capture`;
- 1004: padding (an all-zero row).

Every perspective has 42 slots: 20 pieces, 20 attacked pieces, 2 clock rows.
The Rust crate indexes the same rows; `tests/test_features.py` checks the
encoder against a slow enumeration and the perspective exchange.
"""

from __future__ import annotations

import numpy as np

PIECE_ROWS = 6 * 81
ATTACK_BASE = PIECE_ROWS
CLOCK_BOUNDS = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160], dtype=np.int64)
CLOCK_BUCKETS = len(CLOCK_BOUNDS)
ELAPSED_BASE = 2 * PIECE_ROWS
REMAINING_BASE = ELAPSED_BASE + CLOCK_BUCKETS
FEATURES = REMAINING_BASE + CLOCK_BUCKETS
PAD = FEATURES
MAX_PIECES = 20
SLOTS = 2 * MAX_PIECES + 2

# Bucket boundaries of the output heads by total pieces (DESIGN item 7).
BUCKET_BOUNDS = np.array([0, 5, 9, 13], dtype=np.int64)
BUCKETS = len(BUCKET_BOUNDS)

# Anti-diagonal reflection (the opponent's frame), the diagonal reflection
# (a board symmetry) and the colour swap of cell codes.
ANTI = np.array([(8 - s % 9) * 9 + 8 - s // 9 for s in range(81)])
DIAG = np.array([s % 9 * 9 + s // 9 for s in range(81)])
SWAP = np.array([0, 4, 5, 6, 1, 2, 3], dtype=np.uint8)
NEIGHBOURS = np.array(
    [
        [
            ((s // 9 + dr) * 9 + s % 9 + dc) if 0 <= s // 9 + dr < 9 and 0 <= s % 9 + dc < 9 else 81
            for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        ]
        for s in range(81)
    ]
)


def clock_bucket(plies: np.ndarray) -> np.ndarray:
    """Index of the clock bucket holding `plies` (clipped at zero)."""
    return np.searchsorted(CLOCK_BOUNDS, np.maximum(np.asarray(plies, dtype=np.int64), 0), side="right") - 1


def piece_bucket(board: np.ndarray) -> np.ndarray:
    """Output bucket of every board by its total piece count."""
    pieces = np.count_nonzero(np.asarray(board).reshape(-1, 81), axis=1)
    return np.searchsorted(BUCKET_BOUNDS, pieces, side="right") - 1


def attacked(board: np.ndarray) -> np.ndarray:
    """(N, 81) bool: the piece on the square is attacked by an adjacent enemy that beats it."""
    piece = np.asarray(board, dtype=np.int16).reshape(-1, 81)
    neighbours = np.pad(piece, ((0, 0), (0, 1)))[:, NEIGHBOURS]
    enemy = ((neighbours - 1) // 3 != (piece[:, :, None] - 1) // 3) & (neighbours > 0)
    beats = ((neighbours - 1) % 3 - (piece[:, :, None] - 1) % 3) % 3 == 1
    return (piece > 0) & np.any(enemy & beats, axis=-1)


def feature_ids(board: np.ndarray, since_capture: np.ndarray, clock: np.ndarray) -> np.ndarray:
    """(N, 2, 42) int64 feature ids of every position, padded with `PAD`."""
    board = np.asarray(board, dtype=np.uint8).reshape(-1, 81)
    if np.any(board > 6):
        raise ValueError("cell codes must be 0..6")
    since = np.asarray(since_capture, dtype=np.int64).reshape(-1)
    clock = np.asarray(clock, dtype=np.int64).reshape(-1)
    if len(since) != len(board) or len(clock) != len(board):
        raise ValueError("one since_capture and one clock per board")
    if np.any(clock <= 0) or np.any(since < 0):
        raise ValueError("every position needs a positive clock and a non-negative since_capture")
    both = np.stack((board, SWAP[board[:, ANTI]]), axis=1).reshape(-1, 81)
    row, sq = np.nonzero(both)
    count = np.count_nonzero(both, axis=1)
    if count.max(initial=0) > MAX_PIECES:
        raise ValueError("a position holds at most the original 20 pieces")
    ids = np.full((len(both), SLOTS), PAD, dtype=np.int64)
    col = np.arange(len(row)) - np.repeat(np.cumsum(count) - count, count)
    ids[row, col] = (both[row, sq].astype(np.int64) - 1) * 81 + sq
    hit = attacked(both)
    tr, ts = np.nonzero(hit)
    tc = np.count_nonzero(hit, axis=1)
    tcol = np.arange(len(tr)) - np.repeat(np.cumsum(tc) - tc, tc)
    ids[tr, MAX_PIECES + tcol] = ATTACK_BASE + (both[tr, ts].astype(np.int64) - 1) * 81 + ts
    ids = ids.reshape(-1, 2, SLOTS)
    ids[:, :, 2 * MAX_PIECES] = (ELAPSED_BASE + clock_bucket(since))[:, None]
    ids[:, :, 2 * MAX_PIECES + 1] = (REMAINING_BASE + clock_bucket(clock - since))[:, None]
    return ids


def transform_board(board: np.ndarray, symmetry: int) -> np.ndarray:
    """Board under symmetry `k` in 0..5: `k // 3` reflects across the diagonal,
    `k % 3` renames the piece types cyclically. Every such board is played
    exactly like the original."""
    b = np.asarray(board, dtype=np.uint8).reshape(-1, 81)
    if symmetry // 3:
        b = b[:, DIAG]
    if symmetry % 3:
        occupied = b > 0
        code = b.astype(np.int16) - 1
        b = np.where(occupied, code // 3 * 3 + (code % 3 + symmetry % 3) % 3 + 1, 0).astype(np.uint8)
    return b


def symmetry_table() -> np.ndarray:
    """(6, FEATURES + 1) permutation of feature ids under every symmetry; it
    commutes with both perspectives, so `table[k, ids]` are the ids of the
    transformed position. Clock rows and the padding row map to themselves."""
    table = np.tile(np.arange(FEATURES + 1), (6, 1))
    f = np.arange(2 * PIECE_ROWS)
    group, rest = divmod(f, PIECE_ROWS)
    piece, square = divmod(rest, 81)
    side, kind = divmod(piece, 3)
    for k in range(6):
        sq = DIAG[square] if k // 3 else square
        table[k, f] = group * PIECE_ROWS + (side * 3 + (kind + k % 3) % 3) * 81 + sq
    return table


def orbit_hash(board: np.ndarray, seed: int = 20260909) -> np.ndarray:
    """A 64-bit key equal for every board of one symmetry orbit (diagonal
    reflection and the three type renamings); groups boards so that no orbit
    crosses a split. Collisions only merge groups, never leak an orbit."""
    rng = np.random.default_rng(seed)
    table = rng.integers(0, np.iinfo(np.uint64).max, size=(81, 7), dtype=np.uint64)
    board = np.asarray(board, dtype=np.uint8).reshape(-1, 81)
    squares = np.arange(81)
    low = np.full(len(board), np.iinfo(np.uint64).max, dtype=np.uint64)
    for k in range(6):
        b = transform_board(board, k)
        low = np.minimum(low, np.bitwise_xor.reduce(table[squares, b], axis=1))
    z = low + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))
