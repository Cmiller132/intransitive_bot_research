"""Sparse features of a position (DESIGN item 5) and the board symmetries.

A position is a board of 81 cell codes in the mover's frame (0 empty, 1-3 own
rock/paper/scissors, 4-6 enemy), the plies since the last capture and the
game's capture clock. Both perspectives (the mover; the opponent after the
anti-diagonal reflection with colours swapped) index one feature table whose
row layout is the file version's (`Layout`):

- piece-square rows `486 c + 81 (code - 1) + square`, where `c` is the
  perspective's opponent-material context: format 8 has 27 contexts,
  `min(r, 2) + 3 min(p, 2) + 9 min(s, 2)` over the opponent's rock, paper and
  scissors counts in that perspective's frame; format 6 has the one context 0;
- attacked-piece rows `attack_base + 81 (code - 1) + square`: the piece on
  the square is attacked by an adjacent enemy that beats it;
- 16 elapsed-clock rows (bucket of `since_capture`) and 16 remaining-clock
  rows (bucket of `clock - since_capture`), shared by both perspectives;
- from format 9 on, eight goal-corner rows: four for the occupant of the
  perspective's own goal (square 80 in its frame: empty, or an opponent
  blocker) and four for the occupant of the opponent's goal (square 0:
  empty, or one of its own pieces), one of each active per perspective;
- one padding row after the last feature.

Every perspective has 42 slots (44 from format 9 on): 20 pieces, 20 attacked
pieces, the two clock rows and the two goal rows. `feature_ids` encodes
format 8 and `Layout.rows` maps its ids onto a format-6 or format-9 table, so
one id cache serves every version; `feature_ids9` appends the goal rows, which
need the board. The Rust crate indexes the same rows
(runs/nnue_plan/format8_contract.md and models/nnue/FORMAT9.md are the shared
contracts); `tests/test_features.py` checks the encoder against a slow
enumeration and the perspective exchange.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass

import numpy as np

PIECE_ROWS = 6 * 81
CONTEXTS = 27
CLOCK_BOUNDS = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160], dtype=np.int64)
CLOCK_BUCKETS = len(CLOCK_BOUNDS)
MAX_PIECES = 20
SLOTS = 2 * MAX_PIECES + 2
GOAL_STATES = 4  # a goal corner is empty or holds one of the three piece types
GOAL_GROUPS = 2  # the perspective's own goal (square 80) and the opponent's (square 0)
GOAL_ROWS = GOAL_GROUPS * GOAL_STATES
HEADS = 8  # version 9 output buckets, chosen by the pieces on the board


@dataclass(frozen=True)
class Layout:
    """The row bases of one file version's feature table, its feature slots
    per perspective and its number of output heads."""

    version: int
    contexts: int
    goal: bool = False
    heads: int = 1

    @property
    def attack_base(self) -> int:
        return self.contexts * PIECE_ROWS

    @property
    def elapsed_base(self) -> int:
        return self.attack_base + PIECE_ROWS

    @property
    def remaining_base(self) -> int:
        return self.elapsed_base + CLOCK_BUCKETS

    @property
    def goal_base(self) -> int:
        """Where the goal rows begin; the end of the table without them."""
        return self.remaining_base + CLOCK_BUCKETS

    @property
    def features(self) -> int:
        return self.goal_base + (GOAL_ROWS if self.goal else 0)

    @property
    def slots(self) -> int:
        return SLOTS + (GOAL_GROUPS if self.goal else 0)

    @property
    def pad(self) -> int:
        return self.features

    def rows(self, ids: np.ndarray) -> np.ndarray:
        """This layout's ids from format-8 ids (`feature_ids`): a format-6
        table drops the context and shifts the attacked, clock and padding rows;
        a format-9 table keeps every shared row and only moves the padding,
        since its goal rows come after them (`feature_ids9` appends those)."""
        if self.contexts == CONTEXTS and not self.goal:
            return ids
        ids = np.asarray(ids)
        if self.goal:
            return np.where(ids == FORMAT8.pad, self.pad, ids)
        return np.where(ids < FORMAT8.attack_base, ids % PIECE_ROWS, ids - (FORMAT8.attack_base - self.attack_base))


FORMAT6 = Layout(6, 1)
FORMAT8 = Layout(8, CONTEXTS)
FORMAT9 = Layout(9, CONTEXTS, goal=True, heads=HEADS)
LAYOUTS = {6: FORMAT6, 8: FORMAT8, 9: FORMAT9}

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


def attacked(board: np.ndarray) -> np.ndarray:
    """(N, 81) bool: the piece on the square is attacked by an adjacent enemy that beats it."""
    piece = np.asarray(board, dtype=np.int16).reshape(-1, 81)
    neighbours = np.pad(piece, ((0, 0), (0, 1)))[:, NEIGHBOURS]
    enemy = ((neighbours - 1) // 3 != (piece[:, :, None] - 1) // 3) & (neighbours > 0)
    beats = ((neighbours - 1) % 3 - (piece[:, :, None] - 1) % 3) % 3 == 1
    return (piece > 0) & np.any(enemy & beats, axis=-1)


def perspectives(board: np.ndarray) -> np.ndarray:
    """(2N, 81): every board in the mover's frame followed by its opponent's frame."""
    board = np.asarray(board, dtype=np.uint8).reshape(-1, 81)
    return np.stack((board, SWAP[board[:, ANTI]]), axis=1).reshape(-1, 81)


def context(board: np.ndarray) -> np.ndarray:
    """(N, 2) int64 opponent-material context of both perspectives: the
    capped rock, paper and scissors counts of the enemy pieces in each
    perspective's own frame (the mover's perspective counts codes 4-6 of
    the board, the opponent's perspective codes 1-3)."""
    frames = perspectives(board)
    capped = np.stack([np.minimum(np.count_nonzero(frames == code, axis=1), 2) for code in (4, 5, 6)], axis=1)
    return (capped @ np.array([1, 3, 9])).reshape(-1, 2)


def feature_ids(board: np.ndarray, since_capture: np.ndarray, clock: np.ndarray) -> np.ndarray:
    """(N, 2, 42) int64 format-8 feature ids of every position, padded with
    `FORMAT8.pad`; `Layout.rows` maps them onto a format-6 table."""
    board = np.asarray(board, dtype=np.uint8).reshape(-1, 81)
    if np.any(board > 6):
        raise ValueError("cell codes must be 0..6")
    since = np.asarray(since_capture, dtype=np.int64).reshape(-1)
    clock = np.asarray(clock, dtype=np.int64).reshape(-1)
    if len(since) != len(board) or len(clock) != len(board):
        raise ValueError("one since_capture and one clock per board")
    if np.any(clock <= 0) or np.any(since < 0):
        raise ValueError("every position needs a positive clock and a non-negative since_capture")
    both = perspectives(board)
    row, sq = np.nonzero(both)
    count = np.count_nonzero(both, axis=1)
    if count.max(initial=0) > MAX_PIECES:
        raise ValueError("a position holds at most the original 20 pieces")
    ids = np.full((len(both), SLOTS), FORMAT8.pad, dtype=np.int64)
    col = np.arange(len(row)) - np.repeat(np.cumsum(count) - count, count)
    ids[row, col] = PIECE_ROWS * context(board).reshape(-1)[row] + (both[row, sq].astype(np.int64) - 1) * 81 + sq
    hit = attacked(both)
    tr, ts = np.nonzero(hit)
    tc = np.count_nonzero(hit, axis=1)
    tcol = np.arange(len(tr)) - np.repeat(np.cumsum(tc) - tc, tc)
    ids[tr, MAX_PIECES + tcol] = FORMAT8.attack_base + (both[tr, ts].astype(np.int64) - 1) * 81 + ts
    ids = ids.reshape(-1, 2, SLOTS)
    ids[:, :, 2 * MAX_PIECES] = (FORMAT8.elapsed_base + clock_bucket(since))[:, None]
    ids[:, :, 2 * MAX_PIECES + 1] = (FORMAT8.remaining_base + clock_bucket(clock - since))[:, None]
    return ids


def goal_ids(board: np.ndarray) -> np.ndarray:
    """(N, 2, 2) int64 format-9 goal rows of both perspectives: the occupant of
    the perspective's own goal (square 80 in its frame, where only an opponent
    piece can stand without the game being over) and of the opponent's goal
    (square 0, where only one of its own pieces can stand). Anything else on
    those squares is an already-decided position and reads as empty."""
    frames = perspectives(board).reshape(-1, 2, 81)
    own = frames[:, :, 80].astype(np.int64)
    enemy = frames[:, :, 0].astype(np.int64)
    blocker = np.where((own >= 4) & (own <= 6), own - 3, 0)
    invader = np.where((enemy >= 1) & (enemy <= 3), enemy, 0)
    return np.stack([FORMAT9.goal_base + blocker, FORMAT9.goal_base + GOAL_STATES + invader], axis=2)


def feature_ids9(board: np.ndarray, since_capture: np.ndarray, clock: np.ndarray) -> np.ndarray:
    """(N, 2, 44) int64 format-9 feature ids: the 42 shared slots with the
    padding row moved to `FORMAT9.pad`, then the two goal rows."""
    ids = FORMAT9.rows(feature_ids(board, since_capture, clock))
    return np.concatenate([ids, goal_ids(board)], axis=2)


def piece_bucket(pieces: np.ndarray, heads: int = HEADS) -> np.ndarray:
    """The output head of a position with `pieces` pieces on the board:
    `min(heads - 1, (pieces - 2) * heads // 19)` over the totals 2..20."""
    pieces = np.asarray(pieces, dtype=np.int64)
    return np.minimum(np.maximum(pieces - 2, 0) * heads // 19, heads - 1)


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


def symmetry_table(layout: Layout = FORMAT8) -> np.ndarray:
    """(6, features + 1) permutation of the layout's feature ids under every
    symmetry; it commutes with both perspectives, so `table[k, ids]` are the
    ids of the transformed position. A type renaming also renames the
    context's counts and a goal corner's occupant; the diagonal reflection
    fixes both corners; clock and padding rows map to themselves."""
    table = np.tile(np.arange(layout.features + 1), (6, 1))
    f = np.arange(layout.elapsed_base)
    base = np.where(f < layout.attack_base, 0, layout.attack_base)
    c, rest = divmod(f - base, PIECE_ROWS)
    piece, square = divmod(rest, 81)
    side, kind = divmod(piece, 3)
    counts = np.stack([c % 3, c // 3 % 3, c // 9])
    states = np.arange(GOAL_STATES)
    for k in range(6):
        m = k % 3
        sq = DIAG[square] if k // 3 else square
        renamed = counts[(0 - m) % 3] + 3 * counts[(1 - m) % 3] + 9 * counts[(2 - m) % 3]
        table[k, f] = base + PIECE_ROWS * renamed + (side * 3 + (kind + m) % 3) * 81 + sq
        if layout.goal:
            for group in range(GOAL_GROUPS):
                at = layout.goal_base + GOAL_STATES * group
                table[k, at + states] = at + np.where(states == 0, 0, (states - 1 + m) % 3 + 1)
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


@functools.cache
def signature() -> str:
    """A hash of the encoder's ids on a fixed set of boards; an id cache
    carries it, so a changed encoder invalidates every cache."""
    rng = np.random.default_rng(20260912)
    boards = np.zeros((64, 81), dtype=np.uint8)
    for board in boards:
        pieces = int(rng.integers(1, MAX_PIECES + 1))
        board[rng.choice(81, pieces, replace=False)] = rng.integers(1, 7, pieces)
    clock = rng.integers(1, 201, len(boards))
    since = (rng.random(len(boards)) * clock).astype(np.int64)
    return hashlib.sha256(feature_ids(boards, since, clock).tobytes()).hexdigest()[:16]
