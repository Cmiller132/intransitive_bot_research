"""The feature encoder against a slow enumeration, the perspective exchange,
the symmetry table and the clock buckets."""

import numpy as np
import pytest

from nnue.features import (
    ANTI,
    ATTACK_BASE,
    CLOCK_BOUNDS,
    ELAPSED_BASE,
    FEATURES,
    MAX_PIECES,
    PAD,
    RACE_BASE,
    RACE_BUCKETS,
    REMAINING_BASE,
    SLOTS,
    SWAP,
    clock_bucket,
    feature_ids,
    orbit_hash,
    piece_bucket,
    race_buckets,
    symmetry_table,
    transform_board,
)


def random_boards(seed, count, low=0, high=21):
    rng = np.random.default_rng(seed)
    boards = np.zeros((count, 81), dtype=np.uint8)
    for board in boards:
        pieces = int(rng.integers(low, high))
        board[rng.choice(81, pieces, replace=False)] = rng.integers(1, 7, pieces)
    return boards


def slow_ids(board, since, clock, race):
    """One perspective at a time by plain loops; `race` is the engine's
    (mover, opponent) pair, swapped for the opponent's perspective."""
    perspectives = [list(map(int, board))]
    reverse = []
    for square in range(81):
        rank, file = divmod(square, 9)
        piece = int(board[(8 - file) * 9 + 8 - rank])
        reverse.append(0 if piece == 0 else piece + 3 if piece <= 3 else piece - 3)
    perspectives.append(reverse)
    out = []
    for own, position in enumerate(perspectives):
        pieces, hit = [], []
        for square, piece in enumerate(position):
            if not piece:
                continue
            pieces.append((piece - 1) * 81 + square)
            rank, file = divmod(square, 9)
            threatened = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr, dc) == (0, 0):
                        continue
                    rr, ff = rank + dr, file + dc
                    if not (0 <= rr < 9 and 0 <= ff < 9):
                        continue
                    enemy = position[rr * 9 + ff]
                    if enemy and (enemy <= 3) != (piece <= 3):
                        threatened |= ((enemy - 1) % 3 - (piece - 1) % 3) % 3 == 1
            if threatened:
                hit.append(ATTACK_BASE + (piece - 1) * 81 + square)
        elapsed = int(np.searchsorted(CLOCK_BOUNDS, since, side="right") - 1)
        remaining = int(np.searchsorted(CLOCK_BOUNDS, clock - since, side="right") - 1)
        slots = pieces + [PAD] * (MAX_PIECES - len(pieces)) + hit + [PAD] * (MAX_PIECES - len(hit))
        mine, theirs = (race[0], race[1]) if own == 0 else (race[1], race[0])
        out.append(slots + [ELAPSED_BASE + elapsed, REMAINING_BASE + remaining])
        out[-1] += [RACE_BASE + int(mine), RACE_BASE + RACE_BUCKETS + int(theirs)]
    return np.array(out, dtype=np.int64)


def test_encoder_matches_the_slow_enumeration():
    boards = random_boards(20260909, 64)
    since = np.random.default_rng(1).integers(0, 200, 64)
    clock = np.full(64, 200)
    got = feature_ids(boards, since, clock)
    assert got.shape == (64, 2, SLOTS)
    race = race_buckets(boards)
    assert race.shape == (64, 2) and race.max() <= 8
    want = np.stack([slow_ids(b, int(s), 200, r) for b, s, r in zip(boards, since, race, strict=True)])
    assert np.array_equal(got, want)


def test_perspectives_exchange_under_the_frame_flip():
    boards = random_boards(7, 32)
    since, clock = np.full(32, 17), np.full(32, 120)
    race = race_buckets(boards)
    ids = feature_ids(boards, since, clock, race)
    # The race pair is queried once on the mover's board and swapped, never re-queried
    # on the flipped board (that would change whose move it is).
    flipped = feature_ids(SWAP[boards[:, ANTI]], since, clock, race[:, ::-1])
    assert np.array_equal(ids[:, 0], flipped[:, 1])
    assert np.array_equal(ids[:, 1], flipped[:, 0])


def test_clock_buckets_follow_the_lower_bounds():
    assert clock_bucket(np.array([0, 1, 2, 5, 6, 7, 159, 160, 199, 400])).tolist() == [0, 1, 2, 4, 5, 5, 14, 15, 15, 15]
    ids = feature_ids(np.zeros((1, 81), np.uint8), np.array([3]), np.array([50]))
    assert ids[0, 0, 40] == ELAPSED_BASE + 3
    assert ids[0, 0, 41] == REMAINING_BASE + clock_bucket(np.array([47]))[0]
    assert np.all(ids[:, :, :40] == PAD)


def test_invalid_positions_are_rejected():
    board = np.zeros((1, 81), dtype=np.uint8)
    with pytest.raises(ValueError):
        feature_ids(np.ones((1, 81), np.uint8), [0], [200])
    board[0, 20] = 7
    with pytest.raises(ValueError):
        feature_ids(board, [0], [200])
    with pytest.raises(ValueError):
        feature_ids(np.zeros((1, 81), np.uint8), [0], [0])
    with pytest.raises(ValueError):
        feature_ids(np.zeros((1, 81), np.uint8), [0], [200], race=[[9, 0]])


def test_race_rows_follow_the_engine_query():
    board = np.zeros((1, 81), dtype=np.uint8)
    empty = feature_ids(board, [0], [200])
    assert empty[0, 0, 42:].tolist() == [RACE_BASE + 8, RACE_BASE + RACE_BUCKETS + 8]
    # A lone rock one step from its goal: the mover's runner arrives in one move.
    board[0, 8 * 9 - 9 + 4] = 1
    ids = feature_ids(board, [0], [200])
    mover, opponent = (int(v) for v in race_buckets(board)[0])
    assert ids[0, 0, 42:].tolist() == [RACE_BASE + mover, RACE_BASE + RACE_BUCKETS + opponent]
    assert ids[0, 1, 42:].tolist() == [RACE_BASE + opponent, RACE_BASE + RACE_BUCKETS + mover]
    assert (mover, opponent) != (8, 8)


def test_symmetry_table_matches_transformed_boards():
    boards = random_boards(3, 24)
    since, clock = np.full(24, 40), np.full(24, 200)
    ids = feature_ids(boards, since, clock)
    table = symmetry_table()
    assert table.shape == (6, FEATURES + 1)
    for k in range(6):
        moved = feature_ids(transform_board(boards, k), since, clock)
        assert np.array_equal(np.sort(table[k, ids], axis=-1), np.sort(moved, axis=-1)), k
    assert np.array_equal(transform_board(boards, 0), boards)


def test_orbit_hash_and_piece_bucket():
    boards = random_boards(11, 16, 2, 21)
    keys = orbit_hash(boards)
    for k in range(6):
        assert np.array_equal(orbit_hash(transform_board(boards, k)), keys)
    counts = np.count_nonzero(boards, axis=1)
    want = np.where(counts <= 4, 0, np.where(counts <= 8, 1, np.where(counts <= 12, 2, 3)))
    assert np.array_equal(piece_bucket(boards), want)
