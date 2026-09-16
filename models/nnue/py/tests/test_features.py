"""The feature encoder against a slow enumeration, the material contexts,
the perspective exchange, the format-6 row map, the format-9 goal rows, the
symmetry tables and the clock buckets."""

import numpy as np
import pytest

from nnue.features import (
    ANTI,
    CLOCK_BOUNDS,
    CONTEXTS,
    FORMAT6,
    FORMAT8,
    FORMAT9,
    GOAL_ROWS,
    GOAL_STATES,
    MAX_PIECES,
    PIECE_ROWS,
    SLOTS,
    SWAP,
    clock_bucket,
    context,
    feature_ids,
    feature_ids9,
    orbit_hash,
    piece_bucket,
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


def slow_ids(board, since, clock):
    """One perspective at a time by plain loops."""
    perspectives = [list(map(int, board))]
    reverse = []
    for square in range(81):
        rank, file = divmod(square, 9)
        piece = int(board[(8 - file) * 9 + 8 - rank])
        reverse.append(0 if piece == 0 else piece + 3 if piece <= 3 else piece - 3)
    perspectives.append(reverse)
    out = []
    for position in perspectives:
        enemy = [sum(1 for p in position if p == code) for code in (4, 5, 6)]
        c = min(enemy[0], 2) + 3 * min(enemy[1], 2) + 9 * min(enemy[2], 2)
        pieces, hit = [], []
        for square, piece in enumerate(position):
            if not piece:
                continue
            pieces.append(PIECE_ROWS * c + (piece - 1) * 81 + square)
            rank, file = divmod(square, 9)
            threatened = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr, dc) == (0, 0):
                        continue
                    rr, ff = rank + dr, file + dc
                    if not (0 <= rr < 9 and 0 <= ff < 9):
                        continue
                    enemy_piece = position[rr * 9 + ff]
                    if enemy_piece and (enemy_piece <= 3) != (piece <= 3):
                        threatened |= ((enemy_piece - 1) % 3 - (piece - 1) % 3) % 3 == 1
            if threatened:
                hit.append(FORMAT8.attack_base + (piece - 1) * 81 + square)
        elapsed = int(np.searchsorted(CLOCK_BOUNDS, since, side="right") - 1)
        remaining = int(np.searchsorted(CLOCK_BOUNDS, clock - since, side="right") - 1)
        slots = pieces + [FORMAT8.pad] * (MAX_PIECES - len(pieces)) + hit + [FORMAT8.pad] * (MAX_PIECES - len(hit))
        out.append(slots + [FORMAT8.elapsed_base + elapsed, FORMAT8.remaining_base + remaining])
    return np.array(out, dtype=np.int64)


def test_encoder_matches_the_slow_enumeration():
    boards = random_boards(20260909, 64)
    since = np.random.default_rng(1).integers(0, 200, 64)
    clock = np.full(64, 200)
    got = feature_ids(boards, since, clock)
    assert got.shape == (64, 2, SLOTS)
    want = np.stack([slow_ids(b, int(s), 200) for b, s in zip(boards, since, strict=True)])
    assert np.array_equal(got, want)
    assert got.max() == FORMAT8.pad < 2**15  # the int16 id cache holds every id


def test_context_counts_the_opponent_capped_at_two():
    board = np.zeros((1, 81), dtype=np.uint8)
    board[0, [0, 1, 2]] = 1  # three own rocks: the opponent's perspective caps them at two
    board[0, 3] = 2
    board[0, 40] = 4
    board[0, [70, 71]] = 6
    assert context(board).tolist() == [[1 + 9 * 2, 2 + 3 * 1]]
    assert context(np.zeros((1, 81), dtype=np.uint8)).tolist() == [[0, 0]]
    ids = feature_ids(board, [0], [200])
    assert set((ids[0, 0, :6] // PIECE_ROWS).tolist()) == {19} and set((ids[0, 1, :6] // PIECE_ROWS).tolist()) == {5}


def test_perspectives_exchange_under_the_frame_flip():
    boards = random_boards(7, 32)
    since, clock = np.full(32, 17), np.full(32, 120)
    ids = feature_ids(boards, since, clock)
    flipped = feature_ids(SWAP[boards[:, ANTI]], since, clock)
    assert np.array_equal(ids[:, 0], flipped[:, 1])
    assert np.array_equal(ids[:, 1], flipped[:, 0])
    assert np.array_equal(context(SWAP[boards[:, ANTI]]), context(boards)[:, ::-1])


def test_format6_rows_drop_the_context_and_shift_the_rest():
    assert (FORMAT8.features, FORMAT8.attack_base, FORMAT8.elapsed_base, FORMAT8.remaining_base) == (
        13640,
        13122,
        13608,
        13624,
    )
    assert (FORMAT6.features, FORMAT6.attack_base, FORMAT6.elapsed_base, FORMAT6.remaining_base) == (
        1004,
        486,
        972,
        988,
    )
    boards = random_boards(5, 40)
    since, clock = np.full(40, 9), np.full(40, 200)
    eight = feature_ids(boards, since, clock)
    six = FORMAT6.rows(eight)
    assert FORMAT8.rows(eight) is eight
    piece = eight < FORMAT8.attack_base
    assert np.array_equal(six[piece], eight[piece] % PIECE_ROWS)
    assert np.array_equal(six[~piece], eight[~piece] - (FORMAT8.attack_base - FORMAT6.attack_base))
    assert np.all(six[eight == FORMAT8.pad] == FORMAT6.pad) and six.max() == FORMAT6.pad


def test_clock_buckets_follow_the_lower_bounds():
    assert clock_bucket(np.array([0, 1, 2, 5, 6, 7, 159, 160, 199, 400])).tolist() == [0, 1, 2, 4, 5, 5, 14, 15, 15, 15]
    ids = feature_ids(np.zeros((1, 81), np.uint8), np.array([3]), np.array([50]))
    assert ids[0, 0, 40] == FORMAT8.elapsed_base + 3
    assert ids[0, 0, 41] == FORMAT8.remaining_base + clock_bucket(np.array([47]))[0]
    assert np.all(ids[:, :, :40] == FORMAT8.pad)


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
        feature_ids(np.zeros((2, 81), np.uint8), [0], [200, 200])


@pytest.mark.parametrize("layout", [FORMAT8, FORMAT6])
def test_symmetry_table_matches_transformed_boards(layout):
    boards = random_boards(3, 24)
    since, clock = np.full(24, 40), np.full(24, 200)
    ids = layout.rows(feature_ids(boards, since, clock))
    table = symmetry_table(layout)
    assert table.shape == (6, layout.features + 1)
    for k in range(6):
        assert np.array_equal(np.sort(table[k]), np.arange(layout.features + 1))
        moved = layout.rows(feature_ids(transform_board(boards, k), since, clock))
        assert np.array_equal(np.sort(table[k, ids], axis=-1), np.sort(moved, axis=-1)), k
    assert np.array_equal(transform_board(boards, 0), boards)
    assert CONTEXTS == 27


def cornered_boards(seed=17, count=24):
    """Boards covering every combination of goal-corner occupants: only one of
    the mover's own pieces may stand on square 0 and only an enemy on square 80,
    so those are the sixteen combinations the encoder can meet in play."""
    boards = random_boards(seed, count, 2, 19)
    for n, board in enumerate(boards):
        board[0] = (0, 1, 2, 3)[n % 4]
        board[80] = (0, 4, 5, 6)[n // 4 % 4]
    return boards


def test_symmetry_table_renames_the_goal_corners_of_format9():
    boards = cornered_boards()
    since, clock = np.full(len(boards), 40), np.full(len(boards), 200)
    ids = feature_ids9(boards, since, clock)
    table = symmetry_table(FORMAT9)
    assert table.shape == (6, FORMAT9.features + 1)
    # Both groups and every occupant of each are present, so the goal columns
    # of the table are exercised and not only its identity entries.
    assert {int(i) - FORMAT9.goal_base for i in ids[..., 42:].reshape(-1)} == set(range(GOAL_ROWS))
    for k in range(6):
        assert np.array_equal(np.sort(table[k]), np.arange(FORMAT9.features + 1))
        moved = feature_ids9(transform_board(boards, k), since, clock)
        # The goal rows sit in fixed slots: they match position by position. The
        # diagonal reflection (k // 3) fixes squares 0 and 80, so a corner's
        # occupant is renamed within its group and the two groups never swap.
        assert np.array_equal(table[k, ids][..., 42:], moved[..., 42:]), k
        assert np.all(table[k, ids][..., 42] < FORMAT9.goal_base + GOAL_STATES), k
        assert np.all(table[k, ids][..., 43] >= FORMAT9.goal_base + GOAL_STATES), k
        assert np.array_equal(np.sort(table[k, ids], axis=-1), np.sort(moved, axis=-1)), k
        # A symmetry moves and renames pieces but never adds or removes one.
        assert np.array_equal(
            piece_bucket(np.count_nonzero(transform_board(boards, k), axis=1)),
            piece_bucket(np.count_nonzero(boards, axis=1)),
        ), k
    # A renaming really moves the rows: the identity is the only fixed table.
    assert not np.array_equal(table[1, ids][..., 42:], ids[..., 42:])


def test_orbit_hash_is_constant_on_an_orbit():
    boards = random_boards(11, 16, 2, 21)
    keys = orbit_hash(boards)
    for k in range(6):
        assert np.array_equal(orbit_hash(transform_board(boards, k)), keys)
