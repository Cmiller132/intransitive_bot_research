"""The format 9 fixtures shared with the Rust crate (`tests/fixtures/format9_*`):
a seeded random-weight H32 network whose eight heads all differ, 380 positions
covering every piece-count bucket and every goal-corner occupant, and the
conversion of the format 8 fixture, which must evaluate every shared position
exactly as its source. The test rebuilds all three in memory and compares, so
the files can never drift from the Python side (models/nnue/FORMAT9.md)."""

import json
from pathlib import Path

import engine
import numpy as np

from nnue.export import HEADER, MAGIC, convert, file_size, integer_eval, read
from nnue.features import CONTEXTS, FORMAT9, GOAL_ROWS, HEADS, context, feature_ids9, piece_bucket
from nnue.model import DENSE, QA, QB

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
HIDDEN = 32
REPEATS = 5  # boards per piece count and corner combination
TOTALS = range(2, 21)
CORNERS = range(4)  # bit 0: one of the mover's pieces on square 0; bit 1: an enemy on square 80
COUNT = len(TOTALS) * len(CORNERS) * REPEATS
MATERIAL = np.repeat([1, 2, 3], [3, 4, 3])  # one side's pieces: three rocks, four papers, three scissors


def h32_bytes9() -> bytes:
    """The seeded random-weight H32 version 9 network. Random weights rather
    than a formula: the eight heads differ in every entry, so a value that
    reached the wrong head cannot match the oracle."""
    rng = np.random.default_rng(20260916)
    h, f = HIDDEN, FORMAT9.features
    raw = HEADER.pack(MAGIC, FORMAT9.version, f, h, QA, QB, 600.0, HEADS)
    raw += rng.integers(64, 128, h).astype("<i2").tobytes()
    raw += rng.integers(-15, 16, (f, h)).astype("<i2").tobytes()
    for _ in range(HEADS):
        raw += rng.integers(-7, 8, 2 * h).astype("<i2").tobytes()
        raw += rng.integers(-64, 65, 1).astype("<i4").tobytes()
        raw += rng.integers(-128, 128, (DENSE, 2 * h)).astype("i1").tobytes()
        raw += (rng.integers(-16, 17, DENSE) * QA * DENSE).astype("<i4").tobytes()
        raw += rng.integers(-16, 17, DENSE).astype("<i2").tobytes()
    assert len(raw) == file_size(h, FORMAT9.version)
    return raw


def one_board(rng: np.random.Generator, total: int, corners: int) -> np.ndarray:
    """A legal non-terminal board with `total` pieces and the corner occupants
    `corners` asks for: only the mover's own piece may stand on square 0 and
    only an enemy piece on square 80, anything else has already won."""
    while True:
        own = int(rng.integers(max(1, total - 10), min(10, total - 1) + 1))
        board = np.zeros(81, dtype=np.uint8)
        codes = [list(rng.permutation(MATERIAL)[:own]), list(rng.permutation(MATERIAL)[: total - own] + 3)]
        squares = list(rng.permutation(np.arange(1, 80))[:total])
        for bit, (side, corner) in enumerate([(0, 0), (1, 80)]):
            if corners >> bit & 1:
                board[corner] = codes[side].pop()
                squares.pop()
        for square, code in zip(squares, codes[0] + codes[1], strict=True):
            board[square] = code
        if np.count_nonzero(engine.legal_mask(board.tolist())):
            return board


def positions9() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Boards, since_capture, clock and a ply at least since_capture: every
    piece count from 2 to 20 (so every output bucket) crossed with every
    combination of occupied goal corners."""
    rng = np.random.default_rng(20260916)
    boards = np.stack(
        [one_board(rng, total, corners) for total in TOTALS for corners in CORNERS for _ in range(REPEATS)]
    )
    clock = rng.integers(50, 201, COUNT)
    since = (rng.random(COUNT) * clock).astype(np.int64)
    ply = since + rng.integers(0, 40, COUNT)
    return boards, since, clock, ply


def records9(net: Path) -> list[dict]:
    boards, since, clock, ply = positions9()
    ids = feature_ids9(boards, since, clock)
    contexts = context(boards)
    buckets = piece_bucket(np.count_nonzero(boards, axis=1))
    raw = integer_eval(net, boards, since, clock)
    return [
        {
            "board": boards[n].tolist(),
            "since_capture": int(since[n]),
            "ply": int(ply[n]),
            "clock": int(clock[n]),
            "context": contexts[n].tolist(),
            "bucket": int(buckets[n]),
            "ids": ids[n].tolist(),
            "raw": float(raw[n]),
        }
        for n in range(COUNT)
    ]


def test_fixture_network_and_positions_match_the_python_side():
    net = FIXTURES / "format9_h32.nnue"
    assert net.read_bytes() == h32_bytes9()
    lines = [json.loads(line) for line in (FIXTURES / "format9_positions.jsonl").read_text("utf-8").splitlines()]
    assert lines == records9(net)
    assert {line["bucket"] for line in lines} == set(range(HEADS))
    seen = {c for line in lines for c in line["context"]}
    # Context 0 is an opponent without pieces, a finished game; the rest occur.
    assert seen <= set(range(1, CONTEXTS)) and len(seen) >= 20
    corners = {i - FORMAT9.goal_base for line in lines for side in line["ids"] for i in side[42:]}
    assert corners == set(range(GOAL_ROWS))  # every occupant of both corners, in both frames
    for line in lines:
        pieces = 81 - line["board"].count(0)
        assert line["bucket"] == min(HEADS - 1, (pieces - 2) * HEADS // 19)
        for side in line["ids"]:
            assert len(side) == FORMAT9.slots == 44
            active = [i for i in side if i != FORMAT9.pad]
            assert len(active) == len(set(active))
            assert sum(1 for i in side[42:] if i < FORMAT9.goal_base + 4) == 1  # one row of each group
        # Nobody has already reached a goal: only an enemy stands on square 80
        # and only one of the mover's own pieces on square 0.
        assert line["board"][0] in (0, 1, 2, 3) and line["board"][80] in (0, 4, 5, 6)


def test_the_converted_format8_fixture_evaluates_every_position_identically(tmp_path):
    eight, nine = FIXTURES / "format8_h32.nnue", FIXTURES / "format9_convert_h32.nnue"
    info = convert(eight, tmp_path / "again.nnue", to=9)
    assert nine.read_bytes() == (tmp_path / "again.nnue").read_bytes()
    assert (info["version"], info["buckets"], info["bytes"]) == (9, HEADS, file_size(HIDDEN, 9))
    a, b = read(eight), read(nine)
    assert np.array_equal(b["weights"][: FORMAT9.goal_base], a["weights"])
    assert not b["weights"][FORMAT9.goal_base :].any()
    assert np.array_equal(a["bias"], b["bias"])
    for index in range(HEADS):
        for key in ("output", "dense", "dense_bias", "residual"):
            assert np.array_equal(b[key][index], a[key]), (key, index)
        assert b["output_bias"][index] == a["output_bias"][0]
    lines = [json.loads(line) for line in (FIXTURES / "format8_positions.jsonl").read_text("utf-8").splitlines()]
    boards = np.array([line["board"] for line in lines], dtype=np.uint8)
    since = np.array([line["since_capture"] for line in lines])
    clock = np.array([line["clock"] for line in lines])
    converted = integer_eval(nine, boards, since, clock)
    assert np.array_equal(converted, integer_eval(eight, boards, since, clock))
    assert converted.tolist() == [line["raw"] for line in lines]
    # The positions cover every bucket, so every repeated head was used.
    assert set(piece_bucket(np.count_nonzero(boards, axis=1)).tolist()) == set(range(HEADS))
