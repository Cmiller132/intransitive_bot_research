"""The format 8 fixtures shared with the Rust crate (`tests/fixtures/format8_*`
and `format6_h32.nnue`): two deterministic H32 networks and 512 positions
with their contexts, ids and integer raw values under both. The test rebuilds
them in memory and compares, so the files can never drift from the Python
side (runs/nnue_plan/format8_contract.md)."""

import json
import struct
from pathlib import Path

import numpy as np

from nnue.export import HEADER, MAGIC, file_size, integer_eval
from nnue.features import CONTEXTS, FORMAT6, FORMAT8, PIECE_ROWS, context, feature_ids
from nnue.model import DENSE, QA, QB

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
HIDDEN = 32
COUNT = 512


def h32_bytes(version: int) -> bytes:
    """The deterministic H32 net of a version: weights[r, j] = ((13 r + 7 j + 5 c) mod 31) - 15
    with c the row's context (0 in format 6); bias[j] = 96 + (j mod 7); readout[k] = ((17 k) mod 15) - 7,
    readout bias 0; dense[i, k] = ((13 i + k) mod 255) - 128; dense bias[i] = (i - 16) 255 x 32;
    residual[i] = i - 16."""
    layout = {6: FORMAT6, 8: FORMAT8}[version]
    h, f = HIDDEN, layout.features
    r = np.arange(f)[:, None]
    j = np.arange(h)[None, :]
    c = np.where(r < layout.attack_base, r // PIECE_ROWS, 0)
    weights = ((13 * r + 7 * j + 5 * c) % 31 - 15).astype("<i2")
    bias = (96 + np.arange(h) % 7).astype("<i2")
    readout = ((17 * np.arange(2 * h)) % 15 - 7).astype("<i2")
    i = np.arange(DENSE)[:, None]
    k = np.arange(2 * h)[None, :]
    dense = ((13 * i + k) % 255 - 128).astype("i1")
    dense_bias = ((np.arange(DENSE) - 16) * QA * DENSE).astype("<i4")
    residual = (np.arange(DENSE) - 16).astype("<i2")
    raw = HEADER.pack(MAGIC, version, f, h, QA, QB, 600.0, 1)
    raw += bias.tobytes() + weights.tobytes() + readout.tobytes() + struct.pack("<i", 0)
    raw += struct.pack("<I", DENSE) + dense.tobytes() + dense_bias.tobytes() + residual.tobytes()
    assert len(raw) == file_size(h, 1, version)
    return raw


def positions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """512 positions with legal material (each side keeps at most its 3 rocks,
    4 papers and 3 scissors, at least one piece, nobody on a goal square)."""
    rng = np.random.default_rng(20260912)
    boards = np.zeros((COUNT, 81), dtype=np.uint8)
    for board in boards:
        while True:
            counts = [
                rng.integers(0, 4),
                rng.integers(0, 5),
                rng.integers(0, 4),
                rng.integers(0, 4),
                rng.integers(0, 5),
                rng.integers(0, 4),
            ]
            if sum(counts[:3]) and sum(counts[3:]):
                break
        codes = np.repeat(np.arange(1, 7), counts)
        squares = rng.choice(np.arange(1, 80), len(codes), replace=False)  # squares 0 and 80 are the goals
        board[squares] = codes
    clock = rng.integers(50, 201, COUNT)
    since = (rng.random(COUNT) * clock).astype(np.int64)
    return boards, since, clock


def records(eight: Path, six: Path) -> list[dict]:
    boards, since, clock = positions()
    ids = feature_ids(boards, since, clock)
    contexts = context(boards)
    raw8 = integer_eval(eight, boards, since, clock)
    raw6 = integer_eval(six, boards, since, clock)
    return [
        {
            "board": boards[n].tolist(),
            "since_capture": int(since[n]),
            "clock": int(clock[n]),
            "context": contexts[n].tolist(),
            "ids": ids[n].tolist(),
            "raw": float(raw8[n]),
            "raw6": float(raw6[n]),
        }
        for n in range(COUNT)
    ]


def test_fixture_networks_and_positions_match_the_python_side():
    eight, six = FIXTURES / "format8_h32.nnue", FIXTURES / "format6_h32.nnue"
    assert eight.read_bytes() == h32_bytes(8)
    assert six.read_bytes() == h32_bytes(6)
    lines = [json.loads(line) for line in (FIXTURES / "format8_positions.jsonl").read_text("utf-8").splitlines()]
    assert lines == records(eight, six)
    seen = {c for line in lines for c in line["context"]}
    # Context 0 (an opponent without pieces) is a finished game and never occurs in a legal position.
    assert seen == set(range(1, CONTEXTS)) and any(line["raw"] != line["raw6"] for line in lines)
    for line in lines:
        for perspective in (0, 1):
            active = [i for i in line["ids"][perspective] if i != FORMAT8.pad]
            assert len(active) == len(set(active)) and len(active) <= 42
            pieces = [i for i in active if i < FORMAT8.attack_base]
            assert {i // PIECE_ROWS for i in pieces} == {line["context"][perspective]}
