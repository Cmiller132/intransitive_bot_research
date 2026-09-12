"""The format 8 fixtures shared with the Rust crate (`tests/fixtures/format8_*`
and `format6_h32.nnue`): two deterministic H32 networks and 512 positions
(with a ply, so the file is direct input to the Rust diagnostic) with their
contexts, ids and integer raw values under both. The test rebuilds them in
memory and compares, so the files can never drift from the Python side
(runs/nnue_plan/format8_contract.md)."""

import json
import struct
from pathlib import Path

import engine
import numpy as np

from nnue.export import HEADER, MAGIC, file_size, integer_eval
from nnue.features import CONTEXTS, FORMAT6, FORMAT8, PIECE_ROWS, context, feature_ids
from nnue.model import DENSE, QA, QB

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
HIDDEN = 32
COUNT = 512
CLOCK = 120  # the trajectories' capture clock: shorter draw tails than the site's 200, elapsed buckets to 96..128
MIN_PLIES = 1200
MIN_GAMES = 10
CAPTURE_BIAS = 0.5
STEPS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]  # the action's direction index


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
    assert len(raw) == file_size(h, version)
    return raw


def positions() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """512 positions with legal material (each side keeps at most its 3 rocks,
    4 papers and 3 scissors, at least one piece, nobody on a goal square):
    boards, since_capture, clock and a ply at least since_capture."""
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
    ply = since + rng.integers(0, 40, COUNT)
    return boards, since, clock, ply


def records(eight: Path, six: Path) -> list[dict]:
    boards, since, clock, ply = positions()
    ids = feature_ids(boards, since, clock)
    contexts = context(boards)
    raw8 = integer_eval(eight, boards, since, clock)
    raw6 = integer_eval(six, boards, since, clock)
    return [
        {
            "board": boards[n].tolist(),
            "since_capture": int(since[n]),
            "ply": int(ply[n]),
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


def trajectories() -> list[dict]:
    """Random legal games from the initial position (a capture is chosen with
    probability 0.5 when one exists) until at least ten games and 1,200 plies
    were played and every count transition n -> n - 1 of every piece type on
    both colours occurred. One line per state in the mover's frame with the action played
    from it (None at the end of a game), the outcome the engine returned for
    that state, its legal-move count, contexts, ids and raw values."""
    rng = np.random.default_rng(20260913)
    starts = ((1, 3), (2, 4), (3, 3))  # code, initial count per side
    needed = {(colour, code, n) for colour in (0, 1) for code, start in starts for n in range(1, start + 1)}
    seen: set = set()
    games: list[list[tuple]] = []
    plies = 0
    while plies < MIN_PLIES or len(games) < MIN_GAMES or not needed <= seen:
        board, since, ply, outcome = list(engine.initial_board()), 0, 0, 0
        states = []
        while outcome == 0:
            legal = np.flatnonzero(np.array(engine.legal_mask(board), dtype=bool))
            if not len(legal):
                break
            captures = [a for a in legal if board[a % 81 + STEPS[a // 81][0] * 9 + STEPS[a // 81][1]] != 0]
            pool = captures if captures and rng.random() < CAPTURE_BIAS else legal
            action = int(rng.choice(pool))
            states.append((board, since, ply, outcome, len(legal), action))
            child, since_after, ply_after, outcome = engine.apply(board, since, ply, action, CLOCK)
            child = list(child)
            # The mover's material is codes 1-3 before and 4-6 after the frame flip; the opponent's the reverse.
            for offset, before, after in ((0, (1, 2, 3), (4, 5, 6)), (1, (4, 5, 6), (1, 2, 3))):
                for t in range(3):
                    was = board.count(before[t])
                    if child.count(after[t]) < was:
                        seen.add(((ply + offset) % 2, t + 1, was))
            board, since, ply = child, since_after, ply_after
        states.append((board, since, ply, outcome, int(np.count_nonzero(engine.legal_mask(board))), None))
        games.append(states)
        plies += len(states) - 1
    flat = [(g, *state) for g, states in enumerate(games) for state in states]
    boards = np.array([state[1] for state in flat], dtype=np.uint8)
    since = np.array([state[2] for state in flat], dtype=np.int64)
    clock = np.full(len(flat), CLOCK)
    ids = feature_ids(boards, since, clock)
    contexts = context(boards)
    raw8 = integer_eval(FIXTURES / "format8_h32.nnue", boards, since, clock)
    raw6 = integer_eval(FIXTURES / "format6_h32.nnue", boards, since, clock)
    return [
        {
            "game": g,
            "ply": int(ply),
            "board": list(map(int, board)),
            "since_capture": int(s),
            "clock": CLOCK,
            "outcome": int(outcome),
            "legal": int(legal),
            "action": action,
            "context": contexts[n].tolist(),
            "ids": ids[n].tolist(),
            "raw": float(raw8[n]),
            "raw6": float(raw6[n]),
        }
        for n, (g, board, s, ply, outcome, legal, action) in enumerate(flat)
    ]


def test_fixture_trajectories_match_the_python_side():
    lines = [json.loads(line) for line in (FIXTURES / "format8_trajectories.jsonl").read_text("utf-8").splitlines()]
    assert lines == trajectories()
    games = {line["game"] for line in lines}
    assert len(lines) - len(games) >= MIN_PLIES
    assert all(line["action"] is None for line in lines if line["outcome"] != 0 or line["legal"] == 0)
    # Every state follows from the previous one by the engine.
    for before, after in zip(lines, lines[1:], strict=False):
        if before["action"] is None:
            assert after["game"] == before["game"] + 1 and after["ply"] == 0
            continue
        child, since, ply, outcome = engine.apply(
            before["board"], before["since_capture"], before["ply"], before["action"], CLOCK
        )
        assert list(child) == after["board"]
        assert (since, ply, outcome) == (after["since_capture"], after["ply"], after["outcome"])
    contexts = {c for line in lines for c in line["context"]}
    assert {0, 26} <= contexts and len(contexts) >= 15  # a side lost every piece; the random positions cover the rest
