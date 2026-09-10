"""The teacher labeller: exact proofs from the engine, and the search values
of a fake `bot analyse` written back with the rows' provenance kept."""

import json

import engine
import numpy as np

from nnue import data, features, label, paths


def board_with(cells: dict[int, int]) -> np.ndarray:
    board = np.zeros(81, dtype=np.uint8)
    for square, code in cells.items():
        board[square] = code
    return board


def test_proofs_mark_wins_at_once_and_forced_losses():
    # Own rock one step from the goal corner (80): a win at once.
    win = board_with({71: 1, 0: 4})
    # A lone own rock next to an enemy paper on the goal: every move is
    # captured at once, an exact loss.
    lose = board_with({70: 1, 80: 5, 79: 5, 61: 5, 69: 5, 71: 5})
    boards = np.stack([win, lose, np.frombuffer(engine.initial_board(), dtype=np.uint8)])
    values = label.proofs(boards)
    assert values[0] == 1
    assert values[1] in (-1, 0)  # -1 when every move loses in two, else open
    assert values[2] == 0


def test_label_writes_teacher_values_and_keeps_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    rng = np.random.default_rng(2)
    n = 12
    boards = np.stack([np.frombuffer(engine.initial_board(), dtype=np.uint8)] * n)
    rows = {
        "board": boards,
        "since_capture": rng.integers(0, 30, n).astype(np.uint16),
        "ply": np.arange(n, dtype=np.uint16),
        "capture_clock": np.full(n, 200, dtype=np.uint16),
        "target": np.zeros(n, dtype=np.float32),
        "weight": np.ones(n, dtype=np.float32),
        "kind": np.full(n, data.KIND_RAW, dtype=np.uint8),
        "outcome": np.zeros(n, dtype=np.int8),
        "outcome_ok": np.zeros(n, dtype=bool),
        "source": np.full(n, data.SOURCE_STUDENT, dtype=np.uint8),
        "game": np.arange(n, dtype=np.uint32) // 3,
        "orbit": features.orbit_hash(boards),
        "split": np.zeros(n, dtype=np.uint8),
    }
    data.write(paths.data_dir("raw"), rows, {"producer": "test"})
    seen = []

    def analyse(engine_spec, requests, sims):
        seen.append((engine_spec, sims, len(requests)))
        out = []
        for i, row in enumerate(requests):
            assert len(row["board"]) == 81
            value = 0.5 if i % 2 else -1.5  # the second is clipped
            out.append({"search": {"root_value": value, "lines": [{"q": value / 2}]}} if i != 3 else {"error": "x"})
        return out

    monkeypatch.setattr(label, "analyse", analyse)
    target = label.label("raw", "teacher", "sq:fake", sims=64, workers=2)
    assert target == paths.data_dir("teacher")
    values = np.load(target / "target.npy")
    kinds = np.load(target / "kind.npy")
    assert (kinds == data.KIND_TEACHER).all()
    assert set(np.unique(values)) <= {-1.0, 0.5}
    assert (np.load(target / "source.npy") == data.SOURCE_STUDENT).all()
    assert (np.load(target / "capture_clock.npy") == 200).all()
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["sims"] == 64 and provenance["errors"] == 2
    assert provenance["rows"] == n - 2 and provenance["input"]["rows"] == n
    assert sum(count for _, _, count in seen) == n and all(s == 64 for _, s, _ in seen)
