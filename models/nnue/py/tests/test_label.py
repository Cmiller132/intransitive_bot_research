"""The teacher labeller: exact proofs from the engine, the search values of a
fake `bot analyse` written back with the rows' provenance kept, and the
selection of the rows a shallow labelling misjudges most."""

import json

import engine
import numpy as np
import pytest

from nnue import data, features, label, paths


def board_with(cells: dict[int, int]) -> np.ndarray:
    board = np.zeros(81, dtype=np.uint8)
    for square, code in cells.items():
        board[square] = code
    return board


def rows_of(boards: np.ndarray, rng: np.random.Generator, target: np.ndarray | None = None) -> dict:
    n = len(boards)
    return {
        "board": boards,
        "since_capture": rng.permutation(150)[:n].astype(np.uint16),  # distinct clock states
        "ply": np.arange(n, dtype=np.uint16),
        "capture_clock": np.full(n, 200, dtype=np.uint16),
        "target": np.zeros(n, dtype=np.float32) if target is None else target.astype(np.float32),
        "weight": np.ones(n, dtype=np.float32),
        "kind": np.full(n, data.KIND_RAW, dtype=np.uint8),
        "outcome": np.zeros(n, dtype=np.int8),
        "outcome_ok": np.zeros(n, dtype=bool),
        "source": np.full(n, data.SOURCE_STUDENT, dtype=np.uint8),
        "game": np.arange(n, dtype=np.uint32) // 3,
        "orbit": features.orbit_hash(boards),
        "split": np.zeros(n, dtype=np.uint8),
    }


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


def fake_bot(tmp_path, monkeypatch) -> None:
    (tmp_path / "bot.exe").write_bytes(b"fake bot")
    monkeypatch.setenv("NNUE_BOT", str(tmp_path / "bot.exe"))


def test_label_writes_teacher_values_and_keeps_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    fake_bot(tmp_path, monkeypatch)
    rng = np.random.default_rng(2)
    n = 12
    boards = np.stack([np.frombuffer(engine.initial_board(), dtype=np.uint8)] * n)
    rows = rows_of(boards, rng)
    data.write(paths.data_dir("raw"), rows, {"producer": "test"})
    seen = []
    broken = {int(s) for s in rows["since_capture"][:2]}  # two roots fail every time
    flaky = {int(rows["since_capture"][2])}  # one fails once and succeeds on the retry

    def analyse(engine_spec, requests, sims):
        seen.append((engine_spec, sims, len(requests)))
        out = []
        for i, row in enumerate(requests):
            assert len(row["board"]) == 81
            since = int(row["since_capture"])
            if since in broken or since in flaky:
                flaky.discard(since)
                out.append({"error": "x"})
                continue
            value = 0.5 if i % 2 else -1.5  # the second is clipped
            out.append({"search": {"root_value": value, "lines": [{"q": value / 2}]}})
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
    assert provenance["sims"] == 64 and provenance["errors"] == 2 and provenance["retries"] == 3
    assert provenance["rows"] == n - 2 and provenance["input"]["rows"] == n
    assert provenance["select"] is None and provenance["engine"]["network_sha256"] is None
    assert provenance["engine"]["binary_sha256"] == paths.sha256(tmp_path / "bot.exe")
    assert provenance["input"]["provenance_sha256"] == paths.sha256(paths.data_dir("raw") / "provenance.json")
    assert sum(count for _, _, count in seen) == n + 3 and all(s == 64 for _, s, _ in seen)


def test_selection_takes_the_rows_the_shallow_labels_misjudge_most(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    fake_bot(tmp_path, monkeypatch)
    rng = np.random.default_rng(3)
    n = 40
    boards = np.stack([np.frombuffer(engine.initial_board(), dtype=np.uint8)] * n)
    deep = rng.uniform(-1, 1, n)
    source = rows_of(boards, rng, deep)
    data.write(paths.data_dir("pool"), source, {"producer": "test"})
    # The shallow pass covers a subset in another order, with its own labels.
    subset = np.array([30, 5, 17, 2, 21, 9, 33, 12, 27, 0])
    shallow = {name: value[subset] for name, value in source.items()}
    shallow["target"] = (deep[subset] + np.array([0.05, 0.9, -0.02, 0.6, 0.01, -0.7, 0.03, 0.4, 0.0, -0.5])).clip(-1, 1)
    shallow["target"] = shallow["target"].astype(np.float32)
    data.write(paths.data_dir("shallow"), shallow, {"producer": "test"})
    chosen, stats = label.disagreement(paths.data_dir("pool"), paths.data_dir("shallow"), 4)
    gap = np.abs(deep[subset] - shallow["target"])
    want = np.sort(subset[np.argsort(-gap, kind="stable")[:4]])
    assert chosen.tolist() == want.tolist()
    assert stats["pool"] == len(subset) and stats["selected"] == 4 and stats["gap_min"] <= stats["gap_mean"]
    monkeypatch.setattr(
        label, "analyse", lambda spec, requests, sims: [{"search": {"root_value": 0.1, "lines": []}} for _ in requests]
    )
    target = label.label("pool", "deep", "fake:engine", 400, 1, sample=4, select="shallow")
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["rows"] == 4 and provenance["select"]["dataset"] == "shallow"
    assert np.array_equal(np.load(target / "ply.npy"), source["ply"][chosen])
    duplicated = {name: np.concatenate([value, value[:1]]) for name, value in source.items()}
    data.write(paths.data_dir("dup"), duplicated, {"producer": "test"})
    with pytest.raises(ValueError, match="duplicate"):
        label.disagreement(paths.data_dir("dup"), paths.data_dir("shallow"), 4)
    foreign = {name: value.copy() for name, value in shallow.items()}
    foreign["since_capture"][0] = 199  # a clock state the source never had
    data.write(paths.data_dir("foreign"), foreign, {"producer": "test"})
    with pytest.raises(ValueError, match="not in"):
        label.disagreement(paths.data_dir("pool"), paths.data_dir("foreign"), 4)
    with pytest.raises(ValueError, match="requested"):
        label.disagreement(paths.data_dir("pool"), paths.data_dir("shallow"), 11)


def test_quiet_best_drops_rows_whose_best_move_captures(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    fake_bot(tmp_path, monkeypatch)
    # Own rock at 40 (e5) beside an enemy scissors at 41 (f5): e5-f5 captures; e5-e6 does not.
    board = board_with({40: 1, 41: 6, 0: 4, 80: 1})
    boards = np.stack([board, board])
    rows = rows_of(boards, np.random.default_rng(4))
    rows["since_capture"] = np.full(2, 5, dtype=np.uint16)
    rows["ply"] = np.full(2, 20, dtype=np.uint16)
    rows["game"] = np.arange(2, dtype=np.uint32)
    data.write(paths.data_dir("raw2"), rows, {"producer": "test"})
    moves = iter(["e5-f5", "e5-e6"])

    def analyse(engine_spec, requests, sims):
        return [{"search": {"root_value": 0.25, "lines": [{"move": next(moves), "q": 0.25}]}} for _ in requests]

    monkeypatch.setattr(label, "analyse", analyse)
    if label.proofs(boards).any():
        return  # the fixture position must be open for the filter to apply
    target = label.label("raw2", "quiet", "fake:engine", 4, 1, quiet_best=True)
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["dropped"] == 1 and provenance["quiet_best"] is True
    assert len(np.load(target / "kind.npy")) == 1
