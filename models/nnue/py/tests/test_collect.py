"""The student collector on games played by a fake `bot` (a scripted record
through the engine), so the replay, the value-drop alternatives, the leaf
reservoir and the sampling shares are checked without the binary."""

import json

import engine
import numpy as np
import pytest

from nnue import collect, data, features, games, paths

SQUARES = [f"{'abcdefghi'[s % 9]}{s // 9 + 1}" for s in range(81)]


def spell(action: int, red_to_move: bool) -> str:
    """The engine's canonical action as an absolute token (the inverse of
    `games.parse_action`, derived independently here)."""
    start = action % 81
    d_rank, d_file = games.DIRS[action // 81]
    end = (start // 9 + d_rank) * 9 + start % 9 + d_file
    if red_to_move:
        start, end = int(features.ANTI[start]), int(features.ANTI[end])
    return f"{SQUARES[start]}-{SQUARES[end]}"


def random_game(rng, plies):
    """Moves, the roots the engine saw, and the outcome (0 ongoing, 1 win, 2 draw)."""
    board, since, ply = list(engine.initial_board()), 0, 0
    moves, roots, outcome = [], [], 0
    for _ in range(plies):
        legal = np.flatnonzero(engine.legal_mask(board))
        action = int(rng.choice(legal))
        roots.append((np.array(board, dtype=np.uint8), since, ply, action))
        moves.append(spell(action, red_to_move=ply % 2 == 1))
        child, since, ply, outcome = engine.apply(board, since, ply, action, games.SITE_CLOCK)
        board = list(child)
        if outcome:
            break
    return moves, roots, outcome


def test_replay_reproduces_the_engine_roots():
    rng = np.random.default_rng(3)
    for _ in range(5):
        moves, roots, _ = random_game(rng, 60)
        replayed, _ = games.replay(moves)
        assert len(replayed) == len(roots)
        for (board, since, ply, action), (b2, s2, p2, a2) in zip(roots, replayed, strict=True):
            assert (board == b2).all() and (since, ply, action) == (s2, p2, a2)


def test_parse_action_accepts_capture_and_piece_prefixes():
    assert games.parse_action("Rb4xa4", False) == games.parse_action("b4-a4", False)
    assert games.parse_action("b4a4", False) == games.parse_action("b4-a4", False)


def test_alternatives_exclude_the_played_move_and_finished_games():
    board = np.frombuffer(engine.initial_board(), dtype=np.uint8)
    legal = np.flatnonzero(engine.legal_mask(board.tolist()))
    played = int(legal[0])
    children = collect.alternatives(board, 0, 0, played)
    assert len(children) == len(legal) - 1
    child, since, ply = children[0]
    assert child.shape == (81,) and (since, ply) == (1, 1)


def test_value_drops_find_the_mover_whose_value_fell():
    stats = [None] * 4 + [{"value": v} for v in (0.5, 0.0, 0.1, 0.0, -0.3, 0.0)]
    record = {"moves": ["x"] * 10, "stats": stats}
    # Values are indexed after the four opening plies: 0.5 -> 0.1 (ply 4) and 0.1 -> -0.3 (ply 6).
    assert collect.value_drops(record) == {4, 6}


def test_schedule_alternates_seats_budgets_and_traces():
    games = collect.schedule("s", "p", 6)
    assert len(games) == 12
    assert [g.first for g in games[:2]] == ["nnue:s", "nnue:p"]
    assert {g.budget for g in games} == set(collect.BUDGETS)
    assert [g.family for g in games if g.leaves] == [0, 3] or [g.family for g in games if g.leaves] == [0]
    solo = collect.schedule("s", None, 3)
    assert all(g.first == g.second == "nnue:s" for g in solo)
    assert all(a.budget != b.budget for a, b in zip(solo[::2], solo[1::2], strict=True))
    timed = collect.schedule("s", "sq:w.onnx", 4, move_ms=100)
    assert [g.second for g in timed[:2]] == ["sq:w.onnx", "nnue:s"]
    assert all(g.move_ms == 100 and g.budget == 0 for g in timed)
    assert [g.family for g in timed if g.leaves] == [0]


@pytest.fixture
def fake_bot(monkeypatch, tmp_path):
    """`collect.play` answers from scripted random games; traced games get a leaves file."""
    rng = np.random.default_rng(11)

    def play(game, seed, out):
        moves, roots, outcome = random_game(rng, 80)
        values = []
        for i in range(len(moves)):
            values.append({"value": float(0.6 if i % 4 == 0 else -0.2), "sims": game.budget})
        winner = None if outcome == 2 else ((len(moves) - 1) % 2 if outcome == 1 else None)
        record = {"winner": winner, "moves": moves, "stats": values, "plies": len(moves), "end": "Goal"}
        leaves = None
        if game.leaves:
            leaves = out / f"leaves_{game.family}.jsonl"
            with leaves.open("w", encoding="utf-8") as sink:
                for board, since, ply, _ in roots:
                    for _ in range(3):
                        sink.write(json.dumps({"board": board.tolist(), "since_capture": since, "ply": ply}) + "\n")
        return record, leaves

    monkeypatch.setattr(collect, "play", play)
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    (tmp_path / "a.nnue").write_bytes(b"not a network; only hashed")
    return tmp_path


def test_collect_writes_a_raw_dataset_with_the_declared_shares(fake_bot):
    target = collect.collect("round", fake_bot / "a.nnue", None, families=4, states=200, workers=2, seed=1)
    for name in ("board", "target", "kind", "source", "outcome_ok"):
        assert (target / f"{name}.npy").exists()
    rows = data.Dataset.open(target, split=data.TRAIN).all()
    kind, source = np.load(target / "kind.npy"), np.load(target / "source.npy")
    assert (kind == data.KIND_RAW).all() and (source == data.SOURCE_STUDENT).all()
    assert (np.load(target / "target.npy") == 0).all()
    assert (np.load(target / "capture_clock.npy") == games.SITE_CLOCK).all()
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["games"] == 8 and provenance["pools"]["leaf"] > 0
    assert provenance["pools"]["alternative"] > 0 and provenance["pools"]["targeted"] > 0
    assert 0 < provenance["sampled"] <= 200
    outcome_ok = np.load(target / "outcome_ok.npy")
    assert outcome_ok.any() and not outcome_ok.all()
    assert not list((fake_bot / "runs" / "nnue_collect" / "round").glob("leaves_*"))
    assert (fake_bot / "runs" / "nnue_collect" / "round" / "games.jsonl").exists()
    assert len(rows["board"]) > 0
