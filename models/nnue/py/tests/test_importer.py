"""The conv window importer on a synthetic run: children through the engine,
episode recovery across windows, context merging and split hygiene."""

import json

import engine
import numpy as np
import pytest
import torch

from nnue import data, paths
from nnue.importer import Episodes, import_conv, merge_contexts

ENVS, STEPS, CANDIDATES = 4, 6, 16


def play_windows(rng, iterations):
    """Windows of `iterations` from ENVS random games played through the engine."""
    boards = [engine.initial_board() for _ in range(ENVS)]
    since = [0] * ENVS
    ply = [0] * ENVS
    windows = []
    for it in iterations:
        rows = {
            k: []
            for k in (
                "board",
                "since_capture",
                "ply",
                "clock",
                "action",
                "ret",
                "candidates",
                "candidate_q",
                "candidate_visited",
                "outcome",
                "outcome_ok",
            )
        }
        for _t in range(STEPS):
            for e in range(ENVS):
                legal = np.flatnonzero(engine.legal_mask(boards[e]))
                chosen = rng.choice(legal, min(CANDIDATES, len(legal)), replace=False)
                cands = np.full(CANDIDATES, -1, dtype=np.int16)
                cands[: len(chosen)] = chosen
                visited = np.zeros(CANDIDATES, dtype=bool)
                visited[: len(chosen)] = True
                q = rng.uniform(-1, 1, CANDIDATES).astype(np.float32)
                action = int(chosen[0])
                rows["board"].append(np.frombuffer(bytes(boards[e]), dtype=np.uint8).astype(np.int8))
                rows["since_capture"].append(since[e])
                rows["ply"].append(ply[e])
                rows["clock"].append(200)
                rows["action"].append(action)
                rows["ret"].append(rng.uniform(-1, 1))
                rows["candidates"].append(cands)
                rows["candidate_q"].append(q)
                rows["candidate_visited"].append(visited)
                rows["outcome"].append(rng.integers(-1, 2))
                rows["outcome_ok"].append(rng.random() < 0.5)
                child, s, p, outcome = engine.apply(boards[e], since[e], ply[e], action, 200)
                if outcome != 0 or rng.random() < 0.15:
                    boards[e], since[e], ply[e] = engine.initial_board(), 0, 0
                else:
                    boards[e], since[e], ply[e] = child, s, p
        payload = {"schema": 2, "envs": ENVS, "steps": STEPS, "iteration": it}
        for k, v in rows.items():
            payload[k] = torch.from_numpy(np.array(v))
        payload["since_capture"] = payload["since_capture"].to(torch.int16)
        payload["ply"] = payload["ply"].to(torch.int16)
        payload["clock"] = payload["clock"].to(torch.int16)
        payload["ret"] = payload["ret"].float()
        payload["outcome"] = payload["outcome"].to(torch.int8)
        windows.append(payload)
    return windows


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    rng = np.random.default_rng(3)
    run = tmp_path / "runs" / "toy"
    run.mkdir(parents=True)
    for payload in play_windows(rng, [4, 5, 6]):
        torch.save(payload, run / f"window_{payload['iteration']:06d}.pt")
    (run / "config.json").write_text(json.dumps({"rules": {"max_plies": 1000}}), encoding="utf-8")
    return run


def test_children_are_engine_children_and_labels_follow_the_design(run):
    windows = [torch.load(run / f"window_{it:06d}.pt", weights_only=True) for it in (4, 5, 6)]
    out = import_conv(run, "toy_set", None, 16, 1)
    for split in range(3):
        ds = data.Dataset.open(out, split)
        rows = ds.all()
        assert np.all(rows["capture_clock"] == 200) and np.all(rows["since_capture"] < 200)
        assert set(np.unique(rows["kind"])) <= {data.KIND_RETURN, data.KIND_CHILD}
        assert not np.any(rows["outcome_ok"] & (rows["kind"] == data.KIND_CHILD))
    train = data.Dataset.open(out, data.TRAIN).all()
    # Every child row is the engine's child of some root row of some window.
    seen = set()
    for w in windows:
        for i in range(ENVS * STEPS):
            board = w["board"][i].tolist()
            for j in range(CANDIDATES):
                if w["candidate_visited"][i, j]:
                    child, s, _, outcome = engine.apply(
                        board, int(w["since_capture"][i]), int(w["ply"][i]), int(w["candidates"][i, j]), 200
                    )
                    if outcome == 0:
                        seen.add((bytes(child), s))
    children = train["kind"] == data.KIND_CHILD
    for board, s in zip(train["board"][children], train["since_capture"][children], strict=True):
        assert (bytes(board.tolist()), int(s)) in seen
    assert children.sum() > 0 and (train["kind"] == data.KIND_RETURN).sum() > 0
    prov = json.loads((out / "provenance.json").read_text())
    assert [w["iteration"] for w in prov["windows"]] == [4, 5, 6] and prov["config"]["rules"]["max_plies"] == 1000


def test_no_orbit_crosses_splits(run):
    out = import_conv(run, "toy_split", None, 4, 1)
    splits = {s: data.Dataset.open(out, s).all() for s in range(3)}
    for held in (data.VALIDATION, data.TEST):
        others = np.concatenate([splits[s]["orbit"] for s in range(3) if s != held])
        assert not np.isin(splits[held]["orbit"], others).any()


def test_episodes_follow_the_ply_sequence_across_windows():
    episodes = Episodes(2)
    first = episodes.label(np.array([[0, 5], [1, 6], [2, 0]]), 4)
    assert first[:, 0].tolist() == [0, 0, 0] and first[0, 1] == 1 and first[2, 1] == 2
    second = episodes.label(np.array([[3, 1], [0, 2]]), 5)
    assert second[0, 0] == 0 and second[1, 0] == 3 and second[0, 1] == 2
    later = episodes.label(np.array([[1, 3]]), 9)
    assert later[0, 0] > 3 and later[0, 1] > 3


def test_merge_contexts_averages_by_weight_and_caps():
    board = np.zeros((3, 81), dtype=np.uint8)
    board[:, 0] = 1
    rows = {
        "board": board,
        "since_capture": np.array([1, 1, 2]),
        "ply": np.array([1, 1, 2]),
        "capture_clock": np.array([200, 200, 200]),
        "target": np.array([1.0, -1.0, 0.5], dtype=np.float32),
        "weight": np.array([3.0, 1.0, 9.0], dtype=np.float32),
        "kind": np.array([1, 0, 1], dtype=np.uint8),
        "outcome": np.array([0, 1, 0], dtype=np.int8),
        "outcome_ok": np.array([False, True, False]),
        "source": np.ones(3, dtype=np.uint8),
        "game": np.array([0, 0, 0]),
    }
    merged = merge_contexts(rows)
    assert len(merged["board"]) == 2
    first = merged["since_capture"] == 1
    assert merged["target"][first] == pytest.approx(0.5)
    assert merged["weight"][first] == 4.0 and merged["outcome_ok"][first] and merged["outcome"][first] == 1
    assert merged["kind"][first] == 0 and merged["weight"][~first] == 4.0


# --- self-play records (nnue.importer selfplay) ---------------------------------

SELFPLAY_MOVES = [
    "e3-f2",
    "h6-i6",
    "b5-a5",
    "h5-h6",
    "c5-d6",
    "e8-f9",
    "b4-a3",
    "e7-e6",
    "d6-c5",
    "g6-f5",
    "d4-e4",
    "g7-g6",
    "d2-e3",
    "f8-e7",
    "c5-b4",
    "i6-h7",
    "e2-f1",
    "e6-d7",
    "d3-d4",
    "h7-i8",
    "a5-a4",
    "h6-i5",
    "c4-c5",
    "f9-e8",
    "e4-d3",
    "d7-c6",
    "c5-c4",
    "e8-d7",
    "d4-e4",
    "f5-e5",
    "d3-d4",
    "c6-b5",
    "d4-e5",
    "e7-d6",
    "e3-f4",
    "g5-h6",
]  # a legal prefix of a generated game (turn-19 sample)
SELFPLAY_MOVES_B = [
    "e2-e1",
    "h5-i4",
    "c4-b3",
    "h6-h5",
    "b5-a4",
    "i4-i3",
    "b4-a3",
    "f8-e9",
    "b3-c4",
    "e9-d8",
    "c5-b4",
    "g7-h8",
    "e1-f2",
    "h5-i4",
    "d2-e2",
    "i4-h5",
    "e3-f3",
    "g5-h4",
    "f3-g2",
    "i3-i4",
    "d3-e4",
    "g6-g5",
    "a3-b3",
    "g5-f4",
    "d4-e3",
    "f4-f5",
    "e2-f3",
    "f5-g4",
    "f3-e2",
    "g4-g5",
    "c3-d4",
    "f7-e6",
    "e2-f3",
    "g5-g4",
    "f3-e2",
    "g4-g5",
]  # a second game's prefix


def selfplay_record(game_id: int, censored: bool, moves: list[str], pv: int = 0) -> dict:
    """A game record in the generator's schema: searched roots from ply 8 with
    scores from the mover's view (+300 for player 0, -120 for player 1), one
    unlabelled root at ply 20, one engine proof at ply 30. With `pv` > 0 every
    searched root carries the game's next `pv` moves as its line."""
    from nnue.games import replay

    roots_replay, _ = replay(moves)
    roots = []
    for _board, since, ply, _ in roots_replay[8:]:
        mover = ply % 2
        kind, score = "search", (300 if mover == 0 else -120)
        if ply == 20:
            kind, score = "unlabelled", None
        if ply == 30:
            kind, score = "engine_proof", 29999
        roots.append(
            {
                "ply": ply,
                "mover": mover,
                "since_capture": since,
                "capture_clock": 200,
                "board_fingerprint": 0,
                "searched_best": 0,
                "played_action": 0,
                "root_score": score,
                "score_kind": kind,
                "completed_depth": 6 if score is not None else 0,
                "nodes": 250000,
                "elapsed_ns": 1,
                **({"pv": [a for _, _, _, a in roots_replay[ply : ply + pv]]} if pv and kind == "search" else {}),
            }
        )
    return {
        "first": "nnue:a",
        "second": "nnue:a",
        "winner": None if censored else 0,
        "schema": 1,
        "seed": 1,
        "end": "MaxPlies" if censored else "Goal",
        "plies": len(moves),
        "moves": moves,
        "capture_clock": 200,
        "random_plies": list(range(8)) + [22, 31],
        "random_plan": [22, 31],
        "random_skipped": [],
        "game_id": game_id,
        "roots": roots,
        "opening_plies": 8,
        "censored": censored,
        "outcome": None if censored else 1,
        "outcome_after_ply": 32,
    }


def write_selfplay_records(directory, records: list[dict]) -> None:
    import gzip

    directory.mkdir(parents=True)
    with gzip.open(directory / "shard-000000.jsonl.gz", "wt", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record) + "\n")
    manifest = {
        "schema": 1,
        "model_sha256": "m",
        "binary_sha256": "b",
        "rules_sha256": "r",
        "capture_clock": 200,
        "repetition_draw": False,
        "score_scale": 600,
        "settings": {"nodes": 250000, "threads": 16},
        "shards": [{"file": "shard-000000.jsonl.gz", "first_game": 0}],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_selfplay_roots_become_labelled_rows_from_the_movers_view(tmp_path, monkeypatch):
    from nnue import data
    from nnue.games import replay
    from nnue.importer import import_selfplay

    monkeypatch.setattr("nnue.importer.data_dir", lambda name: tmp_path / "data" / name)
    write_selfplay_records(
        tmp_path / "gen", [selfplay_record(0, False, SELFPLAY_MOVES), selfplay_record(1, True, SELFPLAY_MOVES_B)]
    )
    target = import_selfplay([tmp_path / "gen"], "sp", 16, 3)
    rows = {name: np.load(target / f"{name}.npy") for name in data.FIELDS}
    n = len(rows["board"])
    # Plies 16..35 of two games, minus the unlabelled ply 20 of each: 38 rows, all distinct positions.
    assert n == 38 and rows["ply"].min() == 16 and rows["ply"].max() == 35 and 20 not in set(rows["ply"].tolist())
    by_game = {}
    for game, moves in ((0, SELFPLAY_MOVES), (1, SELFPLAY_MOVES_B)):
        boards, _ = replay(moves)
        by_game[game] = {ply: board for board, _, ply, _ in boards}
    for board, ply, game in zip(rows["board"], rows["ply"], rows["game"], strict=True):
        assert np.array_equal(board, by_game[int(game)][int(ply)])
    even, odd = rows["ply"] % 2 == 0, rows["ply"] % 2 == 1
    proof = rows["kind"] == data.KIND_PROOF
    assert np.allclose(rows["target"][even & ~proof], np.tanh(300 / 600))
    assert np.allclose(rows["target"][odd], np.tanh(-120 / 600))
    assert proof.sum() == 2 and np.all(rows["ply"][proof] == 30) and np.all(rows["target"][proof] > 0.999)
    assert np.all(rows["kind"][~proof] == data.KIND_TEACHER) and np.all(rows["source"] == data.SOURCE_SELFPLAY)
    real = rows["game"] == 0
    assert np.all(rows["outcome"][real & even] == 1) and np.all(rows["outcome"][real & odd] == -1)
    assert np.array_equal(rows["outcome_ok"][real], rows["ply"][real] >= 32)
    assert not rows["outcome_ok"][~real].any() and np.all(rows["outcome"][~real] == 0)
    assert set(rows["game"].tolist()) == {0, 1} and rows["capture_clock"].tolist() == [200] * n
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["counts"] == {
        "games": 2,
        "censored": 1,
        "roots": 56,
        "labelled": 54,
        "eligible": 38,
        "mismatched": 0,
        "unique": 38,
    }


def test_selfplay_interrupted_games_import_as_censored_without_an_outcome_ply(tmp_path, monkeypatch):
    """A game the generator preserved across an interruption (the 2026-09-15 reboot): end Interrupted, no outcome
    and outcome_after_ply null. Its searched roots are rows with outcome_ok False; the import must not crash."""
    from nnue import data
    from nnue.importer import import_selfplay

    monkeypatch.setattr("nnue.importer.data_dir", lambda name: tmp_path / "data" / name)
    interrupted = selfplay_record(1, True, SELFPLAY_MOVES_B)
    interrupted.update(end="Interrupted", outcome_after_ply=None)
    write_selfplay_records(tmp_path / "gen", [selfplay_record(0, False, SELFPLAY_MOVES), interrupted])
    target = import_selfplay([tmp_path / "gen"], "sp", 16, 3)
    rows = {name: np.load(target / f"{name}.npy") for name in data.FIELDS}
    cut = rows["game"] == 1
    assert cut.sum() == 19 and not rows["outcome_ok"][cut].any() and np.all(rows["outcome"][cut] == 0)
    assert rows["outcome_ok"][~cut].any()
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["counts"]["censored"] == 1 and provenance["counts"]["games"] == 2


def test_selfplay_line_rows_follow_the_principal_variation(tmp_path, monkeypatch):
    """--pv-rows K: the first K positions of a root's line become rows with the root's value sign-flipped per
    ply (kind TEACHER, source SOURCE_SELFPLAY_LINE, weight 0.5, no outcome); a line position that is also a
    searched root keeps the root's label; --pv-rows 0 ignores recorded lines (the A/B on identical games)."""
    import engine

    from nnue import data
    from nnue.games import replay
    from nnue.importer import import_selfplay

    monkeypatch.setattr("nnue.importer.data_dir", lambda name: tmp_path / "data" / name)
    write_selfplay_records(
        tmp_path / "gen",
        [selfplay_record(0, False, SELFPLAY_MOVES, pv=3), selfplay_record(1, True, SELFPLAY_MOVES_B, pv=3)],
    )
    roots_only = import_selfplay([tmp_path / "gen"], "roots", 16, 3, pv_rows=0)
    with_lines = import_selfplay([tmp_path / "gen"], "lines", 16, 3, pv_rows=3)
    a = {name: np.load(roots_only / f"{name}.npy") for name in data.FIELDS}
    b = {name: np.load(with_lines / f"{name}.npy") for name in data.FIELDS}
    assert len(a["board"]) == 38 and not (a["source"] == data.SOURCE_SELFPLAY_LINE).any() and np.all(a["weight"] == 1)
    line = b["source"] == data.SOURCE_SELFPLAY_LINE
    for name in data.FIELDS:  # the root rows are the roots-only import, in its order
        assert np.array_equal(a[name], b[name][~line]), name
    # Every line position from plies 16..35 is a searched root (a duplicate, the root's row kept) except ply 20
    # (the unlabelled root) and ply 36 (past the last root): two line rows per game.
    assert sorted(zip(b["game"][line].tolist(), b["ply"][line].tolist(), strict=True)) == [
        (0, 20),
        (0, 36),
        (1, 20),
        (1, 36),
    ]
    assert np.all(b["kind"][line] == data.KIND_TEACHER) and np.all(b["weight"][line] == 0.5)
    assert not b["outcome_ok"][line].any() and np.all(b["outcome"][line] == 0)
    # The first line row at a position wins among lines: ply 20 comes from root 17 (mover 1, -120) three plies
    # on, ply 36 from root 33 (the same): tanh(-120 / 600) * (-1)^3.
    assert np.allclose(b["target"][line], np.tanh(120 / 600))
    for game, moves in ((0, SELFPLAY_MOVES), (1, SELFPLAY_MOVES_B)):
        boards, _ = replay(moves)
        by_ply = {ply: board for board, _, ply, _ in boards}
        board35, since35, _, action35 = boards[35]
        child, _, _, _ = engine.apply(board35.tolist(), since35, 35, action35, 200)
        by_ply[36] = np.array(list(child), dtype=np.uint8)
        for board, ply, g in zip(b["board"][line], b["ply"][line], b["game"][line], strict=True):
            if int(g) == game:
                assert np.array_equal(board, by_ply[int(ply)]), (game, ply)
    provenance = json.loads((with_lines / "provenance.json").read_text(encoding="utf-8"))
    counts = provenance["counts"]
    # 16 searched roots at plies 16..33 with three line positions each, root 34 with two, root 35 with one
    assert counts["line_rows"] == 2 * (16 * 3 + 2 + 1) and counts["line_duplicates"] == counts["line_rows"] - 4
    assert counts["line_terminal"] == 0 and counts["line_illegal"] == 0 and counts["unique"] == 42
    assert provenance["pv_rows"] == 3 and provenance["pv_weight"] == 0.5 and "7" in provenance["sources"]
    plain = json.loads((roots_only / "provenance.json").read_text(encoding="utf-8"))
    assert "pv_rows" not in plain and "line_rows" not in plain["counts"] and "7" not in plain["sources"]
