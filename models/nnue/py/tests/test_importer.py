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
