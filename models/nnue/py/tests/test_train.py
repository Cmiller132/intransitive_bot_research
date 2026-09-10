"""A tiny run end to end: artifacts, export parity, exact resume, the loss
variants and the bucketed start from a one-head file."""

import csv

import engine
import numpy as np
import pytest
import torch

from nnue import data, export, paths
from nnue.features import ELAPSED_BASE, FEATURES, FORMAT6_FEATURES, PAD, feature_ids, orbit_hash, piece_bucket
from nnue.model import NNUE
from nnue.train import Config, initial_model, train

from .test_features import random_boards


def synthetic(directory, seed, rows):
    rng = np.random.default_rng(seed)
    board = random_boards(seed, rows, 2, 21)
    clock = rng.integers(50, 201, rows)
    since = (rng.random(rows) * clock).astype(np.int64)
    known = rng.random(rows) < 0.5
    data.write(
        directory,
        {
            "board": board,
            "since_capture": since,
            "ply": since,
            "capture_clock": clock,
            "target": rng.uniform(-1, 1, rows).astype(np.float32),
            "weight": rng.uniform(0.25, 1, rows).astype(np.float32),
            "kind": rng.integers(0, 2, rows),
            "outcome": rng.integers(-1, 2, rows) * known,
            "outcome_ok": known,
            "source": np.ones(rows),
            "game": np.arange(rows) // 4,
            "orbit": orbit_hash(board),
            "split": np.where(np.arange(rows) % 5 == 0, data.VALIDATION, data.TRAIN),
        },
        {"producer": "test"},
    )


@pytest.fixture
def sets(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    synthetic(paths.data_dir("a"), 1, 400)
    synthetic(paths.data_dir("b"), 2, 200)
    return [("a", 0.7), ("b", 0.3)]


def tiny(**overrides):
    base = dict(hidden=32, batch=64, steps_per_epoch=5, epochs=2, warmup_steps=2, threads=2, val_rows=64)
    return Config(**{**base, **overrides})


def test_run_writes_artifacts_and_the_export_matches(sets, tmp_path):
    out = train("tiny", sets, tiny())
    assert (out / "latest.pt").is_file() and (out / "best.pt").is_file() and (out / "best.nnue").is_file()
    with (out / "log.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and "a.mse" in rows[0] and "b.sparse.mse" in rows[0] or "b.middle.mse" in rows[0]
    best = torch.load(out / "best.pt", weights_only=False)
    model = NNUE(32)
    model.load_state_dict(best["model"])
    val = data.Dataset.open(paths.data_dir("a"), data.VALIDATION).all(50)
    ids = torch.from_numpy(feature_ids(val["board"], val["since_capture"], val["capture_clock"]))
    with torch.no_grad():
        expected = model(ids, torch.from_numpy(piece_bucket(val["board"])), qat=True).numpy()
    got = export.integer_eval(out / "best.nnue", val["board"], val["since_capture"], val["capture_clock"])
    assert np.max(np.abs(got - expected)) < 2e-6


def test_resume_reproduces_the_uninterrupted_run(sets, tmp_path):
    straight = train("straight", sets, tiny())
    first = train("split", sets, tiny(stop_epoch=1))
    resumed = train("split", sets, tiny(resume=str(first / "latest.pt")))
    assert resumed == first
    a = torch.load(straight / "latest.pt", weights_only=False)
    b = torch.load(first / "latest.pt", weights_only=False)
    assert a["epoch"] == b["epoch"] == 1 and a["step"] == b["step"]
    for key, value in a["model"].items():
        assert torch.equal(value, b["model"][key]), key
    with (straight / "log.csv").open() as f, (first / "log.csv").open() as g:
        assert [r["loss"] for r in csv.DictReader(f)] == [r["loss"] for r in csv.DictReader(g)]


def test_resume_rejects_changed_settings(sets):
    first = train("guard", sets, tiny(stop_epoch=1))
    with pytest.raises(ValueError):
        train("guard", sets, tiny(lr=1e-3, resume=str(first / "latest.pt")))


def test_loss_variants_and_bucketed_start(sets):
    train("bce", sets, tiny(loss="bce", outcome_weight=0.1))
    out = train("one", sets, tiny(epochs=1))
    wide = train("four", sets, tiny(buckets=4, init=str(out / "best.nnue"), epochs=1))
    assert export.read(wide / "best.nnue")["buckets"] == 4


def test_race_rows_train_only_with_the_flag_and_old_checkpoints_load(sets, tmp_path):
    out = train("six", sets, tiny(epochs=1))
    assert export.read(out / "best.nnue")["features"] == FORMAT6_FEATURES
    seven = train("seven", sets, tiny(init=str(out / "best.nnue"), race=True, epochs=1, lr=3e-2))
    net = export.read(seven / "best.nnue")
    assert net["features"] == FEATURES and net["weights"][FORMAT6_FEATURES:].any()
    with pytest.raises(ValueError):
        initial_model(tiny(init=str(seven / "best.nnue")))
    # A checkpoint from before the race rows has a 1,005-row table.
    state = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)["model"]
    old = {**state, "embedding.weight": state["embedding.weight"][: FORMAT6_FEATURES + 1].clone()}
    torch.save({"model": old}, tmp_path / "old.pt")
    model = initial_model(tiny(init=str(tmp_path / "old.pt"), race=True))
    assert model.features == FEATURES
    assert torch.equal(model.embedding.weight[:FORMAT6_FEATURES], state["embedding.weight"][:FORMAT6_FEATURES])
    assert not model.embedding.weight[FORMAT6_FEATURES:].any()


def test_clock_ablation_leaves_the_clock_rows_zero_and_inert(sets):
    out = train("noclock", sets, tiny(clock=False))
    weights = export.read(out / "best.nnue")["weights"]
    assert not weights[ELAPSED_BASE:PAD].any() and weights[:ELAPSED_BASE].any()
    board = np.frombuffer(engine.initial_board(), dtype=np.uint8)[None]
    early = export.integer_eval(out / "best.nnue", board, np.array([0]), np.array([200]))
    late = export.integer_eval(out / "best.nnue", board, np.array([150]), np.array([200]))
    assert early == late


def test_id_cache_reproduces_the_encoder(sets, tmp_path):
    from nnue.data import Dataset, encode_ids
    from nnue.train import encode

    directory = paths.data_dir("a")
    encode_ids(directory)
    cached = Dataset.open(directory, data.TRAIN)
    assert "ids" in cached.rows
    rows = cached.all(64)
    with_cache, _ = encode(rows)
    del rows["ids"]
    fresh, _ = encode(rows)
    assert torch.equal(with_cache, fresh)
    out = train("cached", sets, tiny(epochs=1))
    assert (out / "best.nnue").exists()
