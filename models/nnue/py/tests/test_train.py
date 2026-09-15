"""A tiny run end to end: artifacts, export parity, exact resume, the loss
variants and the bucketed start from a one-head file."""

import csv
import shutil

import numpy as np
import pytest
import torch

from nnue import data, export, paths
from nnue.features import FORMAT8, feature_ids, orbit_hash
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
        expected = model(ids, qat=True).numpy()
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


def test_format8_from_a_format6_init_trains_the_context_rows(sets, tmp_path):
    out = train("six", sets, tiny(epochs=1, version=6))
    assert export.read(out / "best.nnue")["version"] == 6
    eight = train("eight", sets, tiny(init=str(out / "best.nnue"), epochs=1, lr=3e-2))
    net = export.read(eight / "best.nnue")
    rows = net["weights"][: FORMAT8.attack_base].reshape(27, 486, 32)
    assert net["version"] == 8 and (rows != rows[0]).any()
    with pytest.raises(ValueError):
        initial_model(tiny(init=str(eight / "best.nnue"), version=6))
    # A checkpoint of the other format does not load: the parameters differ.
    with pytest.raises(RuntimeError):
        initial_model(tiny(init=str(out / "latest.pt")))


def test_id_cache_reproduces_the_encoder(sets, tmp_path):
    from nnue.data import Dataset, encode_ids
    from nnue.train import encode

    directory = paths.data_dir("a")
    encode_ids(directory)
    cached = Dataset.open(directory, data.TRAIN)
    assert "ids" in cached.rows
    # A cache is bound to its dataset and the encoder: a copy under another set or without its binding is refused.
    other = paths.data_dir("b")
    for name in ("ids8.npy", "ids8.json"):
        shutil.copy2(directory / name, other / name)
    with pytest.raises(ValueError, match="not this dataset"):
        Dataset.open(other, data.TRAIN)
    (other / "ids8.json").unlink()
    with pytest.raises(ValueError, match="not this dataset"):
        Dataset.open(other, data.TRAIN)
    (other / "ids8.npy").unlink()
    rows = cached.all(64)
    with_cache = encode(rows, FORMAT8)
    del rows["ids"]
    fresh = encode(rows, FORMAT8)
    assert torch.equal(with_cache, fresh)
    out = train("cached", sets, tiny(epochs=1))
    assert (out / "best.nnue").exists()


def test_ema_exports_the_average_and_keeps_the_live_weights(sets, tmp_path):
    """--ema: the checkpoint's model is the average (begun at the end of the warmup), the live weights ride
    along, and best.nnue is the export of the average; without --ema nothing changes."""
    out = train("ema", sets, tiny(ema=0.9))
    latest = torch.load(out / "latest.pt", weights_only=False)
    assert "live" in latest and latest["ema_started"] == 2 and latest["config"]["ema"] == 0.9
    assert any(not torch.equal(latest["model"][k], latest["live"][k]) for k in latest["model"])
    best = torch.load(out / "best.pt", weights_only=False)
    average = NNUE(32)
    average.load_state_dict(best["model"])
    export.export(average, tmp_path / "average.nnue", {})
    assert (tmp_path / "average.nnue").read_bytes() == (out / "best.nnue").read_bytes()
    plain = torch.load(train("plain", sets, tiny()) / "latest.pt", weights_only=False)
    assert "live" not in plain and "ema_started" not in plain and plain["config"]["ema"] == 0.0


def test_ema_resume_reproduces_the_uninterrupted_run(sets):
    straight = train("ema_straight", sets, tiny(ema=0.9))
    first = train("ema_split", sets, tiny(ema=0.9, stop_epoch=1))
    assert train("ema_split", sets, tiny(ema=0.9, resume=str(first / "latest.pt"))) == first
    a = torch.load(straight / "latest.pt", weights_only=False)
    b = torch.load(first / "latest.pt", weights_only=False)
    assert a["epoch"] == b["epoch"] == 1 and a["ema_started"] == b["ema_started"] == 2
    for key in ("model", "live"):
        for name, value in a[key].items():
            assert torch.equal(value, b[key][name]), (key, name)


def test_resume_across_the_ema_fields_introduction(sets):
    """A checkpoint written before the field existed resumes: the missing field reads as its default."""
    first = train("old", sets, tiny(stop_epoch=1))
    saved = torch.load(first / "latest.pt", weights_only=False)
    del saved["config"]["ema"]
    torch.save(saved, first / "latest.pt")
    train("old", sets, tiny(resume=str(first / "latest.pt")))
