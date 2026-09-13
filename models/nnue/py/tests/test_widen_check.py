"""The widening preflight's structural, QAT and evidence failure gates on toy inputs."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from nnue import widen_check as w
from nnue.export import export, integer_eval
from nnue.features import feature_ids
from nnue.model import NNUE
from nnue.train import Config, initial_model

from .test_export import positions, trained_like


@pytest.fixture
def case(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    path = tmp_path / "parent.nnue"
    export(trained_like(NNUE(32, 6), 4), path)
    config = Config(hidden=64, version=6, init=str(path), seed=2, batch=8, qat_start_epoch=0)
    base = initial_model(replace(config, hidden=32))
    wide = initial_model(config)
    board, since, clock = positions(13, 64)
    ids8 = feature_ids(board, since, clock)
    ids = torch.from_numpy(wide.layout.rows(ids8))
    rows = {
        "ids": ids8[:8],
        "target": np.linspace(-0.8, 0.8, 8, dtype=np.float32),
        "weight": np.ones(8, dtype=np.float32),
    }
    return base, wide, config, ids, rows


def test_initial_gates_cover_every_channel_and_preserved_head(case, tmp_path):
    base, wide, config, ids, _ = case
    result, arrays = w.initial_checks(base, wide, config, ids)
    assert all(result["checks"].values())
    assert arrays["initial_columns"].shape == (32, 1004)
    with torch.no_grad():
        wide.piece[:, 33].copy_(wide.piece[:, 32])
        wide.attack[:, 33].copy_(wide.attack[:, 32])
        wide.clock[:, 33].copy_(wide.clock[:, 32])
    broken, _ = w.initial_checks(base, wide, config, ids)
    assert not broken["checks"]["distinct_nonzero_columns"]
    assert not broken["checks"]["distinct_varying_activations"]
    with torch.no_grad():
        wide.output.weight[0, 32] = 0.02
        wide.dense.weight[0, 0] += 0.02
    broken, _ = w.initial_checks(base, wide, config, ids)
    assert not broken["checks"]["zero_outgoing"] and not broken["checks"]["preserved"]


def test_export_width_parity_on_integer_oracle(case, tmp_path):
    base, wide, _, _, _ = case
    paths = [tmp_path / "a.nnue", tmp_path / "b.nnue"]
    for model, path in zip((base, wide), paths, strict=True):
        export(model, path)
    args = positions(9, 64)
    assert np.array_equal(integer_eval(paths[0], *args), integer_eval(paths[1], *args))


class Batches:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def next(self):
        self.calls += 1
        return self.rows


def test_fixed_probe_crosses_then_learns_reproducibly(case):
    base, wide, config, ids, rows = case
    config = replace(config, lr=0.02, warmup_steps=0)
    batches = Batches(rows)
    result, arrays = w.probe(wide, base.hidden, config, batches, ids)
    assert batches.calls == result["updates"] == 256
    assert all(result["checks"].values())
    assert result["first_quantized_outgoing_update"] < result["first_subsequent_incoming_backward"] <= 256
    assert any(v > 0 for v in result["max_incoming_task_gradient_norm"])
    again, arrays2 = w.probe(initial_model(config), base.hidden, config, Batches(rows), ids)
    assert again == result
    assert all(np.array_equal(v, arrays2[k]) for k, v in arrays.items())


def test_weight_decay_cannot_pass_incoming_task_gradient_gate(case):
    base, wide, config, ids, rows = case
    before = w.incoming(wide, base.hidden).clone()
    config = replace(config, lr=1e-6, warmup_steps=0, weight_decay=100)
    result, _ = w.probe(wide, base.hidden, config, Batches(rows), ids)
    assert result["updates"] == 256
    assert result["first_quantized_outgoing_update"] is None
    assert result["first_subsequent_incoming_backward"] is None
    assert not any(result["max_incoming_task_gradient_norm"])
    assert not torch.equal(before, w.incoming(wide, base.hidden))
    assert not result["checks"]["quantization_crossing"]


def test_configs_reject_recipe_drift():
    train = {"init": "parent.nnue", "config": {"hidden": 512, "version": 6}, "data": [["one", 1.0]]}
    m = {"arms": [{"train": train}, {"train": copy.deepcopy(train)}]}
    m["arms"][1]["train"]["config"]["hidden"] = 1024
    low, high, _ = w.configs(m)
    assert (low.hidden, high.hidden) == (512, 1024)
    for key, value in [("seed", 3), ("version", 8), ("device", "cuda"), ("resume", "old.pt")]:
        bad = copy.deepcopy(m)
        bad["arms"][1]["train"]["config"][key] = value
        with pytest.raises(ValueError, match="matched"):
            w.configs(bad)


def test_rust_reader_rejects_incomplete_or_reordered_output(monkeypatch, tmp_path):
    def command(args, **kwargs):
        assert args[1:3] == ["nnue", "validate"]
        kwargs["stdout"].write(json.dumps({"row": 1, "raw": 0.5}) + "\n")

    monkeypatch.setattr(w.subprocess, "run", command)
    with pytest.raises(ValueError, match="incomplete or reordered"):
        w.rust_values(tmp_path / "bot", tmp_path / "net", tmp_path / "positions", tmp_path / "out", 1)


def test_binding_rejects_changed_init(monkeypatch, tmp_path):
    monkeypatch.setattr(w, "workspace_root", lambda: tmp_path)
    for name in ("manifest", "parent", "sidecar", "bot", "fixture"):
        (tmp_path / name).write_text(name)
    monkeypatch.setattr(w, "FIXTURE", "fixture")
    m = {
        "parent": "parent",
        "parent_record": {"path": "sidecar", "sha256": w.sha256(tmp_path / "sidecar")},
        "init_sha256": w.sha256(tmp_path / "parent"),
        "bot": "bot",
        "bot_sha256": w.sha256(tmp_path / "bot"),
        "data_inventory": [],
    }
    first = w.binding(tmp_path / "manifest", m, [])
    assert first[str(tmp_path / "parent")] == m["init_sha256"]
    (tmp_path / "parent").write_text("changed")
    with pytest.raises(ValueError, match="input identity differs"):
        w.binding(tmp_path / "manifest", m, [])


def test_probe_nonfinite_is_failure(case):
    base, wide, config, ids, rows = case
    rows["target"][0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        w.probe(wide, base.hidden, config, Batches(rows), ids)


@pytest.mark.parametrize("drift", [False, True])
def test_report_binds_evidence_and_refuses_overwrite(case, tmp_path, monkeypatch, drift):
    base, _, config, _, rows = case
    config = replace(config, lr=0.02, warmup_steps=0)
    manifest = tmp_path / "experiment.json"
    manifest.write_text(json.dumps({"parent": config.init, "bot": "scripted-validator"}))
    monkeypatch.setattr(w, "configs", lambda _: (replace(config, hidden=base.hidden), config, [["toy", 1.0]]))
    calls = []

    def binding(*args):
        calls.append(1)
        return {"source": str(len(calls)) if drift else "unchanged"}

    monkeypatch.setattr(w, "binding", binding)

    class Dataset:
        def __init__(self):
            self.rows = rows

        def __len__(self):
            return len(rows["target"])

    monkeypatch.setattr(w.Dataset, "open", lambda *args: Dataset())
    monkeypatch.setattr(w, "Mixture", lambda *args: Batches(rows))

    def rust(bot, net, positions, out, count):
        values = [json.loads(line)["raw6"] for line in positions.read_text().splitlines()]
        assert len(values) == count == 512
        out.write_text("scripted validation\n")
        return np.array(values)

    monkeypatch.setattr(w, "rust_values", rust)
    report = w.run(manifest)
    assert report["passed"] is not drift
    assert bool(report["errors"]) is drift
    assert report["manifest_sha256"] == w.sha256(manifest)
    directory = tmp_path / "preflight"
    for name, digest in report["artifacts"].items():
        assert w.sha256(directory / name) == digest
    original = (directory / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        w.run(manifest)
    assert (directory / "report.json").read_bytes() == original
    torch.set_num_threads(1)
