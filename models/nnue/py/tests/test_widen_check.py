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


@pytest.fixture(params=[6, 8])
def case(tmp_path, request):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    path = tmp_path / "parent.nnue"
    net = trained_like(NNUE(32, request.param), 4)
    board, since, clock = positions(13, 64)
    train_ids = torch.from_numpy(net.layout.rows(feature_ids(board[:8], since[:8], clock[:8])))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (net(train_ids) - torch.linspace(-0.8, 0.8, 8)).square().mean().backward()
        optimizer.step()
        net.constrain()
    export(net, path)
    config = Config(
        hidden=64, version=request.param, init=str(path), seed=2, batch=8, qat_start_epoch=1, steps_per_epoch=1000
    )
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
    config = replace(config, lr=0.001, warmup_steps=0)
    batches = Batches(rows)
    result, arrays = w.probe(wide, base.hidden, config, batches, ids)
    assert batches.calls == result["batches"] == 1001
    assert result["updates"] == 1000
    assert result["samples"][-1]["qat"] and not result["samples"][-1]["updated"]
    assert all(result["checks"].values())
    assert result["first_quantized_outgoing_update"] <= 1000
    assert result["first_float_incoming_backward"] <= 1000
    assert any(v > 0 for v in result["max_shared_task_gradient_norm"])
    again, arrays2 = w.probe(initial_model(config), base.hidden, config, Batches(rows), ids)
    assert again == result
    assert all(np.array_equal(v, arrays2[k]) for k, v in arrays.items())


def test_weight_decay_cannot_pass_incoming_task_gradient_gate(case):
    base, wide, config, ids, rows = case
    before = w.incoming(wide, base.hidden).clone()
    config = replace(config, lr=1e-6, warmup_steps=0, weight_decay=100)
    result, _ = w.probe(wide, base.hidden, config, Batches(rows), ids)
    assert result["updates"] == 1000
    assert result["first_quantized_outgoing_update"] is None
    assert not result["checks"]["qat_incoming_task_gradient"]
    assert not torch.equal(before, w.incoming(wide, base.hidden))
    assert not result["checks"]["served_readout_at_transition"]


def test_seeded_outgoing_mode_gates(case):
    base, _, config, ids, rows = case
    config = replace(config, widen_outgoing=1 / 64, lr=0.001, warmup_steps=0)
    seeded = initial_model(config)
    result, _ = w.initial_checks(base, seeded, config, ids)
    assert "outgoing_seeded" in result["checks"] and "zero_outgoing" not in result["checks"]
    assert all(result["checks"].values())
    with torch.no_grad():
        seeded.dense.weight[0, base.hidden] = 1 / 64
    broken, _ = w.initial_checks(base, seeded, config, ids)
    assert not broken["checks"]["outgoing_seeded"]
    with torch.no_grad():
        seeded.dense.weight[0, base.hidden] = 0
        seeded.output.weight[0, base.hidden] = 0.02
    broken, _ = w.initial_checks(base, seeded, config, ids)
    assert not broken["checks"]["outgoing_seeded"]
    probe, _ = w.probe(initial_model(config), base.hidden, config, Batches(rows), ids)
    assert probe["first_quantized_outgoing_update"] == 1
    assert probe["checks"]["served_readout_at_transition"] and probe["checks"]["qat_incoming_task_gradient"]


def test_configs_reject_recipe_drift():
    train = {
        "init": "parent.nnue",
        "config": {"hidden": 512, "version": 6, "qat_start_epoch": 1, "steps_per_epoch": 1000},
        "data": [["one", 1.0]],
    }
    m = {"arms": [{"train": train}, {"train": copy.deepcopy(train)}]}
    m["arms"][1]["train"]["config"]["hidden"] = 1024
    low, high, _ = w.configs(m)
    assert (low.hidden, high.hidden) == (512, 1024)
    for key, value in [("seed", 3), ("version", 8), ("device", "cuda"), ("resume", "old.pt"), ("qat_start_epoch", 0)]:
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
    config = replace(config, lr=0.001, warmup_steps=0)
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
        values = [
            json.loads(line)["raw" if config.version == 8 else "raw6"] for line in positions.read_text().splitlines()
        ]
        assert len(values) == count == 512
        out.write_text("scripted validation\n")
        return np.array(values)

    monkeypatch.setattr(w, "rust_values", rust)
    report = w.run(manifest)
    assert report["passed"] is not drift
    assert bool(report["errors"]) is drift
    assert report["manifest_sha256"] == w.sha256(manifest)
    directory = tmp_path / "preflight"
    assert (directory / "experiment.json").read_bytes() == manifest.read_bytes()
    assert "transition.nnue" in report["artifacts"]
    assert "transition.nnue.json" in report["artifacts"]
    assert report["version"] == config.version
    for name, digest in report["artifacts"].items():
        assert w.sha256(directory / name) == digest
    original = (directory / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        w.run(manifest)
    assert (directory / "report.json").read_bytes() == original
    torch.set_num_threads(1)


@pytest.mark.parametrize("steps,qat", [(999, 1), (1001, 1), (1000, 0), (1000, 2)])
def test_configs_require_exact_warmin(steps, qat):
    train = {
        "init": "parent.nnue",
        "config": {"hidden": 512, "version": 8, "steps_per_epoch": steps, "qat_start_epoch": qat},
        "data": [["one", 1.0]],
    }
    other = copy.deepcopy(train)
    other["config"]["hidden"] = 1024
    with pytest.raises(ValueError, match="1000-update"):
        w.configs({"arms": [{"train": train}, {"train": other}]})


def test_context_preservation_and_zero_residual_gates(case):
    base, wide, config, ids, _ = case
    if wide.context is None:
        assert w.residuals(wide, base.hidden).shape == (0, 32, 486)
        return
    assert base.context.count_nonzero() > 0
    assert torch.equal(base.context, wide.context[..., : base.hidden])
    assert w.residuals(wide, base.hidden).count_nonzero() == 0
    with torch.no_grad():
        wide.context[4, 20, 0] += 0.1
        wide.context[5, 30, base.hidden] = 0.1
    result, _ = w.initial_checks(base, wide, config, ids)
    assert not result["checks"]["preserved"]
    assert not result["checks"]["zero_new_residuals"]


def test_qat_backward_does_not_update_and_records_unvisited_contexts(case, monkeypatch):
    base, wide, config, ids, rows = case
    config = replace(config, lr=0.001, warmup_steps=0)
    # One fixed context per perspective also remains fixed under this scripted loss.
    single = torch.from_numpy(wide.layout.rows(rows["ids"][:1]))
    calls = []
    snapshot = {}

    def loss(model, rows, table, config, qat, rng):
        calls.append(qat)
        if qat:
            # A controlled active path isolates gradient coverage from convergence.
            with torch.no_grad():
                for parameter in (model.piece, model.attack, model.clock, model.context):
                    if parameter is not None:
                        parameter[..., base.hidden :].zero_()
                model.bias[base.hidden :].fill_(0.15)
                for start, end in [(base.hidden, model.hidden), (model.hidden + base.hidden, 2 * model.hidden)]:
                    model.output.weight[:, start:end].fill_(0.125)
                    model.dense.weight[:, start:end].zero_()
            snapshot.update({k: v.detach().clone() for k, v in model.state_dict().items()})
        return (model(single, qat=qat) - (-0.5 if qat else 0.5)).square().mean(), {}

    monkeypatch.setattr(w, "step_loss", loss)
    result, arrays = w.probe(wide, base.hidden, config, Batches(rows), ids)
    assert calls == [False] * 1000 + [True]
    assert all(torch.equal(v, snapshot[k]) for k, v in wide.state_dict().items())
    assert sum(result["qat_context_perspective_visits"]) == 2
    assert sum(result["context_perspective_visits"]) == 2002
    if wide.context is not None:
        visits = np.array(result["qat_context_perspective_visits"])
        residual = arrays["qat_residual_task_gradient"]
        assert np.count_nonzero(visits) <= 2
        assert np.count_nonzero(residual[visits == 0]) == 0
        assert np.count_nonzero(residual[visits > 0]) > 0


def test_historical_crossing_is_not_transition_success(case, monkeypatch):
    base, wide, config, ids, rows = case
    config = replace(config, lr=0.001, warmup_steps=0)
    constrain = wide.constrain
    calls = 0

    def constrain_then_zero():
        nonlocal calls
        calls += 1
        constrain()
        if calls == 1000:
            with torch.no_grad():
                for layer in (wide.output, wide.dense):
                    layer.weight[:, base.hidden : wide.hidden].zero_()
                    layer.weight[:, wide.hidden + base.hidden :].zero_()

    monkeypatch.setattr(wide, "constrain", constrain_then_zero)
    result, _ = w.probe(wide, base.hidden, config, Batches(rows), ids)
    assert result["first_quantized_outgoing_update"] is not None
    assert not result["checks"]["served_readout_at_transition"]
    assert not result["checks"]["qat_incoming_task_gradient"]


@pytest.mark.parametrize("version", [6, 8])
def test_configs_accept_supported_formats(version):
    train = {
        "init": "parent.nnue",
        "config": {"hidden": 512, "version": version, "steps_per_epoch": 1000, "qat_start_epoch": 1},
        "data": [["one", 1.0]],
    }
    other = copy.deepcopy(train)
    other["config"]["hidden"] = 1024
    assert w.configs({"arms": [{"train": train}, {"train": other}]})[0].version == version
