"""CPU checks for lossless control features and their frozen evidence bindings."""

import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conv.config import Config
from conv.control import State
from conv.fit_control import (
    BATCH,
    FEATURES,
    capture_input,
    comparison,
    extract,
    feature_array,
    feature_tensor,
    main,
    numerical_failures,
    source_data,
)
from conv.probe_control import CandidatePool, state_key, state_record
from conv.probe_parity import SOURCES, VARIANTS, VERSION, digest, pack_arrays, unpack_arrays


def parity_fixture():
    cfg = Config()
    cfg.rules.clock_min = cfg.rules.clock_max = 200
    board = np.zeros(81, np.int8)
    board[10], board[70] = 1, 4
    batch = [state_record(State(board.copy(), i % 100, i % 900, 200)) for i in range(BATCH)]
    plan, units, pools = [], {}, {}

    def add(spec, result):
        plan.append(spec)
        units[spec["id"]] = dict(spec_sha256=digest(spec), sha256=digest(result), result=result)

    for identity in ("discovery0", "discovery1"):
        pool = CandidatePool(1, identity)
        for indices in ((0, 1), (0,)):
            pool.offer(
                [
                    dict(state=batch[i], source="tree", occurrence=[0, i], rank=float(i), regret=float(i) / 2)
                    for i in indices
                ]
            )
        pools[identity] = dict(snapshot=pool.snapshot())
        add(
            dict(id=f"C/pool/{identity}/0", variant="C", kind="pool", pool=identity, start=0, stop=2),
            dict(arrays=[pack_arrays(dict(scores=np.zeros((2, 2), np.float32)))]),
        )
    request = dict(
        version=VERSION,
        checkpoint_sha256="frozen",
        config=asdict(cfg),
        batch=batch,
        panel=[dict(state=s, state_key=state_key(s)) for s in batch[:16]],
        plan=plan,
        pilot_seed=1,
        variants=dict(VARIANTS),
        pools=pools,
        source_sha256={
            name: hashlib.sha256((Path(__file__).parents[1] / "conv" / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
    )
    request["utility_schema"] = 4
    request["utility_definition"] = dict(
        version=4,
        checkpoint_sha256="frozen",
        weights="ema",
        inference_execution="compiled",
        inference_precision="bf16_autocast",
        inference_batch=1,
        execution_scope="development_learnability",
        device="cuda",
        search=asdict(cfg.search),
        rules=asdict(cfg.rules),
        draw_kernel_width=cfg.play.draw_kernel_width,
        runtime=dict(
            cuda_matmul_tf32=True,
            matmul_precision="high",
            cudnn_tf32=True,
            cudnn_benchmark=False,
            cudnn_deterministic=False,
            deterministic_algorithms=False,
        ),
        source_sha256={k: v for k, v in request["source_sha256"].items() if k != "probe_parity.py"},
    )
    return dict(
        schema=VERSION,
        status="complete",
        request=request,
        units=units,
        setups=[dict(variant="C", cuda_matmul_tf32=True, cudnn_tf32=True, matmul_precision="high")],
        outcome="must not enter feature data",
        confirmations=["must not enter feature data"],
    ), cfg


def test_source_preserves_state_and_occurrence_identity_without_labels():
    parity, cfg = parity_fixture()
    data, pools = source_data(parity, "frozen", cfg)
    assert len(data["states"]) == 2
    assert len(data["plan"]) == len(pools) == 2
    for mapping in data["pools"].values():
        assert mapping["state_indices"] == [0, 1]
        assert [r[0] for r in mapping["occurrences"]] == [0, 1, 0]
        assert mapping["batches"] == [2, 1]
    for spec in data["plan"]:
        assert spec["state_indices"] == [0] + [1] * (BATCH - 1)
        assert spec["valid_rows"] == 2
    assert "must not enter" not in json.dumps(data)
    assert data == source_data(deepcopy(parity), "frozen", cfg)[0]


@pytest.mark.parametrize(
    "damage",
    [
        "running",
        "error",
        "checkpoint",
        "source",
        "checksum",
        "settings",
        "shape",
        "plan",
        "pool",
        "encoder",
        "legacy",
        "utility_schema",
    ],
)
def test_source_rejects_incompatible_or_corrupted_evidence(damage):
    parity, cfg = parity_fixture()
    if damage == "running":
        parity["status"] = "running"
    elif damage == "error":
        parity["error"] = "incomplete execution"
    elif damage == "checkpoint":
        parity["request"]["checkpoint_sha256"] = "wrong"
    elif damage == "source":
        parity["request"]["source_sha256"]["model.py"] = "wrong"
    elif damage == "checksum":
        parity["units"]["C/pool/discovery0/0"]["sha256"] = "wrong"
    elif damage == "settings":
        parity["setups"][0]["cuda_matmul_tf32"] = False
    elif damage == "shape":
        unit = parity["units"]["C/pool/discovery0/0"]
        unit["result"]["arrays"] = [pack_arrays(dict(scores=np.zeros((1, 2))))]
        unit["sha256"] = digest(unit["result"])
    elif damage == "plan":
        parity["request"]["plan"][-2:] = parity["request"]["plan"][-2:][::-1]
    elif damage == "pool":
        parity["request"]["pools"]["discovery0"]["snapshot"]["payload"]["sha256"] = "wrong"
    elif damage == "legacy":
        parity["schema"] = parity["request"]["version"] = VERSION - 1
    elif damage == "utility_schema":
        parity["request"]["utility_schema"] = 3
    else:
        cfg.net.value_hidden = 128
        parity["request"]["config"] = asdict(cfg)
    with pytest.raises(ValueError):
        source_data(parity, "frozen", cfg)


def test_capture_is_lossless_and_removes_hook_even_on_failure():
    head = torch.nn.Linear(3, 2)
    inputs = torch.tensor([[1.125, -0.0, 1.0001]])
    buffer = torch.empty_like(inputs)
    expected = head(inputs)
    with torch.no_grad(), capture_input(head, buffer):
        actual = head(inputs)
    assert torch.equal(actual, expected)
    assert torch.equal(buffer.view(torch.int32), inputs.view(torch.int32))
    assert not head._forward_pre_hooks
    with pytest.raises(RuntimeError, match="fixture"), capture_input(head, buffer):
        raise RuntimeError("fixture")
    assert not head._forward_pre_hooks


@pytest.mark.parametrize("shape,dtype", [((1, 3), torch.bfloat16), ((2, 3), torch.float32)])
def test_capture_rejects_shape_or_dtype_conversion(shape, dtype):
    head = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match="actual.*buffer"), capture_input(head, torch.zeros(shape, dtype=dtype)):
        head(torch.ones(1, 3))
    assert not head._forward_pre_hooks


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_feature_encoding_roundtrip_preserves_actual_bits(dtype):
    values = torch.tensor([[0.0, -0.0, 1.125, -3.25, 1e-12]], dtype=dtype)
    encoded = feature_array(values)
    restored = feature_tensor(unpack_arrays(pack_arrays(dict(features=encoded)))["features"], dtype, "cpu")
    bits = torch.uint16 if dtype == torch.bfloat16 else torch.int32
    assert torch.equal(values.view(bits), restored.view(bits))
    with pytest.raises(ValueError, match="encoding"):
        feature_tensor(encoded.astype(np.float64), dtype, "cpu")


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, FEATURES)
        self.regret = torch.nn.Sequential(torch.nn.Linear(FEATURES, 128), torch.nn.ReLU(), torch.nn.Linear(128, 2))

    def forward(self, values):
        state = self.encoder(values).relu()
        return self.regret(state)


def extraction_fixture(change_hook_output=False):
    torch.manual_seed(7)
    net = TinyNet().eval()

    @torch.no_grad()
    def forward(states):
        values = torch.tensor([s["values"] for s in states])
        output = net(values).numpy().copy()
        if change_hook_output and net.regret._forward_pre_hooks:
            output[0, 0] += 0.125
        return dict(rank=output[:, 0], regret=output[:, 1], other=values.numpy())

    states = [dict(state=dict(values=[1.0, 2.0])), dict(state=dict(values=[3.0, 5.0]))]
    rows = [0, 1] * (BATCH // 2)
    output = forward([states[i]["state"] for i in rows])
    plan = [
        dict(
            id="batch",
            state_indices=rows,
            valid_rows=BATCH,
            reference_scores=np.column_stack((output["rank"], output["regret"])).tolist(),
        )
    ]
    return SimpleNamespace(net=net, forward=forward), dict(states=states, plan=plan)


def test_extract_reconstructs_real_hidden_input_with_frozen_parameters_and_repeated_states():
    runner, data = extraction_fixture()
    weights = {name: value.clone() for name, value in runner.net.state_dict().items()}
    report = {}
    extract(runner, data, lambda head: head, torch.float32, "cpu", report)
    assert numerical_failures(report) == []
    arrays = unpack_arrays(report["chunks"]["batch"]["arrays"])
    expected = runner.net.encoder(torch.tensor([[1.0, 2.0], [3.0, 5.0]] * (BATCH // 2))).relu().detach().numpy()
    assert np.array_equal(arrays["features"], expected)
    assert np.array_equal(arrays["features"][0], arrays["features"][2])
    assert not runner.net.regret._forward_pre_hooks
    assert all(not p.requires_grad and p.grad is None for p in runner.net.parameters())
    assert all(torch.equal(value, runner.net.state_dict()[name]) for name, value in weights.items())


@pytest.mark.parametrize("damage", ["hook", "reconstruction", "saved_C"])
def test_extract_retains_exact_failure_evidence_without_a_tolerance(damage):
    runner, data = extraction_fixture(change_hook_output=damage == "hook")
    if damage == "saved_C":
        data["plan"][0]["reference_scores"][0][0] += 0.25
    reconstruct = (lambda head: lambda x: head(x) + 0.5) if damage == "reconstruction" else (lambda head: head)
    report = {}
    extract(runner, data, reconstruct, torch.float32, "cpu", report)
    failure = {"hook": "hooked/rank", "reconstruction": "reconstructed", "saved_C": "saved_C"}[damage]
    assert f"batch/{failure}" in numerical_failures(report)
    assert "features" in unpack_arrays(report["chunks"]["batch"]["arrays"])


def test_comparison_identifies_element_and_state_without_hiding_nonfinite_values():
    result = comparison(np.zeros((2, 2)), np.array([[0, 0], [0, 0.25]]), [41, 83])
    assert result["differing_elements"] == 1
    assert result["worst"] == dict(state_index=83, element=[1, 1], reference=0.0, actual=0.25)
    with pytest.raises(ValueError, match="nonfinite"):
        comparison(np.zeros(1), np.array([np.nan]), [0])


def test_hook_is_a_fullgraph_cpu_operation_and_refreshes_each_call():
    torch.manual_seed(8)
    net = TinyNet().eval().requires_grad_(False)
    inputs = torch.ones(BATCH, 2)
    buffer = torch.full((BATCH, FEATURES), torch.nan)
    compiled = torch.compile(net, backend="eager", fullgraph=True)
    with torch.no_grad(), capture_input(net.regret, buffer):
        for scale in (1, 3):
            values = inputs * scale
            buffer.fill_(torch.nan)
            output = compiled(values)
            expected = net.encoder(values).relu()
            assert torch.equal(buffer, expected)
            assert torch.equal(output, net.regret(expected))
    assert not net.regret._forward_pre_hooks


@pytest.mark.parametrize("skip_guards", [True, False])
def test_warmed_compiled_cache_observes_hook_attach_and_remove_only_with_guards(skip_guards):
    import torch._dynamo.config as dynamo_config

    torch.manual_seed(11)
    net = TinyNet().eval().requires_grad_(False)
    inputs = torch.ones(BATCH, 2)
    buffer = torch.full((BATCH, FEATURES), torch.nan)
    compilations = []

    def backend(graph, inputs):
        compilations.append(graph)
        return graph.forward

    with dynamo_config.patch(skip_nnmodule_hook_guards=skip_guards, suppress_errors=False), torch.no_grad():
        compiled = torch.compile(net, backend=backend, fullgraph=True)
        baseline = compiled(inputs).clone()
        assert len(compilations) == 1
        with capture_input(net.regret, buffer):
            for scale in (1, 3):
                buffer.fill_(torch.nan)
                values = inputs * scale
                result = compiled(values)
                if skip_guards:
                    assert torch.isnan(buffer).all()
                    assert len(compilations) == 1
                else:
                    assert torch.equal(buffer, net.encoder(values).relu())
                    assert len(compilations) == 2
                if scale == 1:
                    assert torch.equal(result, baseline)
        buffer.fill_(torch.nan)
        assert torch.equal(compiled(inputs), baseline)
        assert torch.isnan(buffer).all()
    assert not net.regret._forward_pre_hooks


def test_extract_preserves_warmed_symbolic_batch_through_capture_and_head_reconstruction():
    import torch._dynamo.config as dynamo_config

    runner, data = extraction_fixture()
    runner.net.requires_grad_(False)
    graphs, phase = [], ["network"]

    def backend(graph, inputs):
        shapes = {}
        for node in graph.graph.nodes:
            value = node.meta.get("example_value")
            if node.op == "placeholder" and isinstance(value, torch.Tensor):
                shapes[node.name] = [(str(n), isinstance(n, torch.SymInt)) for n in value.shape]
        graphs.append((phase[0], shapes))
        return graph.forward

    compiled = torch.compile(runner.net, backend=backend, fullgraph=True)

    @torch.no_grad()
    def forward(states):
        values = torch.tensor([s["values"] for s in states])
        scores = compiled(values).numpy().copy()
        return dict(rank=scores[:, 0], regret=scores[:, 1], other=values.numpy())

    def compile_head(head):
        phase[0] = "head"
        return torch.compile(head, backend=backend, fullgraph=True)

    runner.forward = forward
    with (
        dynamo_config.patch(skip_nnmodule_hook_guards=False, suppress_errors=False, allow_ignore_mark_dynamic=False),
        torch.no_grad(),
    ):
        state = data["states"][0]["state"]
        runner.forward([state])
        runner.forward([state] * BATCH)
        warm_shape = graphs[-1][1]["l_values_"]
        assert warm_shape[0][1]
        report = {}
        extract(runner, data, compile_head, torch.float32, "cpu", report)
    network = [shapes for label, shapes in graphs if label == "network"]
    hooked_shape = network[-1]["l_values_"]
    assert hooked_shape[0][1]
    buffers = [shape for shape in network[-1].values() if len(shape) == 2 and shape[1][0] == str(FEATURES)]
    assert any(shape[0] == hooked_shape[0] for shape in buffers)
    head_inputs = [shape for label, shapes in graphs if label == "head" for shape in shapes.values()]
    assert any(len(shape) == 2 and shape[0][1] and shape[1][0] == str(FEATURES) for shape in head_inputs)
    assert numerical_failures(report) == []


@pytest.mark.parametrize("checkpoint", ["latest.pt", "ckpt_000160.pt"])
def test_command_refuses_mutable_or_existing_output_before_cuda(tmp_path, monkeypatch, checkpoint):
    output = tmp_path / "features.json"
    output.write_text("existing")
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_control", "--ckpt", checkpoint, "--parity", "parity.json", "--out", str(output), "--device", "cuda"],
    )
    with pytest.raises(SystemExit):
        main()
    assert output.read_text() == "existing"
