"""CPU oracles for the frozen parity panel, occurrence weighting and resumable evidence."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from conv.config import Config
from conv.control import State
from conv.probe_control import CandidatePool, state_key, state_record
from conv.probe_parity import (
    SOURCES,
    VERSION,
    CudaRunner,
    Progress,
    digest,
    evaluator_comparison,
    execute,
    freeze_panel,
    load_pools,
    make_plan,
    pack_arrays,
    pool_comparison,
    probabilities,
    root_comparison,
    unpack_arrays,
    worker_lock,
)


def candidate(index, rank=0, regret=0, source="tree"):
    board = np.zeros(81, np.int8)
    board[10], board[70] = 1, 4
    return dict(
        state=state_record(State(board, index % 199, index, 200)),
        source=source,
        occurrence=[index, index + 1],
        rank=rank,
        regret=regret,
    )


def large_pools(games=2):
    pools = {}
    for game in range(games):
        identity = f"discovery{game}"
        pool = CandidatePool(1, identity)
        for batch in range(11):
            values = [
                candidate(game * 1100 + i, float(i % 113), float(i % 197), "played" if i % 3 == 0 else "tree")
                for i in range(batch * 100, (batch + 1) * 100)
            ]
            pool.offer(values)
        pools[identity] = pool
    return pools


def test_panel_freezes_all_selections_then_distinct_strata_and_heterogeneous_batch():
    pools = large_pools()
    panel, batch = freeze_panel(pools, 200)
    assert (panel, batch) == freeze_panel(pools, 200)
    assert len(panel) == len({v["state_key"] for v in panel}) == 16
    assert len(batch) == len({state_key(s) for s in batch}) == 1024
    assert batch[:16] == [v["state"] for v in panel]
    reasons = [reason for entry in panel for reason in entry["reasons"]]
    for identity, pool in pools.items():
        for method, selection in pool.selections()["selected"].items():
            entry = next(v for v in panel if v["state_key"] == state_key(selection["state"]))
            assert f"{identity}/{method}" in entry["reasons"]
    assert "initial_position" in reasons
    assert all(
        any(reason.startswith(prefix) for reason in reasons)
        for prefix in ("played_early", "played_late", "tree_early", "tree_late", "near_clock", "rank_max", "rank_min")
    )


def test_corpus_panel_stays_fixed_size_while_rescoring_every_discovery_pool():
    pools = large_pools(3)
    panel, batch = freeze_panel(pools, 200)
    assert len(panel) == 16 and len(batch) == 1024
    reasons = [reason for entry in panel for reason in entry["reasons"]]
    assert not any(reason.startswith("discovery2/") for reason in reasons)
    assert {s["pool"] for s in make_plan(pools) if s["kind"] == "pool"} == set(pools)


def test_plan_has_exact_matched_roots_repeats_mode_switches_and_complete_rescore_chunks():
    pools = large_pools()
    plan = make_plan(pools)
    assert len({s["id"] for s in plan}) == len(plan)
    for variant in ("A", "C"):
        assert sum(s["id"].startswith(f"{variant}/root/") for s in plan) == 64
        assert sum(s["id"].startswith(f"{variant}/repeat/") for s in plan) == 2
        assert sum(s["id"].startswith(f"{variant}/switch/") for s in plan) == 4
        assert sum(s["id"].startswith(f"{variant}/batch/") for s in plan) == 2
    for identity, pool in pools.items():
        chunks = [s for s in plan if s["kind"] == "pool" and s["pool"] == identity]
        assert [i for s in chunks for i in range(s["start"], s["stop"])] == list(range(len(pool.states)))


def test_occurrence_rescore_keeps_multiplicity_batches_and_uniform_rng():
    pool = CandidatePool(4, "discovery0")
    a, b = candidate(0), candidate(1)
    pool.offer([a, a])
    pool.offer([a, b])
    snapshot = deepcopy(pool.snapshot())
    result = pool_comparison(snapshot, np.array([[0, 0], [np.log(3), 2]]), 4, "discovery0")
    assert result["occurrences"] == 4 and result["distinct_states"] == 2
    assert result["rank_probability_tv"] == pytest.approx(0.25)
    assert result["rank"]["mean"] == pytest.approx(np.log(3) / 4)
    assert result["selectors"]["uniform"]["same_occurrence"]
    assert not result["selectors"]["rank_argmax"]["same_state"]
    assert snapshot == pool.snapshot()
    with pytest.raises(ValueError, match="incomplete"):
        pool_comparison(snapshot, np.zeros((1, 2)), 4, "discovery0")


def evaluator_arrays():
    return dict(
        logits=np.array([[0.0, 1.0, 100.0]]),
        legal=np.array([[True, True, False]]),
        count=np.array([2]),
        q=np.array([[0.2, 0.4, 0.0]]),
        v=np.array([0.3]),
        rank=np.array([10.0]),
        regret=np.array([-0.1]),
        draw=np.array([0.2]),
    )


def test_numerical_metrics_separate_policy_offset_from_value_error_and_legality():
    a, b = evaluator_arrays(), evaluator_arrays()
    b["logits"] += 90
    b["v"] += 0.25
    result = evaluator_comparison(a, b)
    assert result["policy_tv"]["maximum"] == pytest.approx(0)
    assert result["v"]["maximum"] == pytest.approx(0.25)
    b["legal"][0, 1] = False
    result = evaluator_comparison(a, b)
    assert not result["legal_exact"] and "policy_tv" not in result
    with pytest.raises(ValueError):
        probabilities([0.0, 0.0], [False, False])


def root_arrays():
    return dict(
        state_key=np.array(["same", "different"]),
        action=np.array([1, 2]),
        moves=np.array([[1, 2], [1, 2]]),
        tactics=np.zeros((2, 2)),
        initial_effective_candidates=np.array([[1, 2], [1, 2]]),
        initial_physical_candidates=np.array([[1, 2], [1, 2]]),
        candidates=np.array([[1], [2]]),
        candidate_states_sha256=np.array(["a", "b"]),
        played_q=np.zeros(2),
        q=np.zeros((2, 2)),
        value=np.zeros(2),
        rank=np.zeros(2),
        regret=np.zeros(2),
        target=np.array([[0.25, 0.75], [0.5, 0.5]]),
    )


def test_root_metrics_exclude_diverged_continuation_states_and_do_not_invent_batch_tree_hashes():
    a, b = root_arrays(), root_arrays()
    b["state_key"][1] = "changed"
    b["action"][1] = 99
    b["target"][0] = [0.75, 0.25]
    result = root_comparison(a, b)
    assert result["matched_state_rows"] == result["action_exact_rows"] == 1
    assert result["target_tv"]["maximum"] == 0.5
    assert not result["tree_context_hash_available"]
    assert "tree_sha256_exact_rows" not in result
    b["moves"][0] = [2, 1]
    result = root_comparison(a, b)
    assert result["aligned_move_rows"] == 0 and "target_tv" not in result and "q" not in result


def test_array_payload_roundtrips_all_evidence_types_and_rejects_corruption():
    arrays = {**root_arrays(), "finite_mask": np.array([True, False]), "infinity": np.array([-1, 1], np.int8)}
    payload = pack_arrays(arrays)
    assert all(np.array_equal(v, unpack_arrays(payload)[k]) for k, v in arrays.items())
    with pytest.raises(ValueError, match="nonfinite"):
        pack_arrays({"value": np.array([float("nan")])})
    altered = {**payload, "sha256": "bad"}
    with pytest.raises(ValueError, match="checksum"):
        unpack_arrays(altered)


def test_progress_resumes_completed_units_only_and_rejects_request_and_evidence_changes(tmp_path):
    path = tmp_path / "parity.json"
    plan = [dict(id=f"A/fixture/{i}", variant="A", kind="fixture") for i in range(3)]
    request = dict(version=VERSION, plan=plan, pools={}, source_sha256="frozen")
    calls, closed = [], []

    class Runner:
        def __init__(self, fail):
            self.fail = fail

        def warmup(self):
            return dict(seconds={"1": 0, "1024": 0})

        def run(self, spec):
            calls.append(spec["id"])
            if self.fail and spec["id"] == plan[1]["id"]:
                raise RuntimeError("fixture interruption")
            return dict(arrays=[pack_arrays(dict(value=np.array([int(spec["id"].split("/")[-1])])))])

        def close(self):
            closed.append(True)

    with pytest.raises(RuntimeError, match="interruption"):
        execute(Progress(path, request, False), lambda variant: Runner(True))
    saved = json.loads(path.read_text())
    assert saved["status"] == "running" and len(saved["remaining"]) == 2
    assert len(saved["units"]) == 1
    with pytest.raises(ValueError, match="request"):
        Progress(path, {**request, "source_sha256": "changed"}, True)
    result = execute(Progress(path, request, True), lambda variant: Runner(False))
    assert result["status"] == "complete" and result["remaining"] == []
    assert calls.count(plan[0]["id"]) == 1 and len(result["setups"]) == 2
    assert len(closed) == 2
    assert "passed" not in result
    complete = Progress(path, request, True)
    execute(complete, lambda variant: pytest.fail("completed run rebuilt evaluator"))
    legacy = deepcopy(result)
    legacy["schema"] = VERSION - 1
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="request or status"):
        Progress(path, request, True)
    corrupt = deepcopy(result)
    corrupt["units"][plan[0]["id"]]["result"]["extra"] = 1
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="checksum"):
        Progress(path, request, True)
    corrupt = deepcopy(result)
    del corrupt["units"][plan[1]["id"]]
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="unrun"):
        Progress(path, request, True)


@pytest.mark.skipif(os.name != "nt", reason="Windows worker mutex")
def test_worker_mutex_rejects_concurrent_owner_and_releases_without_files():
    def competing():
        with pytest.raises(RuntimeError, match="another parity worker"):
            with worker_lock():
                pytest.fail("second worker acquired mutex")

    with worker_lock(), ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(competing).result()
    with worker_lock():
        pass


def test_request_fingerprint_rejects_nonfinite_metadata():
    with pytest.raises(ValueError):
        digest({"value": float("inf")})


def input_fixture():
    cfg = Config()
    cfg.rules.clock_min = cfg.rules.clock_max = 200
    definitions = dict(
        version=4,
        checkpoint_sha256="frozen",
        weights="ema",
        inference_execution="compiled",
        inference_batch=1,
        execution_scope="development_learnability",
        inference_precision="bf16_autocast",
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
        source_sha256={
            name: hashlib.sha256((Path(__file__).parents[1] / "conv" / name).read_bytes()).hexdigest()
            for name in SOURCES
            if name != "probe_parity.py"
        },
    )
    protocol = hashlib.sha256(json.dumps(definitions, sort_keys=True).encode()).hexdigest()
    records, episodes = [], []
    for index in range(2):
        identity = f"discovery{index}"
        pool = CandidatePool(1, identity)
        pool.offer([candidate(index * 2), candidate(index * 2 + 1)])
        records.append(dict(game=identity, discovery_censored=False, candidates=pool.selections()))
        episodes.append(dict(episode_id=identity, candidate_pool=pool.snapshot(), censored=False))
    pilot = dict(
        status="complete",
        schema=4,
        seed=1,
        protocol=protocol,
        definition=definitions,
        request=dict(version=4, games=2, seed=1, protocol=protocol, definition=definitions, selection="uniform_corpus"),
        games=records,
        episodes=episodes,
    )
    return pilot, cfg


def test_source_validation_reproduces_completed_pools():
    pilot, cfg = input_fixture()
    pools = load_pools(pilot, "frozen", cfg)
    assert set(pools) == {"discovery0", "discovery1"}
    assert all(p.count == 2 for p in pools.values())


def test_source_validation_accepts_one_game_corpus_smoke_without_outcome_summaries():
    pilot, cfg = input_fixture()
    pilot["request"]["games"] = 1
    pilot["games"] = pilot["games"][:1]
    pilot["episodes"] = pilot["episodes"][:1]
    assert "comparisons" not in pilot
    assert set(load_pools(pilot, "frozen", cfg)) == {"discovery0"}


@pytest.mark.parametrize(
    "damage",
    ["running", "checkpoint", "protocol", "seed", "source", "selection", "legacy", "compile", "eager", "batch"],
)
def test_source_validation_rejects_malformed_or_incompatible_pilot(damage):
    pilot, cfg = input_fixture()
    if damage == "running":
        pilot["status"] = "running"
    elif damage == "checkpoint":
        pilot["definition"]["checkpoint_sha256"] = "wrong"
    elif damage == "protocol":
        pilot["protocol"] = "wrong"
    elif damage == "seed":
        pilot["request"]["seed"] = 2
    elif damage == "source":
        pilot["definition"]["source_sha256"]["search.py"] = "wrong"
        protocol = hashlib.sha256(json.dumps(pilot["definition"], sort_keys=True).encode()).hexdigest()
        pilot["protocol"] = pilot["request"]["protocol"] = protocol
    elif damage == "selection":
        pilot["games"][0]["candidates"]["selected"]["uniform"]["rank"] = 99
    elif damage == "legacy":
        pilot["schema"] = pilot["request"]["version"] = 3
    elif damage in ("eager", "batch"):
        field, value = ("inference_execution", "eager") if damage == "eager" else ("inference_batch", 1024)
        pilot["definition"][field] = value
        protocol = hashlib.sha256(json.dumps(pilot["definition"], sort_keys=True).encode()).hexdigest()
        pilot["protocol"] = pilot["request"]["protocol"] = protocol
    else:
        cfg.learn.compile = False
    with pytest.raises(ValueError):
        load_pools(pilot, "frozen", cfg)


@pytest.mark.parametrize("clock,recorded", [(1, 1), (2, 2)])
def test_root_phase_boundaries_with_real_cpu_search_and_zero_evaluator(monkeypatch, clock, recorded):
    import conv.probe_parity as parity
    from conv import kernels

    cfg = Config()
    cfg.rules.clock_min = cfg.rules.clock_max = clock
    cfg.rules.max_plies = 20
    cfg.search.sims, cfg.search.candidates = 2, 2
    cfg.search.cheap_sims, cfg.search.cheap_candidates = 1, 1
    cfg.search.reuse_nodes = 2
    state = candidate(4)["state"]
    state.update(since_capture=0, clock=clock)
    runner = object.__new__(CudaRunner)
    runner.cfg, runner.panel, runner.master = cfg, [dict(state=state)], 0

    def evaluate(board, since, ply, clocks):
        _, legal, count = kernels.derive_batch(board, since, ply, clocks)
        action = torch.zeros(len(board), 648)
        value = torch.zeros(len(board))
        return action, action, value, value, legal, count, value, value

    runner.evaluate = evaluate
    original_game = parity.Game

    class CpuGame(original_game):
        def __init__(self, cfg, evaluate, state, seed, device, **kw):
            super().__init__(cfg, evaluate, state, seed, "cpu", **kw)

    monkeypatch.setattr(parity, "Game", CpuGame)
    monkeypatch.setattr(
        runner,
        "tensors",
        lambda states: tuple(
            torch.tensor([state[name] for state in states], dtype=dtype)
            for name, dtype in (
                ("board", torch.int8),
                ("since_capture", torch.int32),
                ("ply", torch.int32),
                ("clock", torch.int32),
            )
        ),
    )
    for name in ("arange", "zeros"):
        original = getattr(torch, name)
        monkeypatch.setattr(
            torch, name, lambda *args, _original=original, **kwargs: _original(*args, **{**kwargs, "device": "cpu"})
        )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    result = runner.root(dict(kind="root", state=0, seed=0, full=True, steps=2))
    assert len(result["arrays"]) == recorded and result["requested_steps"] == 2
    first = unpack_arrays(result["arrays"][0])
    assert first["initial_effective_candidates"].shape == (1, 2)
    assert "final_selection_score_finite" in first and "tree_sha256" in first
    if clock == 1:
        assert result["end_reasons"] == [2]
    else:
        second = unpack_arrays(result["arrays"][1])
        assert second["state_key"][0] != first["state_key"][0]
        assert second["initial_effective_candidates"][0, 1] == -1
