"""Frozen probe provenance, sampling, and real-engine CPU integration with a tiny evaluator."""

import base64
import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
import torch
from compression import zstd

from conv.config import Config, Rules, Search
from conv.control import State
from conv.probe_control import (
    CandidatePool,
    Game,
    Progress,
    atomic_json,
    candidate_batches,
    finite_sample_completion_range,
    paired_observation,
    probe,
    replay_candidate_pool,
    seed_for,
    state_key,
    state_record,
    validate_return_support,
)


def candidate(index, rank=0.0, source="tree"):
    board = np.zeros(81, np.int8)
    board[10], board[70] = 1, 4
    return dict(
        state=state_record(State(board, 0, index, 100)),
        source=source,
        occurrence=[0, index],
        rank=rank,
        regret=rank / 10,
    )


def completion_record(index, opening, uniform_opening, future, uniform_future, *, shared=False, censored=False):
    identity = f"discovery{index}"
    chosen, uniform = candidate(2 * index + 1), candidate(2 * index + 2)
    if shared:
        uniform = chosen
    selected = dict(rank_sample=chosen, rank_argmax=chosen, regret_argmax=chosen, uniform=uniform)
    record = dict(game=identity, discovery_censored=censored, candidates=dict(selected=selected), measurements={})
    if censored:
        return record

    def confirmation(products, identity):
        return dict(
            measurement_id=identity,
            pairs=[
                dict(
                    status="outcome_censored" if product is None else "complete",
                    product=product,
                    episodes=[f"{identity}/pair{i}/{side}" for side in (0, 1)],
                )
                for i, product in enumerate(products)
            ],
            mean=None if None in products else float(np.mean(products)),
        )

    for selection, products, future_products in ((chosen, opening, future), (uniform, uniform_opening, uniform_future)):
        key = state_key(selection["state"])
        if key in record["measurements"]:
            continue
        prefix = f"{identity}/candidate{len(record['measurements'])}"
        anchors = [
            dict(
                index=0,
                state_key=state_key(candidate(7)["state"]),
                base_episode=f"{prefix}/base",
                confirmation=confirmation(values, f"{prefix}/anchor{i}"),
            )
            for i, values in enumerate(future_products or [])
        ]
        means = [a["confirmation"]["mean"] for a in anchors]
        record["measurements"][key] = dict(
            opening=confirmation(products, f"{prefix}/opening"),
            base_episode=f"{prefix}/base",
            base_censored=future_products is None,
            future_anchors=anchors,
            future_mean=float(np.mean(means)) if means and None not in means else None,
        )
    return record


def test_completion_range_signed_products_partial_anchors_and_independent_equal_anchor_states():
    record = completion_record(0, [1, None], [-1, 0.5], [[2, None], [-1, 0]], [[1, 3], [-1, 1]])
    result = finite_sample_completion_range([record], games=1, pairs=2, anchors=2)
    assert result["opening/rank_sample"] == pytest.approx([-5 / 4, 11 / 4])
    assert result["future/rank_sample"] == pytest.approx([-7 / 4, 1 / 4])
    pair = next(iter(record["measurements"].values()))["opening"]["pairs"][1]
    pair.update(status="attempt_cap", episodes=[])
    assert finite_sample_completion_range([record], 1, 2, 2) == result


def test_completion_range_averages_all_planned_games_including_discovery_and_base_censoring():
    partial = completion_record(0, [1, None], [-1, 0.5], [[2, None], [-1, 0]], [[1, 3], [-1, 1]])
    shared = completion_record(1, [None, None], [], None, None, shared=True)
    censored = completion_record(2, [], [], None, None, shared=True, censored=True)
    result = finite_sample_completion_range([partial, shared, censored], 3, 2, 2)
    assert result["opening/rank_argmax"] == pytest.approx([-37 / 12, 43 / 12])

    base_censored = completion_record(1, [0, 0], [0, 0], None, [[2, 2], [2, 2]])
    shared = completion_record(2, [None, None], [], None, None, shared=True)
    censored = completion_record(3, [], [], None, None, shared=True, censored=True)
    result = finite_sample_completion_range([partial, base_censored, shared, censored], 4, 2, 2)
    assert result["future/regret_argmax"] == pytest.approx([-63 / 16, 41 / 16])


def test_completion_range_shared_measurement_cancels_censoring_only_after_completed_discovery():
    shared = completion_record(0, [None, None], [], None, None, shared=True)
    assert all(value == [0, 0] for value in finite_sample_completion_range([shared], 1, 2, 3).values())
    censored = completion_record(0, [], [], None, None, shared=True, censored=True)
    assert all(value == [-8, 8] for value in finite_sample_completion_range([censored], 1, 2, 3).values())


def test_completion_range_preserves_fully_observed_negative_products_and_asymmetric_censoring():
    record = completion_record(0, [-1, -1], [0, 0], [[-1, -1]], [[0, 0]])
    assert all(value == [-1, -1] for value in finite_sample_completion_range([record], 1, 2, 1).values())
    record = completion_record(0, [1, 1, None, None], [0, 0, 0, 0], None, None)
    assert finite_sample_completion_range([record], 1, 4, 1)["opening/rank_sample"] == [-1.5, 2.5]


@pytest.mark.parametrize(
    "damage",
    [
        "missing_game",
        "duplicate_game",
        "discovery_status",
        "missing_selection",
        "missing_measurement",
        "measurement_identity",
        "missing_pair",
        "unfinished_pair",
        "missing_product",
        "censored_product",
        "complete_missing_product",
        "nonfinite_product",
        "out_of_support",
        "reused_episode",
        "incorrect_mean",
        "base_status",
        "base_identity",
        "missing_anchor",
        "anchor_identity",
        "anchor_base_identity",
        "incorrect_future_mean",
        "censored_base_with_anchors",
        "censored_discovery_with_measurements",
    ],
)
def test_completion_range_rejects_unfinished_or_malformed_work(damage):
    records = [completion_record(i, [1, None], [0, 0], [[0, 0]], [[0, 0]], shared=True) for i in range(2)]
    record = records[0]
    measured = next(iter(record["measurements"].values()))
    confirmation = measured["opening"]
    pair = confirmation["pairs"][0]
    if damage == "missing_game":
        records.pop()
    elif damage == "duplicate_game":
        records[1]["game"] = record["game"]
    elif damage == "discovery_status":
        record["discovery_censored"] = None
    elif damage == "missing_selection":
        del record["candidates"]["selected"]["uniform"]
    elif damage == "missing_measurement":
        record["measurements"].clear()
    elif damage == "measurement_identity":
        confirmation["measurement_id"] += "/wrong"
    elif damage == "missing_pair":
        confirmation["pairs"].pop()
    elif damage == "unfinished_pair":
        pair["status"] = "running"
    elif damage == "missing_product":
        del pair["product"]
    elif damage == "censored_product":
        pair["status"] = "outcome_censored"
    elif damage == "complete_missing_product":
        pair["product"] = None
    elif damage == "nonfinite_product":
        pair["product"] = float("nan")
    elif damage == "out_of_support":
        pair["product"] = 4.01
    elif damage == "reused_episode":
        pair["episodes"][1] = pair["episodes"][0]
    elif damage == "incorrect_mean":
        confirmation["mean"] = 1
    elif damage == "base_status":
        measured["base_censored"] = None
    elif damage == "base_identity":
        measured["base_episode"] += "/wrong"
    elif damage == "missing_anchor":
        measured["future_anchors"].clear()
    elif damage == "anchor_identity":
        measured["future_anchors"][0]["confirmation"]["measurement_id"] += "/wrong"
    elif damage == "anchor_base_identity":
        measured["future_anchors"][0]["base_episode"] += "/wrong"
    elif damage == "incorrect_future_mean":
        measured["future_mean"] = 1
    elif damage == "censored_base_with_anchors":
        measured["base_censored"] = True
    else:
        record["discovery_censored"] = True
    with pytest.raises(ValueError):
        finite_sample_completion_range(records, 2, 2, 1)


def test_streaming_candidate_selection_matches_unsplit_set_and_flat_occurrence_baseline():
    candidates = [candidate(1), candidate(1), candidate(1), candidate(2)]
    count_a = 0
    for seed in range(1500):
        whole, split = CandidatePool(seed, "g"), CandidatePool(seed, "g")
        whole.offer(candidates)
        split.offer(candidates[:1])
        split.offer(candidates[1:])
        a, b = whole.selections(), split.selections()
        assert a == b
        assert a["occurrences"] == 4 and a["distinct_states"] == 2
        assert a["selected"]["rank_sample"]["occurrence_probability"] == pytest.approx(0.25)
        count_a += a["selected"]["uniform"]["state"]["ply"] == 1
    assert 0.70 < count_a / 1500 < 0.80


def test_candidate_pool_handles_extreme_scores_and_does_not_change_input():
    values = [candidate(1, -1000), candidate(2, 1000)]
    saved = deepcopy(values)
    pool = CandidatePool(0, "g")
    pool.offer(values)
    result = pool.selections()
    assert result["selected"]["rank_sample"]["occurrence_probability"] == 1
    assert result["selected"]["rank_argmax"]["state"]["ply"] == 2
    assert values == saved
    with pytest.raises(ValueError):
        pool.offer([candidate(3, float("nan"))])


def recorded_pool():
    values = [candidate(i) for i in (1, 2, 2, 3, 1, 4)]
    for value, rank, regret, source, occurrence in zip(
        values,
        (-0.5, 3.0, 3.0, -1.0, 0.2, 3.0),
        (-2.0, 0.0, 1.0, 9.0, 8.0, -4.0),
        ("played", "tree", "played", "tree", "tree", "tree"),
        ([0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]),
        strict=True,
    ):
        value.update(rank=rank, regret=regret, source=source, occurrence=occurrence)
    batches = [values[:2], values[2:3], values[3:]]
    pool = CandidatePool(1, "fixture")
    for batch in batches:
        pool.offer(batch)
    return pool, batches


def test_candidate_payload_roundtrip_preserves_exact_selections_rng_and_occurrences():
    pool, batches = recorded_pool()
    values = [v for batch in batches for v in batch]
    expected = {
        method: {**values[i], "occurrence_probability": probability}
        for method, i, probability in (
            ("rank_sample", 2, 0.32161340995559373),
            ("rank_argmax", 1, 1.0),
            ("regret_argmax", 3, 1.0),
            ("uniform", 5, 1 / 6),
        )
    }
    assert pool.selections()["selected"] == expected
    rng = deepcopy([pool.rank_rng.bit_generator.state, pool.uniform_rng.bit_generator.state])
    snapshot = pool.snapshot()
    assert pool.snapshot()["payload"] is snapshot["payload"]
    assert [pool.rank_rng.bit_generator.state, pool.uniform_rng.bit_generator.state] == rng
    saved = json.loads(json.dumps(snapshot, allow_nan=False))
    assert list(candidate_batches(saved)) == batches
    restored = replay_candidate_pool(saved, 1, "fixture")
    assert restored.snapshot() == saved
    assert restored.selections() == pool.selections()
    assert restored.selections()["sources"] == {"played": 2, "tree": 4}
    assert restored.selections()["distinct_states"] == 4
    assert restored.selections()["occurrences"] == 6
    assert [restored.rank_rng.bit_generator.state, restored.uniform_rng.bit_generator.state] == rng
    decoded = list(candidate_batches(saved))
    decoded[0][1]["state"]["board"][0] = 6
    assert decoded[1][0]["state"]["board"][0] == 0
    assert list(candidate_batches(saved)) == batches
    pool.offer([candidate(8)])
    assert pool.snapshot()["payload"] is not snapshot["payload"]
    assert replay_candidate_pool(snapshot, 1, "fixture").snapshot() == saved


@pytest.mark.parametrize("field,value", [("data", "!bad"), ("sha256", "bad"), ("raw_bytes", 1), ("encoding", "old")])
def test_candidate_payload_rejects_corrupted_envelope(field, value):
    pool, _ = recorded_pool()
    snapshot = deepcopy(pool.snapshot())
    snapshot["payload"][field] = value
    with pytest.raises(ValueError, match="invalid candidate payload"):
        list(candidate_batches(snapshot))


@pytest.mark.parametrize("damage", ["batch", "reference", "board", "source", "score", "unused_state"])
def test_candidate_payload_rejects_invalid_tables_with_valid_checksum(damage):
    pool, _ = recorded_pool()
    snapshot = deepcopy(pool.snapshot())
    payload = snapshot["payload"]
    data = json.loads(zstd.decompress(base64.b64decode(payload["data"])))
    if damage == "batch":
        data["batches"][0] += 1
    elif damage == "reference":
        data["occurrences"][0][0] = len(data["states"])
    elif damage == "board":
        data["states"][0]["board"][0] = 256
    elif damage == "source":
        data["occurrences"][0][1] = "unknown"
    elif damage == "score":
        data["occurrences"][0][4] = float("nan")
    else:
        data["occurrences"][-1][0] = 0
    raw = json.dumps(data, separators=(",", ":")).encode()
    payload.update(
        raw_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), data=base64.b64encode(zstd.compress(raw)).decode()
    )
    with pytest.raises(ValueError, match="invalid candidate payload"):
        list(candidate_batches(snapshot))


def test_candidate_replay_rejects_changed_summary_selection_and_seed():
    pool, _ = recorded_pool()
    for key in ("occurrences", "distinct_states"):
        snapshot = deepcopy(pool.snapshot())
        snapshot[key] += 1
        with pytest.raises(ValueError, match="invalid candidate payload"):
            replay_candidate_pool(snapshot, 1, "fixture")
    snapshot = deepcopy(pool.snapshot())
    snapshot["selected"]["rank_sample"]["occurrence_probability"] = 0.5
    with pytest.raises(ValueError, match="candidate selections"):
        replay_candidate_pool(snapshot, 1, "fixture")
    with pytest.raises(ValueError, match="candidate selections"):
        replay_candidate_pool(pool.snapshot(), 2, "fixture")


def episode(identity, seed, context_seed=None, action=3, q=0.5, outcome=1.0, full=True):
    context_seed = seed_for(seed, "root_context") if context_seed is None else context_seed
    return dict(
        episode_id=identity,
        seed=seed,
        protocol="frozen",
        state_key="same",
        censored=False,
        first_action=action,
        first_q=q,
        outcome=outcome,
        residual=q - outcome,
        streams={k: seed_for(seed, "future", k) for k in ("search", "mode", "environment")},
        context=dict(
            context_id=f"root{context_seed}",
            seed=context_seed,
            action=action,
            q=q,
            full=full,
            protocol="frozen",
            state_key="same",
            tree_sha256=f"tree{context_seed}",
            streams={k: seed_for(context_seed, "root", k) for k in ("search", "mode")},
        ),
    )


def test_pair_allows_coincident_outcomes_but_not_reused_provenance():
    a, b = episode("a", 1), episode("b", 2)
    assert paired_observation(a, b, "state_policy") == 0.25
    for field in ("episode_id", "seed", "streams"):
        bad = {**b, field: a[field]}
        with pytest.raises(ValueError):
            paired_observation(a, bad, "state_policy")
    with pytest.raises(ValueError):
        paired_observation(a, b, "state_policy", {"b"})


@pytest.mark.parametrize(
    "change",
    [
        {"protocol": "other_actor"},
        {"state_key": "other_state"},
        {"censored": True},
        {"residual": None},
        {"residual": float("nan")},
    ],
)
def test_pair_rejects_incompatible_or_censored_labels(change):
    with pytest.raises(ValueError):
        paired_observation(episode("a", 1), {**episode("b", 2), **change}, "state_policy")


def test_pair_estimands_distinguish_shared_roots_actions_and_search_noise():
    a, b = episode("a", 1, context_seed=8), episode("b", 2, context_seed=8)
    assert paired_observation(a, b, "root_context") == 0.25
    for estimand in ("state_policy", "action_mode"):
        with pytest.raises(ValueError):
            paired_observation(a, b, estimand)
    independent = episode("c", 3, context_seed=9, q=0.2)
    assert paired_observation(a, independent, "action_mode") == pytest.approx(0.4)
    with pytest.raises(ValueError):
        paired_observation(a, independent, "root_context")
    for changed in (episode("d", 4, action=7), episode("e", 5, full=False)):
        with pytest.raises(ValueError):
            paired_observation(a, changed, "action_mode")
    malformed = deepcopy(independent)
    malformed["residual"] = 0
    with pytest.raises(ValueError):
        paired_observation(a, malformed, "action_mode")


def test_fine_and_coarse_expectations_separate_action_cancellation_and_root_noise():
    from itertools import product

    state_values, coarse_values, fine_values = [], [], []
    for a, b in product((-1, 1), repeat=2):
        state_values.append(
            paired_observation(
                episode("a", 1, action=a, q=a / 2, outcome=0),
                episode("b", 2, action=b, q=b / 2, outcome=0),
                "state_policy",
            )
        )
        coarse_values.append(
            paired_observation(
                episode("a", 1, action=a, q=a / 2, outcome=0),
                episode("b", 2, action=a, q=a / 2, outcome=0),
                "action_mode",
            )
        )
        fine_values.append(
            paired_observation(
                episode("a", 1, context_seed=8, q=a / 2, outcome=0),
                episode("b", 2, context_seed=8, q=a / 2, outcome=0),
                "root_context",
            )
        )
    assert np.mean(state_values) == 0
    assert np.mean(coarse_values) == np.mean(fine_values) == 0.25
    root_noise = [
        paired_observation(episode("a", 1, q=a / 2, outcome=0), episode("b", 2, q=b / 2, outcome=0), "action_mode")
        for a, b in product((-1, 1), repeat=2)
    ]
    assert np.mean(root_noise) == 0


def small_config(clock=2, max_plies=20):
    cfg = Config()
    cfg.search = Search(
        sims=2, candidates=2, cheap_sims=1, cheap_candidates=1, reuse_nodes=2, temperature_plies=0, full_fraction=0.5
    )
    cfg.rules = Rules(clock_min=clock, clock_max=clock, max_plies=max_plies)
    return cfg


def tiny(board, since, ply, clock):
    from conv import kernels

    _, legal, count = kernels.derive_batch(board, since, ply, clock)
    zeros = torch.zeros(board.shape[0], 648, device=board.device)
    rank = -ply.float() / 10
    return zeros, zeros, zeros[:, 0], zeros[:, 0], legal, count, rank, rank / 10


def opening(clock=2, ply=4, extra_piece=False):
    board = np.zeros(81, np.int8)
    board[10], board[70] = 1, 4
    if extra_piece:
        board[11] = 1
    return State(board, 0, ply, clock)


@pytest.mark.parametrize("penalty", [float("nan"), float("inf"), -float("inf"), 1.01, -1.01])
def test_probe_rejects_unbounded_clock_returns_before_creating_artifact(tmp_path, penalty):
    cfg = small_config()
    cfg.rules.clock_penalty = penalty
    path = tmp_path / "probe.json"
    with pytest.raises(ValueError, match="finite clock_penalty"):
        probe(cfg, None, "p", 0, 1, 1, 1, "cpu", estimand="action_mode", progress_path=path)
    assert not path.exists()


@pytest.mark.parametrize("penalty", [-1.0, 0.0, 1.0])
def test_probe_return_support_accepts_bounded_endpoints(penalty):
    validate_return_support(penalty)


def test_cpu_game_uses_given_opening_and_preserves_behavior_when_recording_candidates():
    cfg = small_config()
    state = opening()
    plain = Game(cfg, tiny, state, 91, "cpu").play("plain", "p", record=True)
    pool = CandidatePool(35, "recorded")
    instrumented = Game(cfg, tiny, state, 91, "cpu").play("recorded", "p", pool, record=True)
    for field in ("initial", "trajectory", "first_q", "first_action", "outcome", "plies", "end_reason"):
        assert plain[field] == instrumented[field]
    assert plain["initial"]["ply"] == 4 and plain["plies"] == 2
    assert pool.selections()["sources"]["tree"] > 0
    assert not plain["censored"]


def test_root_context_reproduces_tree_but_keeps_future_streams_independent():
    cfg, state = small_config(), opening()
    root = Game(cfg, tiny, state, 90, "cpu", context_seed=17).prepare("shared", "p")
    a = Game(cfg, tiny, state, 91, "cpu", context_seed=17).play("a", "p", expected_context=root)
    b = Game(cfg, tiny, state, 92, "cpu", context_seed=17).play("b", "p", expected_context=root)
    assert a["context"]["tree_sha256"] == b["context"]["tree_sha256"] == root["tree_sha256"]
    assert not set(a["streams"].values()) & set(b["streams"].values())
    assert paired_observation(a, b, "root_context") == 0
    altered = {**root, "tree_sha256": "changed_tree"}
    with pytest.raises(ValueError, match="tree_sha256"):
        Game(cfg, tiny, state, 93, "cpu", context_seed=17).play("bad", "p", expected_context=altered)


def test_cpu_game_applies_clock_penalty_and_marks_ply_cap_censored():
    for clock in (1, 2):
        result = Game(small_config(clock), tiny, opening(clock, extra_piece=True), 4, "cpu").play("e", "p")
        assert result["outcome"] == pytest.approx(-0.05)
        assert result["plies"] == clock
    capped = Game(small_config(10, max_plies=5), tiny, opening(10), 4, "cpu").play("e", "p")
    assert capped["censored"] and capped["outcome"] is None and capped["residual"] is None


@pytest.mark.parametrize("pieces,plies,outcome", [({71: 1, 0: 5, 40: 4}, 1, 1), ({40: 1, 1: 4}, 2, -1)])
def test_cpu_game_decisive_returns_use_opening_mover_perspective(pieces, plies, outcome):
    board = np.zeros(81, np.int8)
    for square, piece in pieces.items():
        board[square] = piece
    cfg = small_config(10)
    cfg.search.candidates = 16
    cfg.search.full_fraction = 1.0
    result = Game(cfg, tiny, State(board, 0, 4, 10), 5, "cpu").play("decisive", "p")
    assert result["end_reason"] == 1 and not result["censored"]
    assert result["plies"] == plies and result["outcome"] == outcome
    assert result["residual"] == result["first_q"] - outcome


@pytest.mark.parametrize("estimand", ["state_policy", "root_context", "action_mode"])
def test_full_probe_on_cpu_has_disjoint_independent_episodes_and_finite_summary(estimand):
    clock = 2 if estimand == "state_policy" else 1
    cfg = small_config(clock)
    if estimand != "state_policy":
        cfg.search.temperature = 0
        cfg.search.temperature_plies = cfg.rules.max_plies
    result = probe(
        cfg,
        tiny,
        "cpu_fixture",
        8,
        2,
        1,
        1,
        "cpu",
        [opening(clock), opening(clock)],
        estimand=estimand,
        max_root_attempts=128,
    )
    json.dumps(result, allow_nan=False)
    episodes = result["episodes"]
    assert len({e["episode_id"] for e in episodes}) == len(episodes)
    assert len({e["seed"] for e in episodes}) == len(episodes)
    assert len({s for e in episodes for s in e["streams"].values()}) == 3 * len(episodes)
    assert all(not e["censored"] for e in episodes)
    assert len(result["comparisons"]) == 6
    assert all(c["complete_discovery_games"] == 2 for c in result["comparisons"].values())
    assert all(c["mean_lift"] == pytest.approx(0) for c in result["comparisons"].values())
    for record in result["games"]:
        for measured in record["measurements"].values():
            for anchor in measured["future_anchors"]:
                for pair in anchor["confirmation"]["pairs"]:
                    assert measured["base_episode"] not in pair["episodes"]
                    assert record["game"] not in pair["episodes"]


def test_capped_discovery_never_launches_confirmations(monkeypatch):
    calls = []

    def capped(self, identity, protocol, pool=None, record=False, expected_context=None):
        calls.append(identity)
        assert pool is not None
        pool.offer([candidate(1)])
        return dict(episode_id=identity, censored=True, plies=1, scheduled_simulations=1, seconds=0)

    monkeypatch.setattr(Game, "__init__", lambda *a, **kw: None)
    monkeypatch.setattr(Game, "play", capped)
    result = probe(small_config(), tiny, "p", 0, 1, 1, 1, "cpu", [opening()], estimand="state_policy")
    assert calls == ["discovery0"]
    assert result["cost"]["censored_discoveries"] == 1
    assert result["games"][0]["measurements"] == {}
    assert all(v["complete_discovery_games"] == 0 for v in result["comparisons"].values())
    assert all(v["mean_lift"] is None for v in result["comparisons"].values())
    assert all(v["finite_sample_completion_range"] == [-8, 8] for v in result["comparisons"].values())


def test_coarse_attempt_cap_keeps_missing_pairs_and_never_continues_rejected_roots(monkeypatch):
    prepare = Game.prepare

    def reject(self, identity, protocol, full=None):
        result = prepare(self, identity, protocol, full)
        if "/matched_root" in identity:
            assert full is not None
            result["action"] = -1
        return result

    monkeypatch.setattr(Game, "prepare", reject)
    result = probe(
        small_config(1), tiny, "p", 0, 1, 1, 1, "cpu", [opening(1)], estimand="action_mode", max_root_attempts=2
    )
    coverage = result["coverage"]
    assert coverage["requested_pairs"] > 0
    assert coverage["attempt_capped_pairs"] == coverage["requested_pairs"]
    assert coverage["complete_pairs"] == 0
    assert coverage["matched_root_attempts"] == 2 * coverage["requested_pairs"]
    assert result["cost"]["root_searches"] == 3 * coverage["requested_pairs"]
    assert all("/pair" not in e["episode_id"] for e in result["episodes"])
    assert all(v["mean_lift"] is None for v in result["comparisons"].values())


def test_atomic_progress_preserves_existing_snapshot_on_failed_replacement(tmp_path, monkeypatch):
    path = tmp_path / "probe.json"
    atomic_json(path, {"value": 1}, create=True)
    with pytest.raises(FileExistsError):
        atomic_json(path, {"value": 2}, create=True)

    def fail(*args):
        raise OSError("interrupted replacement")

    monkeypatch.setattr("conv.probe_control.os.replace", fail)
    with pytest.raises(OSError, match="interrupted"):
        atomic_json(path, {"value": 3})
    assert json.loads(path.read_text()) == {"value": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_interrupted_probe_resumes_completed_work_and_rejects_corrupt_cache(tmp_path, monkeypatch):
    path = tmp_path / "probe.json"
    play, calls = Game.play, []

    def interrupt(self, identity, *args, **kwargs):
        calls.append(identity)
        if len(calls) == 2:
            raise RuntimeError("fixture interruption")
        return play(self, identity, *args, **kwargs)

    kwargs = dict(estimand="state_policy", progress_path=path)
    monkeypatch.setattr(Game, "play", interrupt)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        probe(small_config(), tiny, "p", 0, 1, 1, 1, "cpu", [opening()], **kwargs)
    saved = json.loads(path.read_text())
    assert saved["status"] == "running" and len(saved["episodes"]) == 1
    with pytest.raises(ValueError, match="do not match"):
        probe(small_config(), tiny, "p", 1, 1, 1, 1, "cpu", [opening()], resume=True, **kwargs)
    corrupt = deepcopy(saved)
    corrupt["episodes"][0]["state_key"] = "incorrect"
    atomic_json(path, corrupt)
    with pytest.raises(ValueError, match="cached episode"):
        probe(small_config(), tiny, "p", 0, 1, 1, 1, "cpu", [opening()], resume=True, **kwargs)
    atomic_json(path, saved)
    result = probe(small_config(), tiny, "p", 0, 1, 1, 1, "cpu", [opening()], resume=True, **kwargs)
    assert result["status"] == "complete"
    assert calls.count("discovery0") == 1
    assert len({e["episode_id"] for e in result["episodes"]}) == len(result["episodes"])

    def no_play(*args, **kwargs):
        pytest.fail("completed probe launched another episode")

    monkeypatch.setattr(Game, "play", no_play)
    assert probe(small_config(), None, "p", 0, 1, 1, 1, "cpu", [opening()], resume=True, **kwargs) == result
    with pytest.raises(ValueError, match="do not match"):
        Progress(path, {**result["request"], "protocol": "changed"}, True)


def test_coarse_resume_reuses_rejected_attempts_and_matches_uninterrupted_probe(tmp_path, monkeypatch):
    attempts, stop = [], [True]

    class FixtureGame:
        def __init__(self, cfg, evaluate, state, seed, device, context_seed=None):
            self.cfg, self.state, self.seed = cfg, state, seed
            self.context_seed = seed_for(seed, "root_context") if context_seed is None else context_seed

        def prepare(self, identity, protocol, full=None):
            attempts.append(identity)
            if "/matched_root2" in identity and stop[0]:
                stop[0] = False
                raise RuntimeError("root attempt interruption")
            streams = {k: seed_for(self.context_seed, "root", k) for k in ("search", "mode")}
            if "/matched_root" in identity:
                assert full is not None
            else:
                assert full is None
                full = bool(np.random.default_rng(streams["mode"]).random() < self.cfg.search.full_fraction)
            return dict(
                context_id=identity,
                protocol=protocol,
                state_key=self.state.key().hex(),
                initial=state_record(self.state),
                seed=self.context_seed,
                streams=streams,
                full=full,
                action=8 if identity.endswith("matched_root1") else 7,
                q=0.5,
                tree_sha256=f"tree{self.context_seed}",
                scheduled_simulations=1,
                seconds=0.0,
            )

        def play(self, identity, protocol, pool=None, record=False, expected_context=None):
            context = expected_context or self.prepare(f"{identity}/root", protocol)
            if pool is not None:
                for node, source in enumerate(("played", "tree", "tree")):
                    pool.offer(
                        [dict(state=state_record(self.state), source=source, occurrence=[0, node], rank=0, regret=0)]
                    )
            return dict(
                episode_id=identity,
                protocol=protocol,
                seed=self.seed,
                streams={k: seed_for(self.seed, "future", k) for k in ("search", "mode", "environment")},
                context=context,
                state_key=self.state.key().hex(),
                initial=state_record(self.state),
                first_q=context["q"],
                first_action=context["action"],
                outcome=0.0,
                residual=0.5,
                plies=1,
                end_reason=2,
                censored=False,
                scheduled_simulations=1,
                seconds=0.0,
                trajectory=[state_record(self.state)] if record else None,
            )

    monkeypatch.setattr("conv.probe_control.Game", FixtureGame)
    path = tmp_path / "probe.json"
    kwargs = dict(estimand="action_mode", max_root_attempts=2)
    with pytest.raises(RuntimeError, match="root attempt interruption"):
        probe(small_config(), tiny, "p", 0, 1, 1, 1, "cpu", [opening()], progress_path=path, **kwargs)
    saved = json.loads(path.read_text())
    assert len(saved["root_searches"]) == 2
    assert saved["root_searches"][-1]["action"] == 8
    saved_pool = saved["episodes"][0]["candidate_pool"]
    assert [len(batch) for batch in candidate_batches(saved_pool)] == [1, 1, 1]
    assert saved_pool["occurrences"] == 3 and saved_pool["distinct_states"] == 1
    corrupt = deepcopy(saved)
    corrupt["episodes"][0]["candidate_pool"]["payload"]["sha256"] = "corrupt"
    atomic_json(path, corrupt)
    with pytest.raises(ValueError, match="invalid candidate payload"):
        probe(small_config(), tiny, "p", 0, 1, 1, 1, "cpu", [opening()], progress_path=path, resume=True, **kwargs)
    corrupt = deepcopy(saved)
    corrupt["root_searches"][-1]["seed"] += 1
    atomic_json(path, corrupt)
    with pytest.raises(ValueError, match="cached root"):
        probe(small_config(), tiny, "p", 0, 1, 1, 1, "cpu", [opening()], progress_path=path, resume=True, **kwargs)
    atomic_json(path, saved)
    resumed = probe(
        small_config(), tiny, "p", 0, 1, 1, 1, "cpu", [opening()], progress_path=path, resume=True, **kwargs
    )
    first_rejection = "discovery0/candidate0/opening/pair0/matched_root1"
    assert attempts.count(first_rejection) == 1
    assert attempts.count("discovery0/root") == 1
    assert resumed["episodes"][0]["candidate_pool"] == saved_pool
    assert "payload" not in resumed["games"][0]["candidates"]
    assert resumed["coverage"]["complete_pairs"] == 2
    assert resumed["coverage"]["matched_root_attempts"] == 4
    assert len(resumed["root_searches"]) == 6
    calls = len(attempts)
    assert (
        probe(small_config(), None, "p", 0, 1, 1, 1, "cpu", [opening()], progress_path=path, resume=True, **kwargs)
        == resumed
    )
    assert len(attempts) == calls
    uninterrupted = probe(small_config(), tiny, "p", 0, 1, 1, 1, "cpu", [opening()], **kwargs)
    resumed.pop("elapsed_seconds")
    uninterrupted.pop("elapsed_seconds")
    assert resumed == uninterrupted
    completed = json.loads(path.read_text())
    corrupt = deepcopy(completed)
    corrupt["games"][0]["candidates"]["occurrences"] += 1
    atomic_json(path, corrupt)
    with pytest.raises(ValueError, match="candidate report"):
        probe(small_config(), None, "p", 0, 1, 1, 1, "cpu", [opening()], progress_path=path, resume=True, **kwargs)
    completed["episodes"][0]["candidate_pool"]["payload"]["sha256"] = "corrupt"
    atomic_json(path, completed)
    with pytest.raises(ValueError, match="invalid candidate payload"):
        probe(small_config(), None, "p", 0, 1, 1, 1, "cpu", [opening()], progress_path=path, resume=True, **kwargs)
    completed = json.loads(json.dumps(saved))
    completed["request"]["version"] -= 1
    atomic_json(path, completed)
    with pytest.raises(ValueError, match="do not match"):
        probe(small_config(), None, "p", 0, 1, 1, 1, "cpu", [opening()], progress_path=path, resume=True, **kwargs)
