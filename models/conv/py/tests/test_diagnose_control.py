"""Offline ranking diagnostics: pooled confounding, concentration and game reconstruction."""

import numpy as np
import pytest
import torch

from conv.control import Buffer, State
from conv.diagnose_control import Games, checkpoint_report, correlation, ranking_report


def test_per_game_lift_separates_between_game_score_offsets():
    games = [(np.array([10.0, 11.0]), np.array([-1.0, 0.0])), (np.array([0.0, 1.0]), np.array([1.0, 2.0]))]
    report = ranking_report(games, 100, 0)
    assert report["pooled_lift"] < -0.7
    assert report["within_game"]["1"]["lift"] == pytest.approx(0.2310585786)
    assert report["within_game"]["1"]["lift_ci95"][0] > 0


def test_extreme_scores_reduce_effective_sample_size():
    report = ranking_report([(np.array([0.0, 100.0]), np.array([1.0, -1.0]))], 10, 0)
    assert report["pooled_ess"] == pytest.approx(1)
    assert report["pooled_max_row_mass"] == pytest.approx(1)
    assert report["pooled_lift"] == pytest.approx(-1)
    assert correlation(np.array([1, 1, 2]), np.array([3, 3, 1])) == pytest.approx(-1)
    assert correlation(np.ones(3), np.arange(3)) is None


def test_actor_window_offsets_can_be_removed_without_changing_within_window_ranks():
    parts = [(np.array([10.0, 11.0]), np.array([-1.0, 0.0])), (np.array([0.0, 1.0]), np.array([1.0, 2.0]))]
    game = tuple(np.concatenate(v) for v in zip(*parts, strict=True))
    report = ranking_report([game], 10, 0, [parts])
    assert report["within_game"]["1"]["lift"] < -0.7
    assert report["within_actor_game"]["lift"] == pytest.approx(0.2310585786)


def window(ply, reply, rank, regret, replayed):
    T = len(ply)
    data = {
        "steps": T,
        "envs": 1,
        "board": torch.zeros(T, 81, dtype=torch.int8),
        "since_capture": torch.zeros(T, dtype=torch.int16),
        "clock": torch.full((T,), 100),
        "regret_ok": torch.ones(T, dtype=torch.bool),
    }
    for key, values in (("ply", ply), ("reply_ok", reply), ("rank", rank), ("regret", regret), ("replayed", replayed)):
        data[key] = torch.tensor(values)
    return data


def test_games_span_windows_and_exclude_truncated_prefix():
    games = Games(1)
    first = window([7, 8, 0], [True, False, True], [9.0, 8.0, 1.0], [9.0, 8.0, 0.3], [False] * 3)
    fresh, restarts, skipped = games.consume(first)
    assert not fresh and not restarts and skipped == 1
    second = window([1, 20, 21], [False, True, False], [2.0, 3.0, 4.0], [0.2, 0.1, 0.0], [False, True, True])
    fresh, restarts, skipped = games.consume(second)
    assert skipped == 0 and len(fresh) == 1
    np.testing.assert_allclose(fresh[0][0], [1, 2])
    np.testing.assert_allclose(fresh[0][1], [0.3, 0.2])
    assert len(restarts) == 1
    assert next(iter(restarts.values())) == pytest.approx([0.1])


def test_target_audit_preserves_q_perspective_and_actor_identity_across_windows():
    from conv.control_audit import TargetAudit, suffix_targets

    q = np.array([0.2, 0.3, 0.4])
    z, labels, _ = suffix_targets(q, 1.0)
    games = Games(1, audit=True)
    for iteration, indices in ((7, [0, 1]), (8, [2])):
        data = window(indices, [i < 2 for i in indices], q[indices], labels[indices], [False] * len(indices))
        data.update(
            iteration=iteration,
            played_q=torch.tensor(q[indices]),
            ret=torch.tensor(z[indices]),
            outcome=torch.tensor(z[indices]),
            action=torch.tensor(indices),
            target_ok=torch.ones(len(indices), dtype=torch.bool),
            moves=torch.zeros(len(indices), 1, dtype=torch.int16),
            tactics=torch.zeros(len(indices), 1, dtype=torch.uint8),
            candidates=torch.zeros(len(indices), 1, dtype=torch.int16),
            outcome_ok=torch.ones(len(indices), dtype=torch.bool),
        )
        games.consume(data)
        data["played_q"].fill_(99)
    assert len(games.completed) == 1
    np.testing.assert_array_equal(games.completed[0]["actor"], [7, 7, 8])
    np.testing.assert_allclose(games.completed[0]["q"], q)
    audit = TargetAudit()
    audit.consume(games.completed, 8)
    report = audit.report(100, 0)
    assert report["total_label_rows"] == 3 and report["total_label_mismatch_rows"] == 0
    assert report["total_outcome_mismatch_rows"] == 0


def test_unlabelled_ending_does_not_enter_ranking():
    data = window([0, 1], [True, False], [1.0, 2.0], [0.0, 0.0], [False, False])
    data["regret_ok"][:] = False
    assert Games(1).consume(data) == ([], {}, 0)


def test_capped_restart_drops_feedback_before_replay_fold(tmp_path):
    games = Games(1)
    data = window([0, 5, 0], [False, False, False], [1.0, 2.0, 0.0], [0.1, 0.0, 0.0], [False, True, False])
    data["regret_ok"][1] = False
    games.consume(data)
    state = State(np.zeros(81, np.int8), 0, 5, 100)
    assert games.dropped == {state.key()}
    buffer = Buffer(1, 0.5)
    buffer.insert(state, 0.2)
    path = tmp_path / "ckpt.pt"
    torch.save({"control": {"buffer": buffer.state()}}, path)
    report = checkpoint_report(path, {state.key(): [0.1]}, games.dropped)
    assert report["lost_keys"] == 1 and report["first_tree_keys"] == 0


def test_unlabelled_last_row_does_not_prove_a_capped_restart():
    games = Games(1)
    data = window([0, 5], [False, False], [1.0, 2.0], [0.1, 0.0], [False, True])
    data["regret_ok"][1] = False
    games.consume(data)
    assert not games.dropped


def test_first_tree_error_separates_clipping_and_counts_lost_games(tmp_path):
    state = State(np.zeros(81, np.int8), 0, 10, 100)
    buffer = Buffer(1, 0.5)
    buffer.insert(state, 0.2)
    path = tmp_path / "ckpt.pt"
    torch.save({"control": {"buffer": buffer.state()}}, path)
    report = checkpoint_report(path, {state.key(): [-0.3, 0.1], b"lost": [0.2, 0.4]})
    assert report["lost_keys"] == 1 and report["lost_games"] == 2
    assert report["first_tree"]["signed_error"] == pytest.approx(0.3)
    assert report["first_tree"]["clamped_error"] == pytest.approx(0.15)
    assert report["first_tree"]["clipping_gap"] == pytest.approx(0.15)
