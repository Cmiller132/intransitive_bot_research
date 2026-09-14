"""Independent target reconstruction and exact counterexamples for proposed control estimators."""

from itertools import product

import numpy as np
import pytest

from conv.control_audit import (
    TargetAudit,
    exact_examples,
    immediate_win_audit,
    pair_bias,
    suffix_targets,
    teacher_excess,
    top_quartile_weights,
)


@pytest.mark.parametrize("terminal", [-1.0, -0.05, 0.0, 0.05, 1.0])
def test_suffix_reconstruction_matches_direct_sums_with_alternating_perspective(terminal):
    q = np.random.default_rng(9).uniform(-1, 1, 37)
    z, current, paper = suffix_targets(q, terminal)
    assert z[-1] == terminal
    np.testing.assert_allclose(z[:-1], -z[1:])
    for i in range(len(q)):
        assert current[i] == pytest.approx(np.mean(2 * q[i:] * (q[i:] - z[i:])))
        assert paper[i] == pytest.approx(np.mean((q[i:] - z[i:]) ** 2))


def test_pair_statistic_matches_explicit_distinct_pairs_and_allows_negative():
    for n in (2, 3, 8, 31):
        r = np.random.default_rng(n).normal(size=n)
        assert pair_bias(r) == pytest.approx(
            np.mean([a * b for i, a in enumerate(r) for j, b in enumerate(r) if i != j])
        )
    assert pair_bias([-1, 1]) == -1


@pytest.mark.parametrize("bad", [[], [1], [1, np.nan], [[1, 2]]])
def test_pair_statistic_rejects_invalid_observations(bad):
    with pytest.raises(ValueError):
        pair_bias(bad)


def test_independent_teacher_identity_for_discrete_outcomes_and_arbitrary_teachers():
    z = np.array([-1, 0, 1])
    for q, probabilities, m in product(
        [-1, -0.8, -0.3, 0, 0.3, 0.8, 1],
        [[0.5, 0, 0.5], [0.1, 0.3, 0.6], [0.05, 0.9, 0.05]],
        [-0.9, -0.2, 0, 0.2, 0.9],
    ):
        p = np.array(probabilities)
        b = q - p @ z
        assert p @ teacher_excess(q - z, m) == pytest.approx(b**2 - (m - b) ** 2, abs=1e-14)


def test_exact_cases_distinguish_variance_bias_and_nonlinearity():
    report = exact_examples()
    cases = {c["name"]: c for c in report["targets"]}
    assert cases["missed_favorable"]["current_target"] == 0
    assert cases["missed_favorable"]["paired_target"] == pytest.approx(0.64)
    assert cases["underconfident"]["current_target"] == pytest.approx(-0.3)
    assert cases["calibrated_favorable"]["exponential_target"] == pytest.approx(1.575320262369, abs=1e-11)
    assert cases["calibrated_balanced"]["leaked_teacher"] == 1
    for case in cases.values():
        assert case["paired_target"] == pytest.approx(case["squared_bias"], abs=1e-14)
        assert case["independent_exact_teacher"] == pytest.approx(case["squared_bias"], abs=1e-14)
    clipping = report["calibrated_fair_outcome_clipping"]
    assert clipping[0]["clamped_mean"] == 0.5
    assert all(abs(row["raw_mean"]) < 1e-14 and row["clamped_mean"] > 0 for row in clipping)


def test_random_length_surrogate_reverses_sign_but_independent_anchor_pairs_do_not():
    naive, truth, paired = [], [], []
    for z in (-1, 1):
        teachers = np.array([0.5, 0]) if z == -1 else np.array([0.5])
        residual = np.array([0.5 - z, 0]) if z == -1 else np.array([0.5 - z])
        naive.append(np.mean(teacher_excess(residual, teachers)))
        truth.append(np.mean(teachers**2))
        anchor_values = []
        for m in teachers:
            if m == 0:
                anchor_values.append(0)
            else:
                anchor_values.append(np.mean([(0.5 - a) * (0.5 - b) for a, b in product((-1, 1), repeat=2)]))
        paired.append(np.mean(anchor_values))
    assert np.mean(naive) == -0.0625
    assert np.mean(truth) == np.mean(paired) == 0.1875


def test_shared_search_mode_and_action_cancellation_are_different_estimands():
    independent = np.mean([a * b for a, b in product((-1, 1), repeat=2)])
    shared = np.mean([a * a for a in (-1, 1)])
    assert independent == 0 and shared == 1
    assert pair_bias([1, 1, -1, -1]) == pytest.approx(-1 / 3)


def test_even_exact_action_teacher_must_integrate_actions_before_random_suffix_averaging():
    squared_bias = np.array([1.0, 0.0])
    probabilities = np.array([0.5, 0.5])
    state_utility = probabilities @ squared_bias
    base_action_average = probabilities @ np.array([squared_bias[0], (squared_bias[1] + 0) / 2])
    independent_anchor_average = probabilities @ np.array([state_utility, (state_utility + 0) / 2])
    assert base_action_average == 0.5
    assert independent_anchor_average == 0.375
    example = exact_examples()["action_dependent_length"]
    assert example["base_action_squared_bias_suffix"] == base_action_average
    assert example["state_utility_then_suffix"] == independent_anchor_average


def test_leaky_and_independent_noisy_teachers_have_opposite_biases():
    assert np.mean(teacher_excess(np.array([-1, 1]), np.array([-1, 1]))) == 1
    assert np.mean([teacher_excess(r, m) for r, m in product((-1, 1), repeat=2)]) == -1


@pytest.mark.parametrize("scores", [list(range(6)), [0] * 6, [0, 1, 2, 2, 2, 2], [1], list(range(63))])
def test_top_quartile_has_exact_mass_with_small_samples_and_ties(scores):
    weights = top_quartile_weights(scores)
    assert weights.sum() == pytest.approx(len(scores) / 4)
    assert np.all((weights >= 0) & (weights <= 1))
    if scores == list(range(6)):
        np.testing.assert_array_equal(weights, [0, 0, 0, 0, 0.5, 1])


@pytest.mark.parametrize(
    "field,bad", [("regret", [np.nan]), ("rank", [np.inf]), ("outcome", []), ("q", [np.nan]), ("immediate_win", [])]
)
def test_audit_rejects_malformed_or_nonfinite_saved_data(field, bad):
    game = dict(
        q=np.array([0.5]),
        regret=np.array([-0.5]),
        rank=np.array([0]),
        outcome=np.array([1]),
        action=np.array([0]),
        full=np.array([True]),
        actor=np.array([0]),
        immediate_win=np.array([False]),
        terminal_return=1.0,
    )
    game[field] = np.array(bad)
    with pytest.raises(ValueError):
        TargetAudit().consume([game], 0)


def test_audit_detects_corrupted_labels_and_preserves_disjoint_opening_confirmation():
    audit = TargetAudit()
    for key in range(8):
        for step in range(6):
            q = np.array([key / 10])
            _, label, _ = suffix_targets(q, 1.0)
            game = dict(
                q=q,
                regret=label.copy(),
                rank=q.copy(),
                terminal_return=1.0,
                outcome=np.ones(1),
                key=bytes([key]),
                actor=np.array([7]),
                env=step,
                step=step,
                replayed=True,
                action=np.array([key % 2]),
                full=np.array([True]),
                immediate_win=np.array([False]),
            )
            if key == step == 0:
                game["regret"][0] += 0.1
            audit.consume([game], 7)
    report = audit.report(100, 1)
    assert report["total_label_rows"] == 48
    assert report["total_label_mismatch_rows"] == 1
    assert report["total_outcome_mismatch_rows"] == 0
    repeated = report["repeated_openings"]["state"]
    assert repeated["groups_with_disjoint_two_plus_two"] == 8
    assert repeated["observations"] == 48
    assert repeated["scores"]["prefix_pair_bias"]["pooled_spearman_with_confirmation"] == pytest.approx(1)


def test_top_quartile_fixes_actor_mass_despite_opposite_score_offsets():
    weights = top_quartile_weights(np.array([100, 101, 102, 103, 0, 1]), np.array([0, 0, 0, 0, 1, 1]))
    np.testing.assert_array_equal(weights, [0, 0, 0, 1, 0, 0.5])


def test_immediate_win_audit_distinguishes_admission_selection_and_unknown_outcomes():
    data = dict(
        moves=np.array([[9, 1, -1]] * 6),
        tactics=np.array([[1, 4, 1]] * 5 + [[0, 0, 1]], dtype=np.uint8),
        candidates=np.array([[1, -1], [9, 1], [9, 1], [1, -1], [1, -1], [1, -1]]),
        action=np.array([1, 1, 9, 1, 9, 1]),
        target_ok=np.array([True, True, False, False, False, True]),
        outcome=np.array([1, 1, 1, 0, -1, 0]),
        outcome_ok=np.array([True, True, True, False, True, False]),
    )
    report, present = immediate_win_audit(data)
    np.testing.assert_array_equal(present, [True, True, True, True, True, False])
    all_rows = report["by_mode"]["all"]
    assert all_rows["rows"] == 6 and all_rows["immediate_win_opportunities"] == 5
    assert all_rows["omitted_from_candidates"] == all_rows["non_immediate_choices"] == 3
    assert all_rows["retained_but_not_chosen"] == all_rows["chosen_without_candidate"] == 1
    assert all_rows["omitted_eventual_outcomes"] == dict(wins=1, draws=0, losses=1, unknown=1)
    assert report["by_mode"]["full"]["omitted_from_candidates"] == 1
    assert report["by_mode"]["cheap"]["omitted_from_candidates"] == 2
    data["candidates"] = data["candidates"][:-1]
    with pytest.raises(ValueError, match="dimensions"):
        immediate_win_audit(data)


def test_first_win_shortening_relabels_fixed_prefix_without_using_dependent_tail_outcomes():
    q = np.array([-0.2, 0.4, -0.3, 1.0])
    z, labels, _ = suffix_targets(q, 1.0)
    game = dict(
        q=q,
        regret=labels,
        rank=np.array([0.0, 1.0, 1000.0, 1000.0]),
        terminal_return=1.0,
        outcome=z,
        key=b"win",
        actor=np.zeros(4, int),
        env=0,
        step=0,
        replayed=False,
        action=np.arange(4),
        full=np.ones(4, bool),
        immediate_win=np.array([False, True, False, True]),
    )
    audit = TargetAudit()
    audit.consume([game], 0)
    row = audit.report(10, 0)["iterations"][0]
    sensitivity = row["first_win_shortening"]
    assert sensitivity["games_with_delay"] == 1 and sensitivity["removed_rows"] == 2
    assert sensitivity["outcome_disagreements"] == 0
    _, corrected, _ = suffix_targets(np.array([-0.2, 1.0]), 1.0)
    assert sensitivity["opening_label_change_mean"] == pytest.approx(corrected[0] - labels[0])
    p = np.exp([0.0, 1.0]) / np.exp([0.0, 1.0]).sum()
    lift = row["fresh_played_selection"]
    assert lift["first_win_prefix_original"] == pytest.approx(p @ labels[:2] - labels[:2].mean())
    assert lift["first_win_prefix_corrected"] == pytest.approx(p @ corrected - corrected.mean())
    assert lift["first_win_prefix_lift_change"] == pytest.approx(
        lift["first_win_prefix_corrected"] - lift["first_win_prefix_original"]
    )


def test_first_win_shortening_uses_immediate_winner_even_when_recorded_outcome_disagrees():
    q = np.array([0.2, 0.4, 1.0])
    z, labels, _ = suffix_targets(q, 1.0)
    game = dict(
        q=q,
        regret=labels,
        rank=np.arange(3, dtype=float),
        terminal_return=1.0,
        outcome=z,
        key=b"lost",
        actor=np.zeros(3, int),
        env=0,
        step=0,
        replayed=False,
        action=np.arange(3),
        full=np.ones(3, bool),
        immediate_win=np.array([False, True, False]),
    )
    audit = TargetAudit()
    audit.consume([game], 0)
    row = audit.report(10, 0)["iterations"][0]
    assert row["first_win_shortening"]["outcome_disagreements"] == 1
    _, corrected, _ = suffix_targets(np.array([0.2, 1.0]), 1.0)
    assert corrected[0] == pytest.approx(0.24)
    assert row["first_win_shortening"]["opening_label_change_mean"] == pytest.approx(corrected[0] - labels[0])


def test_actual_ranking_loss_can_have_vanishing_corrective_gradient_on_clean_labels():
    import torch

    from conv.train import ranking_loss

    y = torch.tensor([1.0, 0.0], dtype=torch.float64)
    for case in exact_examples()["ranking_gradient"]:
        score = torch.tensor([-float(case["wrong_score_gap"]), 0.0], dtype=torch.float64, requires_grad=True)
        ranking_loss(score, y, torch.ones(2, dtype=torch.bool)).backward()
        assert score.grad[0].item() == pytest.approx(case["correct_state_gradient"], rel=1e-12, abs=1e-45)
    assert abs(score.grad[0]) < 1e-30


def test_actual_expected_ranking_loss_rewards_noise_with_equal_mean_labels():
    import torch

    from conv.train import ranking_loss

    gamma = torch.zeros(2, dtype=torch.float64)
    expected = sum(
        p * ranking_loss(gamma, torch.tensor([y, 0.0], dtype=torch.float64), torch.ones(2, dtype=torch.bool)).item()
        for p, y in ((0.75, -0.5), (0.25, 1.5))
    )
    assert expected == pytest.approx(exact_examples()["ranking_noise"]["expected_loss_equal_mixture"])
    assert expected == pytest.approx(-0.0877638812268683)


def test_direct_utility_fit_recovers_a_wrong_saturated_table_while_listwise_stays_stuck():
    import torch

    from conv.train import ranking_loss

    y = torch.tensor([1.0, 0.0], dtype=torch.float64)
    fitted = []
    for method in ("listwise", "utility_mse"):
        score = torch.tensor([-80.0, 0.0], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD([score], lr=0.1)
        for _ in range(300):
            optimizer.zero_grad()
            loss = (
                ranking_loss(score, y, torch.ones(2, dtype=torch.bool))
                if method == "listwise"
                else (score - y).square().mean()
            )
            loss.backward()
            optimizer.step()
        fitted.append(score.detach())
    assert fitted[0][0] < fitted[0][1]
    torch.testing.assert_close(fitted[1], y, rtol=0, atol=1e-10)
