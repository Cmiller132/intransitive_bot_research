"""Independent target checks and repeated-opening evidence for RGSC diagnostics."""

from __future__ import annotations

from collections import defaultdict
from math import ceil, comb

import numpy as np


def immediate_win_audit(data: dict) -> tuple[dict, np.ndarray]:
    """Count stored exact wins lost at candidate admission or move selection, per played row."""
    moves, tactics, candidates, action, full, outcome, known = (
        np.asarray(data[k]) for k in ("moves", "tactics", "candidates", "action", "target_ok", "outcome", "outcome_ok")
    )
    if (
        moves.ndim != 2
        or tactics.shape != moves.shape
        or candidates.ndim != 2
        or len(candidates) != len(moves)
        or any(v.shape != (len(moves),) for v in (action, full, outcome, known))
    ):
        raise ValueError("inconsistent saved move, tactic, candidate or outcome dimensions")
    wins = ((tactics & 1) != 0) & (moves >= 0)
    present = wins.any(1)
    retained, chosen = np.zeros(len(moves), bool), np.zeros(len(moves), bool)
    rows = np.flatnonzero(present)
    # Bound the temporary candidate-by-move comparison independently of window size.
    for start in range(0, len(rows), 4096):
        ix = rows[start : start + 4096]
        retained[ix] = ((moves[ix, :, None] == candidates[ix, None, :]) & wins[ix, :, None]).any((1, 2))
        chosen[ix] = ((moves[ix] == action[ix, None]) & wins[ix]).any(1)
    omitted = present & ~retained
    result = {"candidate_slots": candidates.shape[1], "by_mode": {}}
    for name, mask in (("all", np.ones(len(moves), bool)), ("full", full.astype(bool)), ("cheap", ~full.astype(bool))):
        affected = omitted & mask
        n = int((present & mask).sum())
        result["by_mode"][name] = {
            "rows": int(mask.sum()),
            "immediate_win_opportunities": n,
            "omitted_from_candidates": int(affected.sum()),
            "omitted_fraction_of_opportunities": float(affected.sum() / n) if n else None,
            "non_immediate_choices": int((present & ~chosen & mask).sum()),
            "retained_but_not_chosen": int((present & retained & ~chosen & mask).sum()),
            "chosen_without_candidate": int((chosen & ~retained & mask).sum()),
            "omitted_eventual_outcomes": {
                "wins": int((affected & known & (outcome == 1)).sum()),
                "draws": int((affected & known & (outcome == 0)).sum()),
                "losses": int((affected & known & (outcome == -1)).sum()),
                "unknown": int((affected & ~known).sum()),
            },
        }
    return result, present


def suffix_targets(q: np.ndarray, terminal_return: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct mover-relative terminal returns and both trajectory targets."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or not np.isfinite(terminal_return):
        raise ValueError("require a nonempty finite prediction vector and terminal return")
    z = terminal_return * np.where((len(q) - 1 - np.arange(len(q))) % 2, -1, 1)
    excess, squared = np.empty_like(q), np.empty_like(q)
    excess_mean = squared_mean = 0.0
    for count, i in enumerate(range(len(q) - 1, -1, -1), 1):
        excess_mean += (2 * q[i] * (q[i] - z[i]) - excess_mean) / count
        squared_mean += ((q[i] - z[i]) ** 2 - squared_mean) / count
        excess[i], squared[i] = excess_mean, squared_mean
    return z, excess, squared


def pair_bias(residuals: np.ndarray) -> float:
    """Unclipped pair-product estimate of squared mean residual for IID observations."""
    r = np.asarray(residuals, dtype=np.float64)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("at least two finite residuals are required")
    return float((r.sum() ** 2 - r @ r) / (len(r) * (len(r) - 1)))


def teacher_excess(residual: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    """Signed excess-risk observation for a teacher independent of the outcome."""
    r, m = np.broadcast_arrays(np.asarray(residual, dtype=np.float64), np.asarray(teacher, dtype=np.float64))
    if not np.isfinite(r).all() or not np.isfinite(m).all():
        raise ValueError("residuals and teacher predictions must be finite")
    return 2 * m * r - m**2


def top_quartile_weights(scores: np.ndarray, actors: np.ndarray | None = None) -> np.ndarray:
    """Select exactly a quarter of the observations, sharing the boundary mass across ties."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("require a nonempty finite score vector")
    if actors is not None:
        actors = np.asarray(actors)
        if actors.shape != scores.shape:
            raise ValueError("one actor identifier per score is required")
        weights = np.empty_like(scores)
        for actor in np.unique(actors):
            mask = actors == actor
            weights[mask] = top_quartile_weights(scores[mask])
        return weights
    threshold = np.sort(scores)[-ceil(len(scores) / 4)]
    weights = (scores > threshold).astype(float)
    tied = scores == threshold
    weights[tied] = (len(scores) / 4 - weights.sum()) / tied.sum()
    return weights


def exact_examples() -> dict:
    """Finite-distribution examples; these are analytic checks, not measured game strength."""
    cases = []
    for name, q, mu in (
        ("calibrated_balanced", 0.0, 0.0),
        ("calibrated_favorable", 0.5, 0.5),
        ("missed_favorable", 0.0, 0.8),
        ("underconfident", 0.3, 0.8),
        ("overconfident", 0.8, 0.3),
    ):
        z = np.array([-1.0, 1.0])
        p = (1 + z * mu) / 2
        r = q - z
        excess = 2 * q * r
        cases.append(
            dict(
                name=name,
                q=q,
                outcome_mean=mu,
                squared_bias=(q - mu) ** 2,
                outcome_variance=1 - mu**2,
                squared_error=float(p @ r**2),
                current_target=float(p @ excess),
                exponential_target=float(p @ np.exp(excess)),
                paired_target=float(np.outer(p, p).ravel() @ np.outer(r, r).ravel()),
                independent_exact_teacher=float(p @ teacher_excess(r, q - mu)),
                leaked_teacher=float(p @ teacher_excess(r, r)),
            )
        )
    clipping = []
    for n in (2, 4, 8, 16, 32, 64):
        values = np.array([((2 * k - n) ** 2 - n) / (n * (n - 1)) for k in range(n + 1)])
        probabilities = np.array([comb(n, k) / 2**n for k in range(n + 1)])
        clipping.append(
            dict(
                observations=n,
                raw_mean=float(probabilities @ values),
                clamped_mean=float(probabilities @ np.maximum(values, 0)),
            )
        )
    return {
        "scope": "Exact synthetic distributions, fixed Q, independent outcomes; not live-run estimates.",
        "targets": cases,
        "calibrated_fair_outcome_clipping": clipping,
        "random_length": {
            "root_bias": 0.5,
            "positive_outcome_length": 1,
            "negative_outcome_length": 2,
            "same_outcome_suffix_proxy": (-0.75 + 1.25 / 2) / 2,
            "independent_anchor_utility": (0.25 + 0.25 / 2) / 2,
        },
        "action_cancellation": {
            "opposite_action_residual_means": [-0.5, 0.5],
            "state_mean_residual_squared": 0.0,
            "mean_action_bias_squared": 0.25,
        },
        "action_dependent_length": {
            "root_action_biases": [1.0, 0.0],
            "action_probabilities": [0.5, 0.5],
            "trajectory_lengths": [1, 2],
            "next_state_utility": 0.0,
            "base_action_squared_bias_suffix": 0.5 * 1 + 0.5 * 0,
            "state_utility_then_suffix": 0.5 * 0.5 + 0.5 * (0.5 + 0) / 2,
        },
        "actor_drift": {
            "old_residual": -0.5,
            "new_residual": 0.5,
            "mixed_pair_product": -0.25,
            "current_squared_bias": 0.25,
        },
        "ranking_gradient": [
            {"wrong_score_gap": gap, "correct_state_gradient": float(1 / (1 + np.exp(gap)) - 1 / (1 + np.exp(gap - 1)))}
            for gap in (0, 20, 80)
        ],
        "ranking_noise": {
            "candidate_expected_labels": [0.0, 0.0],
            "noisy_candidate_labels": [-0.5, 1.5],
            "probabilities": [0.75, 0.25],
            "expected_loss_select_either_exclusively": 0.0,
            "expected_loss_equal_mixture": float(-np.array([0.75, 0.25]) @ np.log((1 + np.exp([-0.5, 1.5])) / 2)),
        },
    }


class TargetAudit:
    """Validate saved labels and retain small opening records for disjoint confirmation checks."""

    def __init__(self):
        self.openings: list[dict] = []
        self.iterations: list[dict] = []
        self.selection_lifts: list[dict] = []

    def consume(self, completed: list[dict], iteration: int) -> None:
        rows = mismatches = outcome_mismatches = 0
        max_error = 0.0
        comparisons = defaultdict(list)
        local = defaultdict(lambda: [0, 0.0, 0.0])
        shortening = []
        for game in completed:
            q, saved = game["q"], game["regret"]
            z, excess, squared = suffix_targets(q, game["terminal_return"])
            for name in ("regret", "rank", "outcome", "action", "full", "actor", "immediate_win"):
                values = np.asarray(game[name])
                if values.shape != q.shape or not np.isfinite(values).all():
                    raise ValueError(f"malformed or nonfinite saved {name}")
            error = np.abs(excess - saved)
            rows += len(q)
            mismatches += int(np.sum(error > 1e-6))
            max_error = max(max_error, float(error.max()))
            sign = np.sign(game["terminal_return"]) if abs(game["terminal_return"]) > 0.5 else 0
            outcome_mismatches += int(np.sum(game["outcome"] != sign * np.where(z * sign >= 0, 1, -1)))
            r = q - z
            first_wins = np.flatnonzero(game["immediate_win"])
            stop = int(first_wins[0]) + 1 if len(first_wins) else len(q)
            corrected = excess
            if len(first_wins):
                shortened_q = q[:stop].copy()
                shortened_q[-1] = 1.0
                _, corrected, _ = suffix_targets(shortened_q, 1.0)
                shortening.append(
                    {
                        "removed_rows": len(q) - stop,
                        "opening_label_change": float(corrected[0] - excess[0]),
                        "prefix_label_absolute_change": float(np.abs(corrected - excess[:stop]).sum()),
                        "prefix_rows": stop,
                        "outcome_disagrees": bool(z[stop - 1] != 1.0),
                    }
                )
            for name, mask in (
                ("all", np.ones(len(q), bool)),
                ("near_zero_q", np.abs(q) <= 0.1),
                ("positive_local_excess", 2 * q * r > 0),
                ("negative_local_excess", 2 * q * r < 0),
            ):
                local[name][0] += int(mask.sum())
                local[name][1] += float(np.sum((2 * q * r)[mask]))
                local[name][2] += float(np.sum((r**2)[mask]))
            if not game["replayed"]:
                rank = game["rank"].astype(np.float64)
                p = np.exp(rank - rank.max())
                p /= p.sum()
                comparisons["current"].append(float(p @ excess - excess.mean()))
                comparisons["paper_squared"].append(float(p @ squared - squared.mean()))
                comparisons["opening_local_squared"].append(float(p @ r**2 - np.mean(r**2)))
                prefix_p = np.exp(rank[:stop] - rank[:stop].max())
                prefix_p /= prefix_p.sum()
                original_lift = float(prefix_p @ excess[:stop] - excess[:stop].mean())
                corrected_lift = float(prefix_p @ corrected - corrected.mean())
                comparisons["first_win_prefix_original"].append(original_lift)
                comparisons["first_win_prefix_corrected"].append(corrected_lift)
                comparisons["first_win_prefix_lift_change"].append(corrected_lift - original_lift)
            self.openings.append(
                {
                    "key": game["key"],
                    "actor": int(game["actor"][0]),
                    "end_actor": int(game["actor"][-1]),
                    "env": game["env"],
                    "step": game["step"],
                    "end_iteration": iteration,
                    "replayed": game["replayed"],
                    "q": float(q[0]),
                    "z": float(z[0]),
                    "residual": float(r[0]),
                    "rank": float(game["rank"][0]),
                    "current": float(saved[0]),
                    "paper": float(squared[0]),
                    "action": int(game["action"][0]),
                    "full": bool(game["full"][0]),
                }
            )
        self.iterations.append(
            {
                "iteration": iteration,
                "complete_games": len(completed),
                "label_rows": rows,
                "label_mismatch_rows": mismatches,
                "label_max_abs_error": max_error,
                "outcome_mismatch_rows": outcome_mismatches,
                "first_win_shortening": {
                    "games_with_immediate_win": len(shortening),
                    "games_with_delay": sum(s["removed_rows"] > 0 for s in shortening),
                    "removed_rows": sum(s["removed_rows"] for s in shortening),
                    "outcome_disagreements": sum(s["outcome_disagrees"] for s in shortening),
                    "opening_label_change_mean": float(np.mean([s["opening_label_change"] for s in shortening]))
                    if shortening
                    else None,
                    "prefix_label_mean_absolute_change": sum(s["prefix_label_absolute_change"] for s in shortening)
                    / max(1, sum(s["prefix_rows"] for s in shortening)),
                    "delay_quantiles_0_50_90_99_100": np.quantile(
                        [s["removed_rows"] for s in shortening if s["removed_rows"] > 0], [0, 0.5, 0.9, 0.99, 1]
                    ).tolist()
                    if any(s["removed_rows"] > 0 for s in shortening)
                    else None,
                },
                "fresh_played_selection": {k: float(np.mean(v)) for k, v in comparisons.items()},
                "local_observations": {
                    k: {"rows": v[0], "mean_excess": v[1] / max(1, v[0]), "mean_squared_error": v[2] / max(1, v[0])}
                    for k, v in local.items()
                },
            }
        )
        self.selection_lifts.append(dict(comparisons))

    def report(self, bootstrap: int, seed: int) -> dict:
        from .diagnose_control import correlation, interval

        iterations = []
        for row, comparisons in zip(self.iterations, self.selection_lifts, strict=True):
            intervals = {}
            for name, lifts in comparisons.items():
                values = np.array(lifts)
                rng = np.random.default_rng(seed + row["iteration"])
                samples = rng.integers(len(values), size=(bootstrap, len(values)))
                intervals[name] = interval(values[samples].mean(1))
            iterations.append(
                {**row, "fresh_played_selection_ci95": intervals, "fresh_games": len(comparisons.get("current", []))}
            )
        repeated = {}
        for context in (False, True):
            groups = defaultdict(list)
            excluded_cross_actor = 0
            for o in self.openings:
                if not o["replayed"] or o["actor"] != o["end_actor"]:
                    excluded_cross_actor += int(o["replayed"] and o["actor"] != o["end_actor"])
                    continue
                key = (o["key"], o["actor"])
                if context:
                    key += (o["action"], o["full"])
                groups[key].append(o)
            estimates, keys, actors = [], [], []
            for key, observations in groups.items():
                if len(observations) < 4:
                    continue
                observations.sort(key=lambda o: (o["step"], o["env"]))
                a, b = np.array_split(np.arange(len(observations)), 2)
                first, confirm = [observations[i] for i in a], [observations[i] for i in b]
                estimates.append(
                    [
                        np.mean([o["rank"] for o in first]),
                        np.mean([o["current"] for o in first]),
                        np.mean([o["paper"] for o in first]),
                        pair_bias([o["residual"] for o in first]),
                        pair_bias([o["residual"] for o in confirm]),
                        len(observations),
                    ]
                )
                keys.append(key[0])
                actors.append(key[1])
            summary = {
                "groups": len(groups),
                "groups_with_disjoint_two_plus_two": len(estimates),
                "replay_games_excluded_cross_actor": excluded_cross_actor,
                "scope": "Replays only, wholly within one actor window; disjoint discovery/confirmation games. "
                "State/action groups also fix selected action and full/cheap opening search. "
                "Selection takes the top quarter separately within each actor; pooled Spearman is descriptive. "
                "Shared batch search-mode randomness and adaptive selection remain; not an IID proof.",
            }
            if estimates:
                values = np.array(estimates)
                # Resample exact opening keys, carrying every actor/context for that opening together.
                unique = {key: i for i, key in enumerate(dict.fromkeys(keys))}
                cluster = np.array([unique[k] for k in keys])
                rng = np.random.default_rng(seed)
                counts = np.bincount(cluster)
                samples = rng.integers(len(unique), size=(bootstrap, len(unique)))
                confirmation_sum = np.bincount(cluster, weights=values[:, 4])
                summary.update(
                    opening_clusters=len(unique),
                    observations=int(values[:, 5].sum()),
                    confirmation_pair_mean=float(values[:, 4].mean()),
                    confirmation_pair_mean_ci95=interval(confirmation_sum[samples].sum(1) / counts[samples].sum(1)),
                    confirmation_negative_fraction=float(np.mean(values[:, 4] < 0)),
                    scores={},
                )
                for col, name in enumerate(("rank", "current_target", "paper_target", "prefix_pair_bias")):
                    score = values[:, col]
                    # Fractional tie weights avoid assigning arbitrary winners to tied scores.
                    weights = top_quartile_weights(score, np.array(actors))
                    numerator = np.bincount(cluster, weights=weights * values[:, 4], minlength=len(unique))
                    denominator = np.bincount(cluster, weights=weights, minlength=len(unique))
                    den = denominator[samples].sum(1)
                    valid = den > 0
                    delta = numerator[samples][valid].sum(1) / den[valid]
                    delta -= confirmation_sum[samples][valid].sum(1) / counts[samples][valid].sum(1)
                    summary["scores"][name] = {
                        "pooled_spearman_with_confirmation": correlation(score, values[:, 4]),
                        "within_actor_top_quartile_confirmation_lift": float(
                            weights @ values[:, 4] / weights.sum() - values[:, 4].mean()
                        ),
                        "lift_ci95": interval(delta) if len(delta) else None,
                        "bootstrap_valid_fraction": float(valid.mean()),
                    }
            repeated["state_action_search" if context else "state"] = summary
        return {
            "scope": "Saved outcomes are observational. Paper squared error is not variance-corrected truth. "
            "Pair products here measure opening residuals, not future-trajectory utility.",
            "first_win_shortening_scope": "End each saved game at its first detected immediate win, set that Q and "
            "terminal return to one, retain earlier logged Q, and recompute suffix labels. Compare original and "
            "corrected labels on the same prefix with renormalized saved ranks; games without immediate wins "
            "remain unchanged. This frozen-prediction sensitivity is not a corrected-policy rollout "
            "or strength estimate.",
            "analytic_examples": exact_examples(),
            "iterations": iterations,
            "total_label_rows": sum(v["label_rows"] for v in self.iterations),
            "total_label_mismatch_rows": sum(v["label_mismatch_rows"] for v in self.iterations),
            "total_outcome_mismatch_rows": sum(v["outcome_mismatch_rows"] for v in self.iterations),
            "repeated_openings": repeated,
        }
