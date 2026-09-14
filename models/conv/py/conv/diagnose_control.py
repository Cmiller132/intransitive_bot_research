"""CPU-only RGSC diagnostics from saved windows, logs and immutable checkpoints."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from compression import zstd

from .control import Buffer, State
from .paths import run_dir


def distribution(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    weights = np.exp((scores - scores.max()) / temperature)
    return weights / weights.sum()


def interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman correlation, with average ranks for ties."""
    ranks = []
    for values in (x, y):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        ranks.append((np.cumsum(counts) - (counts + 1) / 2)[inverse])
    if any(np.std(r) == 0 for r in ranks):
        return None
    return float(np.corrcoef(*ranks)[0, 1])


def ranking_report(
    games: list[tuple[np.ndarray, np.ndarray]],
    bootstrap: int,
    seed: int,
    segments: list[list[tuple[np.ndarray, np.ndarray]]] | None = None,
) -> dict:
    """Pooled readiness metric and per-game played-row selection, with game-cluster intervals."""
    if not games:
        return {"games": 0}
    scores = np.concatenate([s for s, _ in games]).astype(np.float64)
    labels = np.concatenate([y for _, y in games]).astype(np.float64)
    weights = distribution(scores)
    sizes = np.array([len(y) for _, y in games])
    offsets = np.r_[0, sizes.cumsum()[:-1]]
    mass = np.add.reduceat(weights, offsets)
    weighted = np.add.reduceat(weights * labels, offsets)
    sums = np.add.reduceat(labels, offsets)
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(games), size=(bootstrap, len(games)))
    pooled = weighted[samples].sum(1) / mass[samples].sum(1) - sums[samples].sum(1) / sizes[samples].sum(1)
    result = {
        "games": len(games),
        "rows": len(labels),
        "pooled_lift": float(weights @ labels - labels.mean()),
        "pooled_lift_ci95": interval(pooled),
        "pooled_ess": float(1 / (weights @ weights)),
        "pooled_game_ess": float(1 / (mass @ mass)),
        "pooled_max_row_mass": float(weights.max()),
        "pooled_max_game_mass": float(mass.max()),
        "rank_quantiles_0_50_90_99_100": np.quantile(scores, [0, 0.5, 0.9, 0.99, 1]).tolist(),
        "label_mean": float(labels.mean()),
        "spearman": correlation(scores, labels),
        "within_game": {},
    }
    for temperature in (1, 2, 4):
        lifts, ess, top, exp_lifts, uniform = [], [], [], [], []
        for s, y in games:
            p = distribution(s.astype(np.float64), temperature)
            lifts.append(float(p @ y - y.mean()))
            ess.append(float(1 / (p @ p) / len(p)))
            top.append(float(p.max()))
            exp_y = np.exp(y.astype(np.float64))
            exp_lifts.append(float(p @ exp_y - exp_y.mean()))
            uniform.append(float(y.mean()))
        lifts = np.asarray(lifts)
        result["within_game"][str(temperature)] = {
            "lift": float(lifts.mean()),
            "lift_ci95": interval(lifts[samples].mean(1)),
            "positive_lift_fraction": float(np.mean(lifts > 0)),
            "median_ess_fraction": float(np.median(ess)),
            "median_max_row_mass": float(np.median(top)),
            "max_row_mass_over_half_fraction": float(np.mean(np.array(top) > 0.5)),
            "exp_label_lift": float(np.mean(exp_lifts)),
            "uniform_label_mean": float(np.mean(uniform)),
        }
    if segments is not None:
        # Fix each actor window's mass to its row fraction, then rank within it.
        lifts = np.array(
            [
                sum(len(y) * (distribution(s.astype(np.float64)) @ y - y.mean()) for s, y in parts)
                / sum(len(y) for _, y in parts)
                for parts in segments
            ]
        )
        result["within_actor_game"] = {
            "lift": float(lifts.mean()),
            "lift_ci95": interval(lifts[samples].mean(1)),
        }
    return result


class Games:
    """Reconstruct complete games across windows using the persisted same-game reply mask."""

    def __init__(self, envs: int, audit: bool = False):
        self.pending: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(envs)]
        self.opening: list[dict | None] = [None] * envs
        self.partial = np.ones(envs, dtype=bool)
        self.fresh_segments: list[list[tuple[np.ndarray, np.ndarray]]] = []
        self.dropped: set[bytes] = set()
        self.audit = audit
        self.details: list[list[dict]] = [[] for _ in range(envs)]
        self.completed: list[dict] = []

    def consume(self, data: dict) -> tuple[list, dict, int]:
        T, N = data["steps"], data["envs"]
        if N != len(self.pending):
            raise ValueError("environment count changed inside the requested range")

        def column(name):
            return data[name].reshape(T, N, *data[name].shape[1:]).numpy()

        rank, regret, ok = (column(k) for k in ("rank", "regret", "regret_ok"))
        reply, replayed, ply = (column(k) for k in ("reply_ok", "replayed", "ply"))
        fresh, restarts, skipped = [], defaultdict(list), 0
        self.fresh_segments = []
        self.dropped = set()
        self.completed = []
        columns = {k: column(k) for k in ("played_q", "ret", "action", "outcome", "target_ok")} if self.audit else {}
        if self.audit:
            from .control_audit import immediate_win_audit

            self.search_selection, immediate_win = immediate_win_audit(data)
            columns["immediate_win"] = immediate_win.reshape(T, N)
        for n in range(N):
            start = 0
            ends = np.flatnonzero(~reply[:, n]).tolist()
            if not ends or ends[-1] != T - 1:
                ends.append(T - 1)
            for end in ends:
                if self.opening[n] is None:
                    self.partial[n] &= ply[start, n] != 0
                    state = State(
                        column("board")[start, n].copy(),
                        int(column("since_capture")[start, n]),
                        int(ply[start, n]),
                        int(column("clock")[start, n]),
                    )
                    self.opening[n] = {
                        "key": state.key(),
                        "replayed": bool(replayed[start, n]),
                        "regret": float(regret[start, n]),
                        "env": n,
                        "step": start,
                    }
                self.pending[n].append((rank[start : end + 1, n].copy(), regret[start : end + 1, n].copy()))
                if self.audit:
                    self.details[n].append(
                        {
                            **{k: v[start : end + 1, n].copy() for k, v in columns.items()},
                            "actor": np.full(end - start + 1, data["iteration"], dtype=np.int32),
                        }
                    )
                if not reply[end, n]:
                    if not self.partial[n] and ok[end, n]:
                        opening = self.opening[n]
                        if self.audit:
                            detail = {k: np.concatenate([d[k] for d in self.details[n]]) for k in self.details[n][0]}
                            self.completed.append(
                                {
                                    **opening,
                                    "q": detail["played_q"],
                                    "terminal_return": float(detail["ret"][-1]),
                                    "action": detail["action"],
                                    "full": detail["target_ok"],
                                    "outcome": detail["outcome"],
                                    "actor": detail["actor"],
                                    "immediate_win": detail["immediate_win"],
                                    "rank": np.concatenate([p[0] for p in self.pending[n]]),
                                    "regret": np.concatenate([p[1] for p in self.pending[n]]),
                                }
                            )
                        if opening["replayed"]:
                            restarts[opening["key"]].append(opening["regret"])
                        else:
                            fresh.append(tuple(np.concatenate(v) for v in zip(*self.pending[n], strict=True)))
                            self.fresh_segments.append(self.pending[n])
                    elif self.partial[n] and ok[end, n]:
                        skipped += 1
                    elif end < T - 1 and not self.partial[n] and self.opening[n]["replayed"]:
                        self.dropped.add(self.opening[n]["key"])
                    self.pending[n] = []
                    self.details[n] = []
                    self.opening[n] = None
                    self.partial[n] = False
                start = end + 1
        return fresh, restarts, skipped


def checkpoint_report(path: Path, restarts: dict, dropped: set[bytes] | None = None) -> dict:
    """Compare next-window replay outcomes with the preceding saved buffer, including lost keys."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["control"]["buffer"]
    buffer = Buffer(len(state["count"]), 0.5)
    buffer.load(state)
    entries = list(buffer.entries.values())
    counts = np.array([e.count for e in entries])
    priorities = buffer.priorities()
    for key in dropped or ():
        buffer.drop(key)
    first = [
        (buffer.entries[k].predicted, labels)
        for k, labels in restarts.items()
        if k in buffer.entries and buffer.entries[k].tree and buffer.entries[k].count == 0
    ]
    result = {
        "checkpoint": str(path),
        "buffer_size": len(entries),
        "observations_0_1_2plus": [int(np.sum(counts == 0)), int(np.sum(counts == 1)), int(np.sum(counts >= 2))],
        "scored_zero_priority": int(np.sum((counts >= 2) & (priorities == 0))),
        "restart_keys": len(restarts),
        "lost_keys": sum(k not in buffer.entries for k in restarts),
        "restart_games": sum(len(v) for v in restarts.values()),
        "lost_games": sum(len(v) for k, v in restarts.items() if k not in buffer.entries),
        "first_tree_keys": len(first),
    }
    if first:
        prediction = np.array([p for p, _ in first])
        raw = np.array([np.mean(y) for _, y in first])
        clamped = np.array([np.maximum(y, 0).mean() for _, y in first])
        result["first_tree"] = {
            "prediction_mean": float(prediction.mean()),
            "observed_raw_mean": float(raw.mean()),
            "observed_clamped_mean": float(clamped.mean()),
            "signed_error": float((prediction - raw).mean()),
            "clamped_error": float((prediction - clamped).mean()),
            "clipping_gap": float((clamped - raw).mean()),
            "prediction_observed_spearman": correlation(prediction, raw),
            "negative_observed_fraction": float(np.mean(raw < 0)),
        }
    return result


def diagnose(directory: Path, start: int, stop: int | None, bootstrap: int, seed: int, audit: bool = False) -> dict:
    with (directory / "log.csv").open(newline="") as f:
        logs = {int(row["iter"]): row for row in csv.DictReader(f)}
    stop = max(logs) if stop is None else stop
    if start < 0 or stop < start or stop > max(logs) or bootstrap < 1:
        raise ValueError("require 0 <= start <= stop <= last logged iteration and bootstrap >= 1")
    report = {
        "run": directory.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "start": start,
        "stop": stop,
        "bootstrap": bootstrap,
        "seed": seed,
        "scope": "CPU, saved actor scores; complete fresh games; played rows only. No counterfactual tree baseline.",
        "uncertainty": "95% game-bootstrap intervals, conditional on saved data; games can share actors/openings.",
        "iterations": [],
    }
    games = None
    if audit:
        from .control_audit import TargetAudit

        target_audit = TargetAudit()
    for it in range(start, stop + 1):
        path = directory / f"window_{it:06d}.pt.zst"
        raw = path.read_bytes()
        data = torch.load(io.BytesIO(zstd.decompress(raw)), map_location="cpu", weights_only=True)
        if data.get("schema") != 2 or data.get("regret_version") != 2 or data["iteration"] != it:
            raise ValueError(f"{path}: requires current schema 2 / regret version 2")
        if games is None:
            games = Games(data["envs"], audit=audit)
        fresh, restarts, skipped = games.consume(data)
        if audit:
            target_audit.consume(games.completed, it)
        metrics = ranking_report(fresh, bootstrap, seed + it, games.fresh_segments)
        if audit:
            metrics["search_selection"] = games.search_selection
        metrics.update(iteration=it, partial_games_excluded=skipped, logged_lift=float(logs[it]["rank_lift"]))
        if fresh:
            metrics["log_lift_error"] = metrics["pooled_lift"] - metrics["logged_lift"]
            metrics["log_lift_matches"] = abs(metrics["log_lift_error"]) <= 1e-5
        prior = directory / f"ckpt_{it - 1:06d}.pt"
        if prior.is_file():
            feedback = checkpoint_report(prior, restarts, games.dropped)
            feedback["logged_lost_keys"] = int(logs[it]["buffer_lost"])
            feedback["lost_keys_match"] = feedback["lost_keys"] == feedback["logged_lost_keys"]
            if "first_tree" in feedback:
                feedback["logged_tree_error"] = float(logs[it]["tree_regret_error"])
                feedback["tree_error_matches"] = (
                    abs(feedback["first_tree"]["signed_error"] - feedback["logged_tree_error"]) <= 1e-5
                )
            metrics["prior_buffer"] = feedback
        report["iterations"].append(metrics)
        if fresh:
            local = metrics["within_game"]["1"]
            print(
                f"it {it}: pooled {metrics['pooled_lift']:+.5f}, ESS {metrics['pooled_ess']:.1f}, "
                f"within-game {local['lift']:+.5f} {local['lift_ci95']}, "
                f"log match {metrics['log_lift_matches']}",
                flush=True,
            )
        del raw, data, fresh, restarts
    if audit:
        report["target_audit"] = target_audit.report(bootstrap, seed)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="verify targets, tactical candidate selection and repeated-opening evidence",
    )
    parser.add_argument("--out", type=Path, help="default: runs/<run>/rgsc_diagnostics.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = run_dir(args.run)
    report = diagnose(directory, args.start, args.stop, args.bootstrap, args.seed, audit=args.audit)
    out = args.out or directory / "rgsc_diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=out.parent, prefix=".rgsc_", suffix=".tmp", encoding="utf-8", delete=False
        ) as f:
            name = f.name
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(name, out)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)
    print(out)


if __name__ == "__main__":
    main()
