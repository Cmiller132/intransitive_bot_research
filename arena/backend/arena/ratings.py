"""Bayesian Elo on the Elo scale, after the KataGo training server.

The fit is the server's `BayesianRatingService`: a Bradley-Terry model over
the games with draws counted as half a win each way, a prior of four
virtual draws between every checkpoint and its parent (the previous
checkpoint of the same series), a very weak prior of equal strength to
every other player for a player without a parent, minorisation-
maximisation sweeps that move each rating by the log of its actual over
its expected score, and the anchor reset to zero after every sweep. A
player's uncertainty is its own: the inverse square root of the
information its games carry against its opponents' fitted ratings,
`sum(games * p * (1 - p))`: how well the player is pinned to the pool. The
anchor's own row carries how well the pool is pinned to the anchor.
"""

from __future__ import annotations

import math
import re
import sqlite3
from datetime import UTC, datetime
from typing import Any

import numpy as np

ELO = 400.0 / math.log(10.0)  # Elo per unit of log-gamma
Z95 = 1.96
MIN_GAMES = 8
VIRTUAL_DRAWS = 4.0  # the parent prior, in games at an even score
ORPHAN_DRAWS = 0.01  # spread over every other player for a player without a parent
MAX_UNCERTAINTY = 10.0  # log-gamma units, as the server caps it
SERIES = re.compile(r"^(.*)_(\d+)$")


def now_iso() -> str:
    """The current UTC time as an ISO-8601 second-resolution string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parents_of(names: list[str]) -> dict[str, str | None]:
    """The previous checkpoint of the same series for every `<series>_<n>` name."""
    series: dict[str, list[tuple[int, str]]] = {}
    for name in names:
        match = SERIES.match(name)
        if match:
            series.setdefault(match.group(1), []).append((int(match.group(2)), name))
    parents: dict[str, str | None] = {name: None for name in names}
    for members in series.values():
        members.sort()
        for (_, name), (_, parent) in zip(members[1:], members[:-1], strict=True):
            parents[name] = parent
    return parents


def with_priors(
    counts: np.ndarray, scores: np.ndarray, parents: list[int | None], anchor: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Add the virtual draws: `VIRTUAL_DRAWS` between a player and its parent,
    `ORPHAN_DRAWS` spread over everyone else for a player without one."""
    counts = counts.copy()
    scores = scores.copy()
    size = counts.shape[0]
    for i, parent in enumerate(parents):
        if parent is not None and parent != i:
            counts[i, parent] += VIRTUAL_DRAWS
            counts[parent, i] += VIRTUAL_DRAWS
            scores[i, parent] += VIRTUAL_DRAWS / 2.0
            scores[parent, i] += VIRTUAL_DRAWS / 2.0
        elif i != anchor and size > 1:
            share = ORPHAN_DRAWS / (size - 1)
            for j in range(size):
                if j != i:
                    counts[i, j] += share
                    counts[j, i] += share
                    scores[i, j] += share / 2.0
                    scores[j, i] += share / 2.0
    return counts, scores


def win_probabilities(log_gamma: np.ndarray) -> np.ndarray:
    """p[i, j]: the model's probability that i beats j."""
    diff = np.clip(log_gamma[:, None] - log_gamma[None, :], -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-diff))


def fit_ratings(
    counts: np.ndarray,
    scores: np.ndarray,
    anchor: int | None,
    parents: list[int | None] | None = None,
    start: np.ndarray | None = None,
    sweeps: int = 1000,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ratings and their own standard errors, both on the Elo scale.

    `counts[i, j]` games and `scores[i, j]` points of i against j (draws
    half); `parents[i]` the index of i's parent or None; `start` warm-starts
    the sweeps (Elo). A player without games keeps rating 0 and an infinite
    error."""
    size = counts.shape[0]
    if size == 0:
        return np.zeros(0), np.zeros(0)
    parents = parents if parents is not None else [None] * size
    counts, scores = with_priors(counts, scores, parents, anchor)
    log_gamma = np.zeros(size) if start is None else np.asarray(start, dtype=float) / ELO
    played = counts.sum(axis=1) > 0.0
    actual = scores.sum(axis=1)
    for _ in range(sweeps):
        largest = 0.0
        for i in range(size):
            if not played[i] or actual[i] <= 0.0:
                continue
            expected = float((counts[i] * win_probabilities(log_gamma)[i]).sum())
            if expected <= 0.0:
                continue
            step = math.log(actual[i] / expected)
            log_gamma[i] += step
            largest = max(largest, abs(step))
        if anchor is not None:
            log_gamma -= log_gamma[anchor]
        if largest < tolerance:
            break
    prob = win_probabilities(log_gamma)
    precision = (counts * prob * (1.0 - prob)).sum(axis=1)
    errors = np.full(size, math.inf)
    positive = precision > 0.0
    errors[positive] = np.minimum(1.0 / np.sqrt(precision[positive]), MAX_UNCERTAINTY) * ELO
    return log_gamma * ELO, errors


def tally(
    names: list[str], games: list[tuple[str, str, str | None]]
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, dict[str, int]]]]:
    """Build the pairwise game-count and score matrices plus per-opponent records."""
    index = {name: i for i, name in enumerate(names)}
    size = len(names)
    counts = np.zeros((size, size))
    scores = np.zeros((size, size))
    records: dict[str, dict[str, dict[str, int]]] = {name: {} for name in names}
    for a, b, winner in games:
        if a not in index or b not in index or a == b:
            continue
        for one, other in ((a, b), (b, a)):
            slot = records[one].setdefault(other, {"games": 0, "wins": 0, "draws": 0, "losses": 0})
            slot["games"] += 1
            if winner is None:
                slot["draws"] += 1
            elif winner == one:
                slot["wins"] += 1
            else:
                slot["losses"] += 1
        i, j = index[a], index[b]
        counts[i, j] += 1
        counts[j, i] += 1
        share = 0.5 if winner is None else (1.0 if winner == a else 0.0)
        scores[i, j] += share
        scores[j, i] += 1.0 - share
    return counts, scores, records


def load_players(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """The players table as a dict keyed by name."""
    rows = conn.execute("SELECT name, spec, added, broken, retired FROM players").fetchall()
    return {row["name"]: dict(row) for row in rows}


def load_games(conn: sqlite3.Connection) -> list[tuple[str, str, str | None]]:
    """Every finished arena game as (blue, red, winner name) tuples."""
    rows = conn.execute("SELECT blue, red, result_winner FROM games WHERE source = 'arena' AND live = 0").fetchall()
    games = []
    for row in rows:
        winner = row["result_winner"]
        games.append(
            (row["blue"], row["red"], None if winner is None else (row["blue"] if winner == "blue" else row["red"]))
        )
    return games


def pick_anchor(players: dict[str, Any], preferred: str) -> str | None:
    """The player whose rating is fixed at zero."""
    if not players:
        return None
    if preferred in players:
        return preferred
    return min(players, key=lambda name: (players[name]["added"], name))


def latest_ratings(conn: sqlite3.Connection) -> dict[str, float]:
    """Each player's most recently saved rating, to warm-start the sweeps."""
    rows = conn.execute(
        "SELECT player, rating FROM ratings WHERE rowid IN (SELECT MAX(rowid) FROM ratings GROUP BY player)"
    ).fetchall()
    return {row["player"]: float(row["rating"]) for row in rows}


def build_report(conn: sqlite3.Connection, target: float, preferred_anchor: str) -> dict[str, Any]:
    """Compute the full ratings report from the database."""
    players = load_players(conn)
    games = load_games(conn)
    anchor = pick_anchor(players, preferred_anchor)
    names = sorted(players)
    parents = parents_of(names)
    counts, scores, records = tally(names, games)
    fitted = [i for i, name in enumerate(names) if records[name] or name == anchor]
    ratings: dict[str, float] = {}
    errors: dict[str, float] = {}
    if fitted:
        grid = np.ix_(fitted, fitted)
        seat = {names[i]: k for k, i in enumerate(fitted)}
        anchor_seat = seat.get(anchor) if anchor is not None else None
        parent_seats = [seat.get(parents[names[i]] or "") for i in fitted]
        previous = latest_ratings(conn)
        start = np.array([previous.get(names[i], 0.0) for i in fitted])
        values, sigmas = fit_ratings(counts[grid], scores[grid], anchor_seat, parent_seats, start)
        ratings = {names[i]: values[k] for k, i in enumerate(fitted)}
        errors = {names[i]: sigmas[k] for k, i in enumerate(fitted)}
    entries = []
    for name, row in players.items():
        record = records[name]
        total = sum(item["games"] for item in record.values())
        rating = float(ratings.get(name, 0.0))
        error = float(errors.get(name, math.inf))
        half = Z95 * error
        parent = parents[name]
        entries.append(
            {
                "name": name,
                "spec": row["spec"],
                "kind": str(row["spec"]).split(":", 1)[0],
                "rating": round(rating, 1),
                "se": None if math.isinf(error) else round(error, 2),
                "half_width": None if math.isinf(half) else round(half, 1),
                "games": total,
                "wins": sum(item["wins"] for item in record.values()),
                "draws": sum(item["draws"] for item in record.values()),
                "losses": sum(item["losses"] for item in record.values()),
                "settled": total >= MIN_GAMES and half <= target,
                "broken": bool(row["broken"]),
                "retired": bool(row["retired"]),
                "added": row["added"],
                "parent": parent,
                "vs_anchor": record.get(anchor) if anchor is not None and name != anchor else None,
                "vs_parent": record.get(parent) if parent is not None else None,
                "opponents": record,
            }
        )
    entries.sort(key=lambda entry: (-entry["rating"], entry["name"]))
    return {
        "fitted": now_iso(),
        "target": target,
        "anchor": anchor,
        "players": entries,
    }


def save_history(conn: sqlite3.Connection, report: dict[str, Any]) -> None:
    """Append one ratings history row per fitted player (one with games)."""
    conn.executemany(
        "INSERT INTO ratings (fitted, player, rating, se, games) VALUES (?, ?, ?, ?, ?)",
        [
            (report["fitted"], entry["name"], entry["rating"], entry["se"], entry["games"])
            for entry in report["players"]
            if entry["se"] is not None
        ],
    )
    conn.commit()


def history(conn: sqlite3.Connection, player: str | None = None, limit: int = 500) -> dict[str, list[dict[str, Any]]]:
    """Rating history per player, thinned to at most `limit` points each."""
    if player:
        rows = conn.execute(
            "SELECT fitted, player, rating, se, games FROM ratings WHERE player = ? ORDER BY fitted, rowid", (player,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT fitted, player, rating, se, games FROM ratings ORDER BY fitted, rowid").fetchall()
    series: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        series.setdefault(row["player"], []).append(
            {"fitted": row["fitted"], "rating": row["rating"], "se": row["se"], "games": row["games"]}
        )
    for name, points in series.items():
        if len(points) > limit:
            step = len(points) / limit
            sampled = [points[int(index * step)] for index in range(limit - 1)]
            sampled.append(points[-1])
            series[name] = sampled
    return series


def expected_score(rating_a: float, rating_b: float) -> float:
    """The model's expected score for A against B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
