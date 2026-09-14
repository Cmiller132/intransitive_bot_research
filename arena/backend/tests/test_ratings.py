"""The Bradley-Terry fit and the ratings endpoints."""

from __future__ import annotations

import random

import numpy as np
from conftest import add_player

from arena.core.game import GameRecord, PlayerRef, Result
from arena.db import store
from arena.ratings import build_report, fit_ratings, history, save_history


def test_fit_recovers_a_synthetic_pool() -> None:
    truth = {"anchor": 0.0, "beta": 60.0, "gamma": 120.0, "delta": -80.0, "omega": 200.0}
    names = list(truth)
    size = len(names)
    counts = np.zeros((size, size))
    scores = np.zeros((size, size))
    rng = random.Random(20260909)
    for i in range(size):
        for j in range(i + 1, size):
            prob = 1.0 / (1.0 + 10.0 ** ((truth[names[j]] - truth[names[i]]) / 400.0))
            wins = sum(1 for _ in range(300) if rng.random() < prob)
            counts[i, j] = counts[j, i] = 300
            scores[i, j] = wins
            scores[j, i] = 300 - wins
    ratings, errors = fit_ratings(counts, scores, names.index("anchor"))
    for index, name in enumerate(names):
        assert abs(ratings[index] - truth[name]) <= 3.0 * errors[index] + 1e-9
        assert errors[index] < 25.0


def arena_game(index: int, blue: str, red: str, winner: str | None) -> GameRecord:
    """Build a finished arena game between two players."""
    return GameRecord(
        id=f"arena-{index}-0-0",
        source="arena",
        frame="meaf",
        players={"blue": PlayerRef(blue, kind="engine"), "red": PlayerRef(red, kind="engine")},
        result=Result(winner, "corner" if winner else "stagnation", ""),
        meta={"job": index, "pair": 0, "game": 0},
        sims=4,
    )


def test_report_and_history_over_stored_games(settings, conn) -> None:
    add_player(settings, conn, "alpha")
    add_player(settings, conn, "beta")
    for index in range(10):
        store.save_arena_game(conn, arena_game(index, "alpha", "beta", "blue" if index % 2 else "red"), live=False)
    report = build_report(conn, settings.target, settings.anchor)
    assert report["anchor"] == "alpha"
    assert {entry["name"] for entry in report["players"]} == {"alpha", "beta"}
    alpha = next(entry for entry in report["players"] if entry["name"] == "alpha")
    assert alpha["games"] == 10
    assert alpha["wins"] + alpha["draws"] + alpha["losses"] == 10
    assert alpha["opponents"]["beta"]["games"] == 10
    save_history(conn, report)
    series = history(conn)
    assert set(series) == {"alpha", "beta"}
    assert series["alpha"][0]["games"] == 10


def test_ratings_and_matchups_endpoints(client, conn) -> None:
    for index in range(4):
        store.save_arena_game(conn, arena_game(index, "alpha", "beta", "blue"), live=False)
    payload = client.get("/api/ratings").json()
    assert payload["anchor"] == "alpha"
    assert payload["target"] == 30.0
    beta = next(entry for entry in payload["players"] if entry["name"] == "beta")
    assert beta["games"] == 4 and beta["losses"] == 4
    assert beta["kind"] == "sq"
    matchups = client.get("/api/matchups").json()
    assert matchups["players"] == [entry["name"] for entry in payload["players"]]
    assert len(matchups["cells"]) == 1
    cell = matchups["cells"][0]
    assert cell["games"] == 4
    assert cell["wins_a"] + cell["draws"] + cell["wins_b"] == 4
    assert 0.0 <= cell["expected_a"] <= 1.0
    pair = client.get(f"/api/matchups/{cell['a']}/{cell['b']}").json()
    assert len(pair["games"]) == 4
    assert pair["avg_plies"] == 0.0
    assert pair["end_reasons"]["corner"] == 4


def test_parents_follow_the_series_numbering() -> None:
    from arena.ratings import parents_of

    parents = parents_of(["conv_g128_20", "conv_g128_0", "conv_g128_10", "sq_g128", "Vlad"])
    assert parents == {
        "conv_g128_0": None,
        "conv_g128_10": "conv_g128_0",
        "conv_g128_20": "conv_g128_10",
        "sq_g128": None,
        "Vlad": None,
    }


def test_uncertainty_is_the_players_own_not_the_anchor_link() -> None:
    """Two players 300 games apart from each other but 600 Elo below the anchor
    with a handful of anchor games: their errors reflect their own games."""
    names = ["anchor", "a", "b"]
    counts = np.zeros((3, 3))
    scores = np.zeros((3, 3))
    counts[1, 2] = counts[2, 1] = 300.0
    scores[1, 2], scores[2, 1] = 150.0, 150.0
    for i in (1, 2):
        counts[0, i] = counts[i, 0] = 10.0
        scores[0, i], scores[i, 0] = 9.7, 0.3
    ratings, errors = fit_ratings(counts, scores, names.index("anchor"), [None, None, 1])
    assert ratings[0] == 0.0 and ratings[1] < -400.0 and abs(ratings[1] - ratings[2]) < 5.0
    assert errors[1] < 25.0 and errors[2] < 25.0
    assert errors[0] > errors[1]  # the anchor is the weakly pinned one
