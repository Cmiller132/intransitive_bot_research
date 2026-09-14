"""Ratings, rating history and matchup endpoints."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from arena.api.deps import api_error, get_db, ratings_report
from arena.db import store
from arena.ratings import expected_score, history

router = APIRouter(prefix="/api", tags=["ratings"])
Db = Annotated[sqlite3.Connection, Depends(get_db)]


@router.get("/ratings")
def ratings(request: Request, db: Db) -> dict[str, Any]:
    """The current ratings table."""
    return ratings_report(request, db)


@router.get("/ratings/history")
def rating_history(db: Db, player: str | None = None) -> dict[str, Any]:
    """The stored rating history, sampled to at most 500 points per player."""
    return {"players": history(db, player)}


@router.get("/matchups")
def matchups(request: Request, db: Db) -> dict[str, Any]:
    """The pairwise results matrix over the pool."""
    report = ratings_report(request, db)
    entries = [entry for entry in report["players"] if not entry["retired"]]
    ratings_by_name = {entry["name"]: entry["rating"] for entry in entries}
    cells = []
    for entry in entries:
        for other, record in entry["opponents"].items():
            if other not in ratings_by_name or other <= entry["name"]:
                continue
            games = record["games"]
            score = (record["wins"] + 0.5 * record["draws"]) / games if games else 0.0
            cells.append(
                {
                    "a": entry["name"],
                    "b": other,
                    "games": games,
                    "wins_a": record["wins"],
                    "draws": record["draws"],
                    "wins_b": record["losses"],
                    "expected_a": round(expected_score(entry["rating"], ratings_by_name[other]), 4),
                    "score_a": round(score, 4),
                }
            )
    return {"players": [entry["name"] for entry in entries], "cells": cells}


@router.get("/matchups/{a}/{b}")
def matchup(a: str, b: str, db: Db) -> dict[str, Any]:
    """Every game between two players."""
    rows = db.execute(
        "SELECT * FROM games WHERE source = 'arena' AND live = 0"
        " AND ((blue = ? AND red = ?) OR (blue = ? AND red = ?))"
        " ORDER BY imported_at DESC",
        (a, b, b, a),
    ).fetchall()
    if not rows and (store.get_player(db, a) is None or store.get_player(db, b) is None):
        raise api_error("player_not_found", "one of the players is unknown", status_code=404)
    reviewed = {row["game_id"] for row in db.execute("SELECT game_id FROM reviews")}
    games = [store.summarise(row, row["id"] in reviewed) for row in rows]
    wins_a = draws = wins_b = 0
    plies = 0
    reasons: dict[str, int] = {}
    by_colour = {"a_blue": {"wins_a": 0, "draws": 0, "wins_b": 0}, "b_blue": {"wins_a": 0, "draws": 0, "wins_b": 0}}
    for row in rows:
        slot = by_colour["a_blue" if row["blue"] == a else "b_blue"]
        winner = row["result_winner"]
        name = None if winner is None else (row["blue"] if winner == "blue" else row["red"])
        if name is None:
            draws += 1
            slot["draws"] += 1
        elif name == a:
            wins_a += 1
            slot["wins_a"] += 1
        else:
            wins_b += 1
            slot["wins_b"] += 1
        plies += int(row["ply_count"])
        reasons[str(row["result_reason"])] = reasons.get(str(row["result_reason"]), 0) + 1
    total = len(rows)
    return {
        "games": games,
        "wins_a": wins_a,
        "draws": draws,
        "wins_b": wins_b,
        "by_colour": by_colour,
        "avg_plies": round(plies / total, 1) if total else 0.0,
        "end_reasons": reasons,
    }
