"""Per-player performance aggregates for the library and the rankings page."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from typing import Any


def player_insights(conn: sqlite3.Connection, player: str, report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarise every stored game of one player."""
    rows = conn.execute(
        """SELECT id, blue, red, result_winner, result_reason, source,
                  started_at, imported_at, record_json
           FROM games WHERE live = 0 AND (lower(blue) = lower(?) OR lower(red) = lower(?))
           ORDER BY COALESCE(started_at, imported_at)""",
        (player, player),
    ).fetchall()
    results: Counter[str] = Counter()
    colors: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    opponents: dict[str, dict[str, int]] = {}
    timeline: list[dict[str, Any]] = []
    reviewed: list[tuple[float, float]] = []
    for row in rows:
        side = "blue" if str(row["blue"]).casefold() == player.casefold() else "red"
        opponent = row["red"] if side == "blue" else row["blue"]
        winner = row["result_winner"]
        outcome = "draw" if winner is None else ("win" if winner == side else "loss")
        results[outcome] += 1
        colors[side] += 1
        reasons[str(row["result_reason"])] += 1
        sources[str(row["source"])] += 1
        slot = opponents.setdefault(str(opponent), {"games": 0, "wins": 0, "draws": 0, "losses": 0})
        slot["games"] += 1
        slot[{"win": "wins", "draw": "draws", "loss": "losses"}[outcome]] += 1
        record = json.loads(row["record_json"])
        rating = record.get("players", {}).get(side, {}).get("rating_after")
        timeline.append({"game_id": row["id"], "date": row["started_at"], "outcome": outcome, "rating": rating})
        review_row = conn.execute("SELECT review_json FROM reviews WHERE game_id = ? LIMIT 1", (row["id"],)).fetchone()
        if review_row:
            review = json.loads(review_row["review_json"])
            reviewed.append((float(review["accuracy"][side]), float(review["agreement"][side])))
    games = len(rows)
    score = (results["win"] + 0.5 * results["draw"]) / games if games else 0.0
    entry = None
    for item in (report or {}).get("players", []):
        if item["name"] == player:
            entry = item
            break
    return {
        "player": player,
        "rating": entry["rating"] if entry else None,
        "half_width": entry["half_width"] if entry else None,
        "games": games,
        "wins": results["win"],
        "draws": results["draw"],
        "losses": results["loss"],
        "score": round(score, 4),
        "by_color": dict(colors),
        "by_reason": dict(reasons),
        "by_source": dict(sources),
        "opponents": [
            {"name": name, **counts, "score": round((counts["wins"] + 0.5 * counts["draws"]) / counts["games"], 4)}
            for name, counts in sorted(opponents.items(), key=lambda item: -item[1]["games"])
        ],
        "accuracy": round(sum(v[0] for v in reviewed) / len(reviewed), 2) if reviewed else None,
        "agreement": round(sum(v[1] for v in reviewed) / len(reviewed), 4) if reviewed else None,
        "timeline": timeline,
    }
