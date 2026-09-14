"""Opening explorer queries over arena and imported games."""

from __future__ import annotations

import sqlite3
from typing import Any

from arena.core.game import (
    GameRecord,
    PositionState,
    initial_board_list,
    position_key,
    replay,
)
from arena.core.notation import parse_move
from arena.core.symmetry import symmetry_key, transform_board, transform_square
from arena.db.store import get_game


def resolve_line(line: str) -> PositionState:
    """Replay a compact MEAF line into the absolute query position."""
    record = GameRecord(setup=initial_board_list())
    for token in line.replace(",", " ").split():
        if token.endswith(".") or token in {"1-0", "0-1", "0-0", "1/2-1/2"}:
            continue
        parsed = parse_move(token, "meaf", "meaf")
        record.moves.append(parsed)
    return replay(record)[-1]


def explore(
    conn: sqlite3.Connection,
    line: str,
    *,
    symmetry: bool = True,
    source: str | None = None,
    player: str | None = None,
    min_games: int = 0,
) -> dict[str, Any]:
    """Summarise the moves played from one position across the library."""
    state = resolve_line(line)
    key = symmetry_key(state) if symmetry else position_key(state)
    column = "symmetry_key" if symmetry else "position_key"
    clauses = [f"p.{column} = ?", "p.move_from IS NOT NULL", "g.live = 0"]
    params: list[Any] = [key]
    if source:
        clauses.append("g.source = ?")
        params.append(source)
    if player:
        clauses.append("(g.blue LIKE ? OR g.red LIKE ?)")
        params.extend([f"%{player}%", f"%{player}%"])
    rows = conn.execute(
        "SELECT p.game_id, p.ply, p.move_from, p.move_to, g.result_winner"
        " FROM plies p JOIN games g ON g.id = p.game_id"
        f" WHERE {' AND '.join(clauses)}",
        params,
    ).fetchall()
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        record = get_game(conn, str(row["game_id"]))
        if record is None:
            continue
        move = (int(row["move_from"]), int(row["move_to"]))
        if symmetry:
            record.moves = record.moves[: int(row["ply"])]
            stored = replay(record)[-1]
            for diagonal in (False, True):
                if any(transform_board(stored.board, diagonal, cycle) == state.board for cycle in range(3)):
                    move = (transform_square(move[0], diagonal), transform_square(move[1], diagonal))
                    break
        item = grouped.setdefault(
            move,
            {
                "move": {"from": move[0], "to": move[1]},
                "games": 0,
                "wins_blue": 0,
                "draws": 0,
                "wins_red": 0,
                "rating_total": 0.0,
                "rating_count": 0,
            },
        )
        item["games"] += 1
        winner = row["result_winner"]
        item["wins_blue"] += int(winner == "blue")
        item["wins_red"] += int(winner == "red")
        item["draws"] += int(winner is None)
        ratings = [p.rating_before for p in record.players.values() if p.rating_before is not None]
        if ratings:
            item["rating_total"] += sum(ratings) / len(ratings)
            item["rating_count"] += 1
    moves = []
    for item in grouped.values():
        count = item.pop("rating_count")
        total = item.pop("rating_total")
        item["avg_rating"] = round(total / count, 1) if count else None
        if item["games"] >= min_games:
            moves.append(item)
    moves.sort(key=lambda item: (-item["games"], item["move"]["from"], item["move"]["to"]))
    return {
        "position": {
            "board": state.board,
            "to_move": state.to_move,
            "psc": state.psc,
            "ply": state.ply,
        },
        "moves": moves,
    }
