"""SQLite boundary shared by the API, the scheduler and the workers."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arena.core.game import (
    GameRecord,
    dumps_record,
    position_key,
    record_id,
    replay,
)
from arena.core.symmetry import symmetry_key

SCHEMA = Path(__file__).resolve().parent / "schema.sql"
_INIT_LOCK = threading.Lock()
_INITIALIZED: set[str] = set()


def now_iso() -> str:
    """The current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def connect(path: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open one connection to the arena database, creating the schema once."""
    target = os.fspath(path)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    if target == ":memory:" or target not in _INITIALIZED:
        with _INIT_LOCK:
            conn.executescript(SCHEMA.read_text(encoding="utf-8"))
            conn.commit()
            _INITIALIZED.add(target)
    return conn


def _write_plies(conn: sqlite3.Connection, record: GameRecord, states: list[Any]) -> None:
    """Rebuild the explorer index rows for one game."""
    conn.execute("DELETE FROM plies WHERE game_id = ?", (record.id,))
    conn.executemany(
        "INSERT INTO plies(game_id, ply, position_key, symmetry_key, move_from, move_to) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                record.id,
                index,
                position_key(state),
                symmetry_key(state),
                record.moves[index].from_ if index < len(record.moves) else None,
                record.moves[index].to if index < len(record.moves) else None,
            )
            for index, state in enumerate(states)
        ],
    )


def insert_game(conn: sqlite3.Connection, record: GameRecord) -> str:
    """Store an imported or migrated game, reusing the id of a duplicate."""
    if record.source_id:
        existing = conn.execute(
            "SELECT id FROM games WHERE source = ? AND source_id = ?", (record.source, record.source_id)
        ).fetchone()
        if existing:
            record.id = existing["id"]
            return str(existing["id"])
    states = replay(record)
    record.id = record.id or record_id(record)
    if conn.execute("SELECT 1 FROM games WHERE id = ?", (record.id,)).fetchone():
        return record.id
    conn.execute(
        """INSERT INTO games
           (id, source, source_id, source_url, frame, blue, red, blue_kind, red_kind,
            job, pair, game, result_winner, result_reason, result_reported, ply_count,
            started_at, ended_at, imported_at, record_json, live)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
        (
            record.id,
            record.source,
            record.source_id,
            record.source_url,
            record.frame,
            record.players["blue"].name,
            record.players["red"].name,
            record.players["blue"].kind,
            record.players["red"].kind,
            record.meta.get("job"),
            record.meta.get("pair"),
            record.meta.get("game"),
            record.result.winner,
            record.result.reason,
            record.result.reported,
            len(record.moves),
            record.started_at,
            record.ended_at,
            now_iso(),
            dumps_record(record),
        ),
    )
    _write_plies(conn, record, states)
    conn.commit()
    return record.id


def save_arena_game(conn: sqlite3.Connection, record: GameRecord, live: bool) -> str:
    """Insert or replace one arena game row, indexing it once it is finished."""
    states = None if live else replay(record)
    conn.execute(
        """INSERT INTO games
           (id, source, source_id, source_url, frame, blue, red, blue_kind, red_kind,
            job, pair, game, result_winner, result_reason, result_reported, ply_count,
            started_at, ended_at, imported_at, record_json, live)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             result_winner=excluded.result_winner, result_reason=excluded.result_reason,
             result_reported=excluded.result_reported, ply_count=excluded.ply_count,
             ended_at=excluded.ended_at, record_json=excluded.record_json,
             live=excluded.live""",
        (
            record.id,
            record.source,
            record.source_id,
            record.source_url,
            record.frame,
            record.players["blue"].name,
            record.players["red"].name,
            record.players["blue"].kind,
            record.players["red"].kind,
            record.meta.get("job"),
            record.meta.get("pair"),
            record.meta.get("game"),
            record.result.winner,
            record.result.reason,
            record.result.reported,
            len(record.moves),
            record.started_at,
            record.ended_at,
            now_iso(),
            dumps_record(record),
            int(live),
        ),
    )
    if states is None:
        conn.execute("DELETE FROM plies WHERE game_id = ?", (record.id,))
    else:
        _write_plies(conn, record, states)
    conn.commit()
    return str(record.id)


def get_game(conn: sqlite3.Connection, game_id: str) -> GameRecord | None:
    """Load one game record, or None when it is unknown."""
    row = conn.execute("SELECT record_json FROM games WHERE id = ?", (game_id,)).fetchone()
    return GameRecord.from_dict(json.loads(row["record_json"])) if row else None


def delete_game(conn: sqlite3.Connection, game_id: str) -> bool:
    """Delete one game and its index rows."""
    cursor = conn.execute("DELETE FROM games WHERE id = ?", (game_id,))
    conn.execute("DELETE FROM plies WHERE game_id = ?", (game_id,))
    conn.commit()
    return cursor.rowcount > 0


def summarise(row: sqlite3.Row, reviewed: bool = False) -> dict[str, Any]:
    """Build a GameSummary from a games row."""
    record = json.loads(row["record_json"])
    players = {}
    for side in ("blue", "red"):
        player = dict(record["players"][side])
        player.setdefault("kind", row[f"{side}_kind"])
        players[side] = player
    return {
        "id": row["id"],
        "source": row["source"],
        "source_id": row["source_id"],
        "source_url": row["source_url"],
        "frame": row["frame"],
        "players": players,
        "result": {
            "winner": row["result_winner"],
            "reason": row["result_reason"],
            "reported": row["result_reported"] or "",
        },
        "ply_count": row["ply_count"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "imported_at": row["imported_at"],
        "live": bool(row["live"]),
        "sims": record.get("sims"),
        "job": row["job"],
        "pair": row["pair"],
        "game": row["game"],
        "review_status": "done" if reviewed else "not_analysed",
    }


def list_games(
    conn: sqlite3.Connection,
    *,
    q: str | None = None,
    source: str | None = None,
    player: str | None = None,
    color: str | None = None,
    result: str | None = None,
    reason: str | None = None,
    engine: str | None = None,
    live: bool | None = None,
    from_: str | None = None,
    to: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List games matching the library filters, newest first."""
    clauses: list[str] = []
    params: list[Any] = []
    if q:
        clauses.append("(id LIKE ? OR source_id LIKE ? OR blue LIKE ? OR red LIKE ?)")
        params.extend([f"%{q}%"] * 4)
    if source:
        clauses.append("source = ?")
        params.append(source)
    if player:
        if color in ("blue", "red"):
            clauses.append(f"{color} LIKE ?")
            params.append(f"%{player}%")
        else:
            clauses.append("(blue LIKE ? OR red LIKE ?)")
            params.extend([f"%{player}%", f"%{player}%"])
    if result:
        clauses.append("result_winner IS ?")
        params.append(None if result in ("draw", "none") else result)
    if reason:
        clauses.append("result_reason = ?")
        params.append(reason)
    if engine:
        clauses.append("((blue_kind = 'engine' AND blue LIKE ?) OR (red_kind = 'engine' AND red LIKE ?))")
        params.extend([f"%{engine}%", f"%{engine}%"])
    if live is not None:
        clauses.append("live = ?")
        params.append(int(live))
    if from_:
        clauses.append("COALESCE(started_at, imported_at) >= ?")
        params.append(from_)
    if to:
        clauses.append("COALESCE(started_at, imported_at) <= ?")
        params.append(to)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    total = conn.execute("SELECT count(*) FROM games" + where, params).fetchone()[0]
    rows = conn.execute(
        "SELECT * FROM games" + where + " ORDER BY imported_at DESC, id LIMIT ? OFFSET ?",
        [*params, max(0, limit), max(0, offset)],
    ).fetchall()
    reviewed = {row["game_id"] for row in conn.execute("SELECT game_id FROM reviews")}
    return {"items": [summarise(row, row["id"] in reviewed) for row in rows], "total": total}


def cached_analysis(conn: sqlite3.Connection, key: str, engine: str, sims: int) -> dict[str, Any] | None:
    """Read one cached eval response."""
    row = conn.execute(
        "SELECT response_json FROM analyses WHERE position_key = ? AND engine = ? AND sims = ?", (key, engine, sims)
    ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["response_json"])
    except json.JSONDecodeError:
        return None


def save_analysis(
    conn: sqlite3.Connection, key: str, engine: str, sims: int, multipv: int, response: dict[str, Any]
) -> None:
    """Write one eval response into the cache."""
    conn.execute(
        """INSERT INTO analyses (position_key, engine, sims, multipv, created_at, response_json)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(position_key, engine, sims) DO UPDATE SET
             multipv=excluded.multipv, created_at=excluded.created_at,
             response_json=excluded.response_json""",
        (key, engine, sims, multipv, now_iso(), json.dumps(response, separators=(",", ":"), ensure_ascii=False)),
    )
    conn.commit()


def position_analyses(conn: sqlite3.Connection, key: str) -> dict[str, Any]:
    """Every cached evaluation of one position."""
    rows = conn.execute(
        "SELECT engine, sims, multipv, created_at, response_json FROM analyses"
        " WHERE position_key = ? ORDER BY created_at DESC",
        (key,),
    ).fetchall()
    evaluations = []
    for row in rows:
        try:
            response = json.loads(row["response_json"])
        except json.JSONDecodeError:
            continue
        evaluations.append(
            {
                "engine": row["engine"],
                "sims": row["sims"],
                "multipv": row["multipv"],
                "created_at": row["created_at"],
                "value": response.get("value"),
                "search": response.get("search"),
            }
        )
    return {"evals": evaluations}


def save_review(conn: sqlite3.Connection, game_id: str, review: dict[str, Any]) -> None:
    """Store a finished review."""
    conn.execute(
        """INSERT INTO reviews (game_id, engine, sims, created_at, review_json)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(game_id, engine) DO UPDATE SET
             sims=excluded.sims, created_at=excluded.created_at,
             review_json=excluded.review_json""",
        (
            game_id,
            review["engine"],
            int(review["sims"]),
            now_iso(),
            json.dumps(review, separators=(",", ":"), ensure_ascii=False),
        ),
    )
    conn.commit()


def get_review(conn: sqlite3.Connection, game_id: str, engine: str | None = None) -> dict[str, Any] | None:
    """Read a stored review, preferring one engine when asked."""
    if engine:
        row = conn.execute(
            "SELECT review_json FROM reviews WHERE game_id = ? AND engine = ?", (game_id, engine)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT review_json FROM reviews WHERE game_id = ? ORDER BY created_at DESC LIMIT 1", (game_id,)
        ).fetchone()
    return json.loads(row["review_json"]) if row else None


def create_review_job(conn: sqlite3.Connection, job_id: str, game_id: str, engine: str, sims: int) -> dict[str, Any]:
    """Insert a queued review job."""
    stamp = now_iso()
    conn.execute(
        "INSERT INTO review_jobs (id, game_id, engine, sims, status, progress, detail,"
        " created_at, updated_at) VALUES (?, ?, ?, ?, 'queued', 0.0, '', ?, ?)",
        (job_id, game_id, engine, sims, stamp, stamp),
    )
    conn.commit()
    return get_review_job(conn, job_id) or {}


def update_review_job(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    detail: str | None = None,
) -> None:
    """Update a review job's status, progress or detail."""
    fields: list[str] = []
    params: list[Any] = []
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if progress is not None:
        fields.append("progress = ?")
        params.append(max(0.0, min(1.0, float(progress))))
    if detail is not None:
        fields.append("detail = ?")
        params.append(detail)
    fields.append("updated_at = ?")
    params.extend([now_iso(), job_id])
    conn.execute(f"UPDATE review_jobs SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()


def get_review_job(conn: sqlite3.Connection, job_id: str) -> dict[str, Any] | None:
    """Read one review job."""
    row = conn.execute("SELECT * FROM review_jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def active_review_job(conn: sqlite3.Connection, game_id: str, engine: str | None = None) -> dict[str, Any] | None:
    """The queued or running review job for a game, if any."""
    if engine:
        row = conn.execute(
            "SELECT * FROM review_jobs WHERE game_id = ? AND engine = ?"
            " AND status IN ('queued', 'running') ORDER BY created_at DESC LIMIT 1",
            (game_id, engine),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM review_jobs WHERE game_id = ? AND status IN ('queued', 'running')"
            " ORDER BY created_at DESC LIMIT 1",
            (game_id,),
        ).fetchone()
    return dict(row) if row else None


def count_active_review_jobs(conn: sqlite3.Connection) -> int:
    """How many reviews are queued or running."""
    return int(conn.execute("SELECT count(*) FROM review_jobs WHERE status IN ('queued', 'running')").fetchone()[0])


def upsert_player(conn: sqlite3.Connection, name: str, spec: str, added: str) -> None:
    """Add or refresh one player, clearing its broken flag."""
    conn.execute(
        """INSERT INTO players (name, spec, added, broken, retired) VALUES (?, ?, ?, 0, 0)
           ON CONFLICT(name) DO UPDATE SET spec=excluded.spec, added=excluded.added, broken=0""",
        (name, spec, added),
    )
    conn.commit()


def set_player_flag(conn: sqlite3.Connection, name: str, field: str, value: bool) -> bool:
    """Set a player's broken or retired flag."""
    if field not in ("broken", "retired"):
        raise ValueError(f"unknown player flag {field!r}")
    cursor = conn.execute(f"UPDATE players SET {field} = ? WHERE name = ?", (int(value), name))
    conn.commit()
    return cursor.rowcount > 0


def get_player(conn: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    """Read one player row."""
    row = conn.execute("SELECT * FROM players WHERE name = ?", (name,)).fetchone()
    return dict(row) if row else None
