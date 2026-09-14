"""Service metadata and the health probe."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import Any

from fastapi import APIRouter, Request

from arena.api.deps import database_path, get_pool, request_settings
from arena.db import store

router = APIRouter(tags=["meta"])


@router.get("/api/meta")
def meta(request: Request) -> dict[str, Any]:
    """Describe the service and its display defaults."""
    settings = request_settings(request)
    return {
        "name": settings.name,
        "version": settings.version,
        "public": settings.public,
        "frames": ["meaf", "henhen"],
        "default_frame": "meaf",
        "piece_sets": ["glyph", "hands", "emblem", "letters"],
        "target": settings.target,
        "anchor": settings.anchor,
        "sims": settings.sims,
    }


@router.get("/healthz")
def health(request: Request) -> dict[str, Any]:
    """Report database, analysis pool and scheduler health."""
    players = 0
    db_ok = False
    try:
        with closing(store.connect(database_path(request))) as conn:
            players = int(conn.execute("SELECT count(*) FROM players").fetchone()[0])
        db_ok = True
    except (OSError, sqlite3.Error):
        db_ok = False
    scheduler = getattr(request.app.state, "scheduler", None)
    return {
        "ok": db_ok,
        "engine_loaded": get_pool(request).alive(),
        "db": db_ok,
        "jobs_running": len(scheduler.running) if scheduler else 0,
        "players": players,
    }
