"""Opening explorer and player insight endpoints."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request

from arena.analysis.explorer import explore
from arena.analysis.insights import player_insights
from arena.api.deps import api_error, get_db, ratings_report

router = APIRouter(prefix="/api", tags=["analysis"])
Db = Annotated[sqlite3.Connection, Depends(get_db)]


@router.get("/explorer")
def opening_explorer(
    db: Db,
    line: str = "",
    symmetry: int = Query(1, ge=0, le=1),
    source: str | None = None,
    player: str | None = None,
    min_games: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Summarise the moves played from one line across the library."""
    try:
        return explore(db, line, symmetry=bool(symmetry), source=source, player=player, min_games=min_games)
    except ValueError as exc:
        raise api_error("invalid_line", str(exc)) from exc


@router.get("/insights/{player}")
def insights(player: str, request: Request, db: Db) -> dict[str, Any]:
    """Summarise one player's games, rating and opponents."""
    return player_insights(db, player, ratings_report(request, db))
