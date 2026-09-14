"""Content negotiation endpoint for games and positions."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends

from arena.api.deps import api_error, get_db
from arena.core.export import export as export_value
from arena.core.game import PositionState, replay
from arena.db import store
from arena.engines import resolve_position

router = APIRouter(prefix="/api/export", tags=["export"])
Db = Annotated[sqlite3.Connection, Depends(get_db)]
REQUEST_BODY = Body(...)


@router.post("")
def export(db: Db, body: dict[str, Any] = REQUEST_BODY) -> dict[str, str]:
    """Export a game or a position in one of the public formats."""
    selectors = [name for name in ("game_id", "position") if name in body and body[name] is not None]
    if len(selectors) != 1:
        raise api_error("invalid_export", "select a game or a position")
    frame = body.get("frame", "meaf")
    if not isinstance(frame, str) or frame not in {"meaf", "henhen"}:
        raise api_error("invalid_export", "frame must be meaf or henhen")
    export_format = body.get("format", "json")
    if not isinstance(export_format, str):
        raise api_error("invalid_export", "format must be a string")
    try:
        if selectors[0] == "game_id":
            value: Any = store.get_game(db, str(body["game_id"]))
            if value is None:
                raise api_error("game_not_found", "game not found", status_code=404)
            if "ply" in body and body["ply"] is not None:
                states = replay(value)
                ply = int(body["ply"])
                if not 0 <= ply < len(states):
                    raise api_error("invalid_ply", "ply is out of range")
                value = states[ply]
        else:
            resolved = resolve_position(body["position"], db)
            value = PositionState(resolved.board, resolved.to_move, resolved.ply, resolved.psc, 1, resolved.terminal)
        return export_value(value, export_format, frame)
    except LookupError as exc:
        raise api_error("game_not_found", str(exc), status_code=404) from exc
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise api_error("invalid_export", str(exc)) from exc
