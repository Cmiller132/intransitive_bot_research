"""Position evaluation and the cached-position endpoint."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from arena.api.deps import api_error, get_db, get_pool, is_admin, resolve_engine
from arena.db import store
from arena.engines import STANDARD_SIMS, EngineError, evaluate, resolve_sims

router = APIRouter(prefix="/api", tags=["positions"])
Db = Annotated[sqlite3.Connection, Depends(get_db)]
AdminToken = Annotated[str | None, Header(alias="X-Admin-Token")]


class EvalRequest(BaseModel):
    """One position evaluation request."""

    model_config = ConfigDict(extra="ignore")

    position: dict[str, Any]
    engine: str = "strongest"
    budget: str | dict[str, StrictInt] = "standard"
    multipv: int = Field(default=4, ge=1, le=128)
    heads: list[str] | None = None


@router.post("/eval")
def evaluate_position(body: EvalRequest, request: Request, db: Db, x_admin_token: AdminToken = None) -> dict[str, Any]:
    """Evaluate one position with a pooled analysis process."""
    try:
        _key, sims = resolve_sims(body.budget)
    except (TypeError, ValueError) as exc:
        raise api_error("invalid_eval", str(exc)) from exc
    if sims > STANDARD_SIMS and not is_admin(request, x_admin_token):
        raise api_error("admin_required", "deep analysis requires the admin token", status_code=401)
    name, spec = resolve_engine(request, db, body.engine)
    try:
        response = evaluate(get_pool(request), db, name, spec, body.position, body.budget, body.multipv)
    except LookupError as exc:
        raise api_error("position_not_found", str(exc), status_code=404) from exc
    except EngineError as exc:
        raise api_error("engine_unavailable", str(exc), status_code=503) from exc
    except (TypeError, ValueError) as exc:
        raise api_error("invalid_eval", str(exc)) from exc
    if body.heads is not None:
        wanted = set(body.heads)
        response = dict(response)
        response["heads"] = {key: value for key, value in response["heads"].items() if key in wanted}
    return response


@router.get("/positions/{position_key}")
def position_analyses(position_key: str, db: Db) -> dict[str, Any]:
    """Every cached evaluation of one position."""
    key = position_key.lower()
    if len(key) != 40 or any(char not in "0123456789abcdef" for char in key):
        raise api_error("invalid_position_key", "position key must be 40 hex characters")
    return store.position_analyses(db, key)
