"""Game import, catalogue, record, review and analysis endpoints."""

from __future__ import annotations

import json
import sqlite3
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Body, Depends, Header, Query, Request
from pydantic import BaseModel, ConfigDict, StrictInt

from arena.analysis.review import REVIEW_SIMS
from arena.api.deps import (
    api_error,
    get_db,
    get_reviewer,
    is_admin,
    request_settings,
    resolve_engine,
)
from arena.core import fetch
from arena.core.export import study_main_line
from arena.core.game import GameRecord, Result, position_key, replay
from arena.core.meaf import game_history, parse_line, workshop
from arena.core.notation import format_move
from arena.core.pgn import loads as pgn_loads
from arena.core.pgn import parse_fen, parse_pieces
from arena.core.symmetry import symmetry_key
from arena.db import store
from arena.engines import STANDARD_SIMS, resolve_sims

router = APIRouter(prefix="/api/games", tags=["games"])
Db = Annotated[sqlite3.Connection, Depends(get_db)]
AdminToken = Annotated[str | None, Header(alias="X-Admin-Token")]
REQUEST_BODY = Body(...)
FRAMES = frozenset({"meaf", "henhen"})
SOURCES = frozenset({"henhen", "meaf", "local", "arena"})
IMPORT_PAYLOAD_CHARS = 2_000_000
IMPORT_MAX_GAMES = 25
IMPORT_MAX_PLIES = 2_000


class ImportLimitError(ValueError):
    """An import exceeds a deliberately generous workload ceiling."""


class AnalyseRequest(BaseModel):
    """A request to review one game."""

    model_config = ConfigDict(extra="forbid")

    engine: str = "strongest"
    budget: str | dict[str, StrictInt] = "standard"


def _bounded(text: str) -> str:
    """Reject an oversized import payload."""
    if len(text) > IMPORT_PAYLOAD_CHARS:
        raise ImportLimitError(f"import payload may not exceed {IMPORT_PAYLOAD_CHARS} characters")
    return text


async def _records_for(source: str, payload: str, frame: str) -> list[GameRecord]:
    """Turn one import request into game records."""
    if source == "henhen_pgn":
        return [pgn_loads(payload)]
    if source == "henhen_id":
        return [pgn_loads(_bounded(await fetch.henhen_pgn(payload)))]
    if source == "henhen_series":
        game_ids = await fetch.henhen_series(payload)
        if len(game_ids) > IMPORT_MAX_GAMES:
            raise ImportLimitError(f"a series import may contain at most {IMPORT_MAX_GAMES} games")
        return [pgn_loads(_bounded(await fetch.henhen_pgn(game_id))) for game_id in game_ids]
    if source == "meaf_id":
        return [game_history(await fetch.meaf_game(payload))]
    if source == "meaf_line":
        moves, offset, result = parse_line(payload)
        if offset:
            raise ValueError(
                "a MEAF continuation omits the initial Blue half-move;"
                " import a complete line or provide a position setup"
            )
        record = GameRecord(
            source="meaf",
            frame="meaf",
            moves=moves,
            result=result or Result(None, "unknown", ""),
            meta={"leading_ply_offset": 0, "start_to_move": "blue"},
        )
        replay(record)
        return [record]
    if source == "meaf_workshop":
        raw: str | dict[str, Any] = payload
        if payload.lstrip().startswith("{"):
            raw = json.loads(payload)
        elif not payload.lstrip().startswith("http"):
            raw = await fetch.meaf_workshop(payload)
        study = workshop(raw)
        record = study_main_line(study)
        record.source = "meaf"
        record.tags = {"Event": study.title}
        return [record]
    if source == "fen":
        fields = payload.split()
        if len(fields) < 2:
            raise ValueError("FEN needs piece and side-to-move fields")
        side_field = fields[1].lower()
        if side_field not in {"b", "blue", "r", "red", "-"}:
            raise ValueError(f"unknown FEN side to move {fields[1]!r}")
        side = "red" if side_field in {"r", "red"} else "blue"
        if frame == "henhen":
            board, _side, territory = parse_fen(payload)
        else:
            board, territory = parse_pieces(fields[0]), (fields[2] if len(fields) > 2 else None)
        return [
            GameRecord(
                source="local",
                frame=frame,
                setup=board,
                result=Result(None, "unknown", ""),
                meta={"start_to_move": side, "territory": territory},
            )
        ]
    if source == "json":
        raw_json = json.loads(payload)
        if not isinstance(raw_json, dict):
            raise ValueError("JSON import root must be an object")
        if "gameID" in raw_json and "moveHistory" in raw_json:
            return [game_history(raw_json)]
        return [GameRecord.from_dict(raw_json)]
    raise ValueError(f"unknown import source {source!r}")


def _validate(record: GameRecord) -> None:
    """Reject a record whose dynamically-typed fields are unusable."""
    if record.source not in SOURCES:
        raise ValueError(f"unknown game source {record.source!r}")
    if record.frame not in FRAMES:
        raise ValueError(f"unknown game frame {record.frame!r}")
    if record.result.winner not in {None, "blue", "red"}:
        raise ValueError("result winner must be blue, red, or null")
    if set(record.players) != {"blue", "red"}:
        raise ValueError("players must contain exactly blue and red")
    for side, player in record.players.items():
        if player.kind not in {"human", "bot", "engine"}:
            raise ValueError(f"{side} player kind must be human, bot, or engine")
    if len(record.moves) > IMPORT_MAX_PLIES:
        raise ImportLimitError(f"a game import may contain at most {IMPORT_MAX_PLIES} plies")
    record.id = None
    record.live = False


@router.post("/import")
async def import_games(db: Db, body: dict[str, Any] = REQUEST_BODY) -> dict[str, Any]:
    """Import games from a site, a PGN, a MEAF line, a FEN or raw JSON."""
    source = body.get("source")
    payload = body.get("payload")
    frame = body.get("frame", "meaf")
    if not isinstance(source, str) or not isinstance(payload, str):
        raise api_error("invalid_import", "source and payload must be strings")
    if not isinstance(frame, str) or frame not in FRAMES:
        raise api_error("invalid_import", "frame must be meaf or henhen")
    try:
        _bounded(payload)
        records = await _records_for(source, payload, frame)
        if len(records) > IMPORT_MAX_GAMES:
            raise ImportLimitError(f"an import may contain at most {IMPORT_MAX_GAMES} games")
        for record in records:
            _validate(record)
        game_ids = [store.insert_game(db, record) for record in records]
    except ImportLimitError as exc:
        raise api_error("import_too_large", str(exc), status_code=413) from exc
    except httpx.TimeoutException as exc:
        raise api_error("upstream_timeout", "the source site timed out while importing", status_code=504) from exc
    except httpx.HTTPStatusError as exc:
        raise api_error(
            "upstream_error", f"the source site returned HTTP {exc.response.status_code}", status_code=502
        ) from exc
    except httpx.RequestError as exc:
        raise api_error("upstream_error", "the source site could not be reached", status_code=502) from exc
    except sqlite3.IntegrityError as exc:
        raise api_error("import_conflict", "the imported record already exists", status_code=409) from exc
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise api_error("invalid_import", str(exc)) from exc
    return {"game_ids": game_ids, "warnings": []}


@router.get("")
def games(
    db: Db,
    q: str | None = None,
    source: str | None = None,
    player: str | None = None,
    color: str | None = None,
    result: str | None = None,
    reason: str | None = None,
    engine: str | None = None,
    live: int | None = None,
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
    limit: int = Query(50, ge=0, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """List the games matching the library filters."""
    return store.list_games(
        db,
        q=q,
        source=source,
        player=player,
        color=color,
        result=result,
        reason=reason,
        engine=engine,
        live=None if live is None else bool(live),
        from_=from_,
        to=to,
        limit=limit,
        offset=offset,
    )


@router.get("/{game_id}")
def game(game_id: str, db: Db) -> dict[str, Any]:
    """One game record with its positions and review status."""
    record = store.get_game(db, game_id)
    if record is None:
        raise api_error("game_not_found", f"game {game_id} was not found", status_code=404)
    states = replay(record)
    payload = record.to_dict()
    dialect = "henhen" if record.frame == "henhen" else "meaf"
    for index, move in enumerate(record.moves):
        ending = (
            index == len(record.moves) - 1
            and record.result.winner is not None
            and record.result.reason in {"corner", "no_moves", "no_pieces"}
        )
        payload["moves"][index]["san"] = format_move(move, states[index].board, dialect, record.frame, ending)
    payload["positions"] = [
        {
            "ply": state.ply,
            "to_move": state.to_move,
            "psc": state.psc,
            "repetition": state.repetition,
            "position_key": position_key(state),
            "symmetry_key": symmetry_key(state),
            "result": state.result.to_dict() if state.result else None,
        }
        for state in states
    ]
    review = store.get_review(db, game_id)
    active = store.active_review_job(db, game_id) if review is None else None
    payload["review_status"] = "done" if review else (active["status"] if active else "not_analysed")
    payload["review_job"] = (
        {"id": active["id"], "status": active["status"], "progress": float(active["progress"])} if active else None
    )
    return payload


@router.delete("/{game_id}")
def delete_game(game_id: str, request: Request, db: Db, x_admin_token: AdminToken = None) -> dict[str, bool]:
    """Delete one game."""
    if not is_admin(request, x_admin_token):
        raise api_error("admin_required", "A valid X-Admin-Token is required", status_code=401)
    if not store.delete_game(db, game_id):
        raise api_error("game_not_found", f"game {game_id} was not found", status_code=404)
    return {"deleted": True}


@router.get("/{game_id}/review")
def game_review(game_id: str, db: Db, engine: str | None = None) -> dict[str, Any]:
    """The stored review of one game."""
    if store.get_game(db, game_id) is None:
        raise api_error("game_not_found", f"game {game_id} was not found", status_code=404)
    review = store.get_review(db, game_id, engine)
    if review is None:
        raise api_error("review_not_found", f"game {game_id} has not been analysed", status_code=404)
    return review


@router.post("/{game_id}/analyse", status_code=202)
def analyse_game(
    game_id: str, request: Request, db: Db, body: AnalyseRequest | None = None, x_admin_token: AdminToken = None
) -> dict[str, Any]:
    """Start, or attach to, a review of one game."""
    body = body or AnalyseRequest()
    record = store.get_game(db, game_id)
    if record is None:
        raise api_error("game_not_found", f"game {game_id} was not found", status_code=404)
    if not record.moves:
        raise api_error("invalid_job", "the game has no plies to analyse")
    budget = body.budget
    if isinstance(budget, str):
        if budget not in REVIEW_SIMS:
            raise api_error("invalid_job", "a review budget must be standard or deep")
        sims = REVIEW_SIMS[budget]
        key = budget
    else:
        key, sims = resolve_sims(budget)
        if sims < 1:
            raise api_error("invalid_job", "a review needs a searched budget")
    if sims > STANDARD_SIMS and not is_admin(request, x_admin_token):
        raise api_error("admin_required", "deep reviews require the admin token", status_code=401)
    name, spec = resolve_engine(request, db, body.engine)
    active = store.active_review_job(db, game_id, name)
    if active is not None:
        return {"job_id": active["id"], "reused": True}
    if store.get_review(db, game_id, name) is not None:
        raise api_error("review_exists", "this game already has a review from that engine", status_code=409)
    settings = request_settings(request)
    if store.count_active_review_jobs(db) >= max(1, settings.analysis_procs * 2):
        raise api_error("queue_full", "too many reviews are already queued", status_code=429)
    job_id = get_reviewer(request).submit(db, game_id, name, spec, key, sims)
    return {"job_id": job_id, "reused": False}
