"""Player uploads and administrative player controls."""

from __future__ import annotations

import shutil
import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from arena.api.deps import admin_required, api_error, get_db, ratings_report, request_settings
from arena.db import store
from arena.players import NAME_RE, install_upload, split_spec
from arena.ratings import now_iso

router = APIRouter(prefix="/api/players", tags=["players"])
Db = Annotated[sqlite3.Connection, Depends(get_db)]
NAME_FIELD = Form(...)
SPEC_FIELD = Form(...)
REPLACE_FIELD = Form(0)
FILES_FIELD = File(...)


@router.get("")
def players(request: Request, db: Db) -> dict[str, Any]:
    """Every player in the pool, retired ones included."""
    return ratings_report(request, db)


@router.post("", dependencies=[Depends(admin_required)])
async def upload_player(
    request: Request,
    db: Db,
    name: str = NAME_FIELD,
    spec: str = SPEC_FIELD,
    replace: int = REPLACE_FIELD,
    files: list[UploadFile] = FILES_FIELD,
) -> dict[str, Any]:
    """Install an uploaded player into the pool."""
    if not NAME_RE.match(name):
        raise api_error("invalid_player", "bad player name")
    if split_spec(spec) is None:
        raise api_error("invalid_player", "spec must be sq:<path>, conv:<path> or rpsi:<command>")
    if not files:
        raise api_error("invalid_player", "no files uploaded")
    settings = request_settings(request)
    directory = settings.players_dir / name
    if directory.exists():
        if not replace:
            raise api_error("player_exists", f"player {name} already exists", status_code=409)
        shutil.rmtree(directory)
    payload = [(item.filename or "", await item.read()) for item in files]
    added = now_iso()
    error = install_upload(settings.players_dir, name, spec, payload, added)
    if error:
        raise api_error("invalid_player", f"upload rejected: {error}")
    if replace:
        db.execute("DELETE FROM games WHERE source = 'arena' AND (blue = ? OR red = ?)", (name, name))
        db.commit()
    store.upsert_player(db, name, spec, added)
    return {"name": name, "spec": spec, "added": added, "replaced": bool(replace)}


@router.post("/{name}/retire", dependencies=[Depends(admin_required)])
def retire(name: str, db: Db) -> dict[str, Any]:
    """Stop scheduling a player without deleting its games."""
    if not store.set_player_flag(db, name, "retired", True):
        raise api_error("player_not_found", f"unknown player {name}", status_code=404)
    return {"name": name, "retired": True}


@router.post("/{name}/unretire", dependencies=[Depends(admin_required)])
def unretire(name: str, db: Db) -> dict[str, Any]:
    """Return a retired player to the pool."""
    if not store.set_player_flag(db, name, "retired", False):
        raise api_error("player_not_found", f"unknown player {name}", status_code=404)
    return {"name": name, "retired": False}
