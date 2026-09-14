"""The arena pool seen as analysis engines."""

from __future__ import annotations

import re
import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from arena.api.deps import api_error, get_db, ratings_report, strongest_name
from arena.engines import head_manifest
from arena.players import split_spec

router = APIRouter(prefix="/api/engines", tags=["engines"])
Db = Annotated[sqlite3.Connection, Depends(get_db)]
RUN_RE = re.compile(r"^(?P<run>.+?)_(?P<iter>\d+)$")


def run_and_iteration(name: str) -> tuple[str, int | None]:
    """Split a player name into its run prefix and checkpoint number."""
    match = RUN_RE.match(name)
    if not match:
        return name, None
    return match.group("run"), int(match.group("iter"))


def describe(entry: dict[str, Any], strongest: str | None) -> dict[str, Any]:
    """Build the Engine shape from one ratings entry."""
    run, iteration = run_and_iteration(entry["name"])
    kind = split_spec(entry["spec"])
    return {
        "id": entry["name"],
        "label": entry["name"],
        "run": run,
        "iter": iteration,
        "params": 0,
        "heads": head_manifest(entry["spec"]),
        "strongest": entry["name"] == strongest,
        "notes": "",
        "spec": entry["spec"],
        "kind": kind[0] if kind else "sq",
        "rating": entry["rating"] if entry["games"] or entry["settled"] else None,
        "half_width": entry["half_width"],
        "games": entry["games"],
        "settled": entry["settled"],
        "broken": entry["broken"],
        "retired": entry["retired"],
        "added": entry["added"],
        "analysable": bool(kind and kind[0] != "rpsi"),
    }


@router.get("")
def engines(request: Request, db: Db) -> list[dict[str, Any]]:
    """List every player in the pool as an analysis engine."""
    report = ratings_report(request, db)
    strongest = strongest_name(report)
    return [describe(entry, strongest) for entry in report["players"]]


@router.get("/strongest")
def strongest(request: Request, db: Db) -> dict[str, str]:
    """Name the strongest settled player and the evidence for it."""
    report = ratings_report(request, db)
    name = strongest_name(report)
    if name is None:
        raise api_error("engine_not_found", "the arena has no players yet", status_code=404)
    entry = next(item for item in report["players"] if item["name"] == name)
    half = "inf" if entry["half_width"] is None else f"{entry['half_width']:.1f}"
    return {"id": name, "evidence": f"rating {entry['rating']:.1f} ± {half} after {entry['games']} games"}


@router.get("/{engine_id}/heads")
def engine_heads(engine_id: str, request: Request, db: Db) -> list[dict[str, Any]]:
    """The head manifest one engine's responses can fill."""
    report = ratings_report(request, db)
    entry = next((item for item in report["players"] if item["name"] == engine_id), None)
    if entry is None:
        raise api_error("engine_not_found", f"unknown engine {engine_id}", status_code=404)
    return head_manifest(entry["spec"])
