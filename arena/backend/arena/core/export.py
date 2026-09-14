"""One dispatch point for every public export format."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .game import GameRecord, PositionState, Result, Study, game_to_study, replay
from .meaf import encode_workshop, format_line
from .pgn import dumps as pgn_dumps
from .pgn import format_fen


def export(
    value: GameRecord | Study | PositionState | dict[str, Any], format: str, frame: str = "meaf"
) -> dict[str, str]:
    """Return API-ready content metadata without coupling codecs to FastAPI."""
    if format == "json":
        if isinstance(value, PositionState):
            payload = asdict(value)
        elif hasattr(value, "to_dict"):
            payload = value.to_dict()  # type: ignore[union-attr]
        else:
            payload = value
        return {
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
            "mime": "application/json",
            "filename": "intransitive.json",
        }
    if format == "henhen_pgn":
        if isinstance(value, Study):
            value = study_main_line(value)
        if not isinstance(value, GameRecord):
            raise ValueError("henhen_pgn export requires a game or study")
        return {
            "content": pgn_dumps(value),
            "mime": "application/x-chess-pgn",
            "filename": f"{value.id or value.source_id or 'game'}.pgn",
        }
    if format == "meaf_line":
        if isinstance(value, Study):
            value = study_main_line(value)
        if not isinstance(value, GameRecord):
            raise ValueError("meaf_line export requires a game or study")
        return {"content": format_line(value), "mime": "text/plain", "filename": f"{value.id or 'line'}.txt"}
    if format == "meaf_workshop_link":
        if isinstance(value, GameRecord):
            value = game_to_study(value)
        if not isinstance(value, Study):
            raise ValueError("meaf_workshop_link export requires a game or study")
        link, _ = encode_workshop(value)
        return {"content": link, "mime": "text/uri-list", "filename": f"{value.id or 'study'}.url"}
    if format == "henhen_fen":
        if isinstance(value, GameRecord):
            state = replay(value)[-1]
        elif isinstance(value, PositionState):
            state = value
        elif isinstance(value, dict):
            state = PositionState(
                list(value["board"]),
                value.get("to_move", "blue"),
                int(value.get("ply", 0)),
                int(value.get("psc", 0)),
                int(value.get("repetition", 1)),
                None,
            )
        else:
            raise ValueError("henhen_fen export requires a position or game")
        return {"content": format_fen(state.board, state.to_move), "mime": "text/plain", "filename": "position.fen"}
    raise ValueError(f"unknown export format {format!r}")


def study_main_line(study: Study) -> GameRecord:
    """Choose the first child at each branch, giving a study's main line as a game."""
    by_parent: dict[object, list[tuple[object, Any]]] = {}
    for node_id, node in study.nodes.items():
        by_parent.setdefault(node.parent, []).append((node_id, node))
    moves = []
    parent: object = None
    seen: set[object] = set()
    while by_parent.get(parent):
        node_id, node = min(by_parent[parent], key=lambda item: str(item[0]))
        if node_id in seen or node.move is None:
            break
        seen.add(node_id)
        moves.append(node.move)
        parent = node_id
    return GameRecord(
        source="local", setup=list(study.root_setup), moves=moves, result=Result(None, "unknown", ""), frame="meaf"
    )
