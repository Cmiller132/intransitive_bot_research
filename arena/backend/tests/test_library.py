"""Imports, the game page, the explorer, exports and the lab migration."""

from __future__ import annotations

import json

from conftest import GOLDEN

from arena.core.game import replay
from arena.core.pgn import loads as pgn_loads
from arena.db import store


def import_payload(client, source: str, payload: str) -> list[str]:
    """Import one payload and return the stored game ids."""
    response = client.post("/api/games/import", json={"source": source, "payload": payload})
    assert response.status_code == 200, response.text
    return response.json()["game_ids"]


def test_henhen_pgn_import(client, conn) -> None:
    text = (GOLDEN / "altfish_game2.pgn").read_text(encoding="utf-8")
    game_id = import_payload(client, "henhen_pgn", text)[0]
    assert len(game_id) == 12
    row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    assert row["source"] == "henhen" and row["frame"] == "henhen"
    assert row["ply_count"] == 209
    assert row["source_id"] == "757f2991f5723929c03a8401"
    assert conn.execute("SELECT count(*) FROM plies WHERE game_id = ?", (game_id,)).fetchone()[0] == 210
    assert import_payload(client, "henhen_pgn", text) == [game_id]
    page = client.get(f"/api/games/{game_id}").json()
    assert page["players"]["blue"]["name"] == "VladKlentBot"
    assert page["moves"][0]["san"].endswith(page["moves"][0]["san"][-2:])
    assert len(page["positions"]) == 210
    assert page["positions"][0]["position_key"] == page["positions"][0]["position_key"].lower()
    assert page["review_status"] == "not_analysed" and page["live"] is False
    listing = client.get("/api/games", params={"source": "henhen"}).json()
    assert listing["total"] == 1 and listing["items"][0]["id"] == game_id
    assert client.get("/api/games", params={"player": "ALTFish"}).json()["total"] == 1
    assert client.get("/api/games", params={"live": 1}).json()["total"] == 0


def test_meaf_line_import_and_explorer(client, conn, monkeypatch) -> None:
    from arena.analysis import explorer

    text = (GOLDEN / "meaf_line.txt").read_text(encoding="utf-8").strip()
    game_id = import_payload(client, "meaf_line", text)[0]
    record = store.get_game(conn, game_id)
    assert [move.to_dict()["from"] for move in record.moves] == [22, 43, 37, 58]
    assert record.source == "meaf"
    lengths = []

    def replay_prefix(record):
        lengths.append(len(record.moves))
        return replay(record)

    monkeypatch.setattr(explorer, "replay", replay_prefix)
    root = client.get("/api/explorer").json()
    assert lengths == [0, 0]
    assert root["position"]["to_move"] == "blue"
    assert sum(move["games"] for move in root["moves"]) == 1
    after = client.get("/api/explorer", params={"line": "1. E3-F3"}).json()
    assert after["position"]["ply"] == 1
    assert after["position"]["board"][22] == 0 and after["position"]["board"][23] == 3
    evaluated = client.post("/api/eval", json={"position": after["position"], "budget": "quick"})
    assert evaluated.status_code == 200, evaluated.text
    assert evaluated.json()["to_move"] == "red"
    assert sum(move["games"] for move in after["moves"]) == 1
    assert after["moves"][0]["wins_blue"] + after["moves"][0]["draws"] + after["moves"][0]["wins_red"] == 1
    filtered = client.get("/api/explorer", params={"min_games": 2}).json()
    assert filtered["moves"] == []
    assert client.get("/api/explorer", params={"source": "henhen"}).json()["moves"] == []
    assert client.get("/api/explorer", params={"line": "nonsense"}).status_code == 400


def test_fen_and_json_imports(client, conn) -> None:
    fen = "9/9/9/9/9/9/9/9/RPS6 b"
    game_id = import_payload(client, "fen", fen)[0]
    record = store.get_game(conn, game_id)
    assert record.setup is not None and record.setup[72:75] == [1, 2, 3]
    assert record.source == "local"
    payload = json.dumps(record.to_dict())
    assert import_payload(client, "json", payload) == [game_id]


def test_insights_and_export(client, conn) -> None:
    text = (GOLDEN / "meaf_line.txt").read_text(encoding="utf-8").strip()
    game_id = import_payload(client, "meaf_line", text)[0]
    insights = client.get("/api/insights/Blue").json()
    assert insights["games"] == 1 and insights["by_color"] == {"blue": 1}
    assert insights["opponents"][0]["name"] == "Red"
    assert insights["rating"] is None
    exported = client.post("/api/export", json={"game_id": game_id, "format": "meaf_line"}).json()
    assert exported["content"] == text
    assert exported["mime"] == "text/plain"
    pgn = client.post("/api/export", json={"game_id": game_id, "format": "henhen_pgn"}).json()
    assert pgn["content"].startswith("[")
    assert pgn_loads(pgn["content"]).moves[0].to == 23
    position = client.post(
        "/api/export", json={"position": {"game_id": game_id, "ply": 2}, "format": "henhen_fen"}
    ).json()
    assert position["content"].split()[1] == "b"
    assert client.post("/api/export", json={"format": "json"}).status_code == 400
    del conn


def test_manual_analysis_and_position_download_use_the_same_setup(client) -> None:
    from arena.core.game import initial_board_list

    position = {
        "board": initial_board_list(),
        "to_move": "blue",
        "psc": 0,
        "ply": 0,
        "history": [{"from": 38, "to": 48}],
    }
    evaluation = client.post("/api/eval", json={"position": position, "budget": "quick"})
    assert evaluation.status_code == 200, evaluation.text
    assert evaluation.json()["to_move"] == "red"
    response = client.post("/api/export", json={"position": position, "format": "json"})
    assert response.status_code == 200, response.text
    exported = json.loads(response.json()["content"])
    assert exported["board"][38] == 0 and exported["board"][48] == 3
    assert exported["ply"] == 1 and exported["psc"] == 1 and exported["to_move"] == "red"


def test_analysis_continuation_from_a_red_setup(client) -> None:
    board = [0] * 81
    board[40], board[20] = 4, 1
    position = {"board": board, "to_move": "red", "psc": 19, "history": [{"from": 40, "to": 41}]}
    response = client.post("/api/eval", json={"position": position, "budget": "quick"})
    assert response.status_code == 200, response.text
    assert response.json()["to_move"] == "blue"
    exported = client.post("/api/export", json={"position": position, "format": "json"})
    value = json.loads(exported.json()["content"])
    assert value["board"][40] == 0 and value["board"][41] == 4
    assert value["psc"] == 20


def test_meaf_history_and_workshop_imports(client, conn) -> None:
    raw = (GOLDEN / "meaf_game_history_sample.json").read_text(encoding="utf-8")
    game_id = import_payload(client, "json", raw)[0]
    record = store.get_game(conn, game_id)
    assert record.source == "meaf" and record.source_id == "fht134"
    assert len(record.moves) == 15
    assert record.players["blue"].name == "acro"
    assert record.result.reason == "timeout"
    assert len(replay(record)) == 16

    workshop_raw = (GOLDEN / "meaf_workshop_z7jvfz.json").read_text(encoding="utf-8")
    study_game = import_payload(client, "meaf_workshop", workshop_raw)[0]
    line = store.get_game(conn, study_game)
    assert line.source == "meaf" and line.moves
    assert line.setup is not None
    link = client.post("/api/export", json={"game_id": study_game, "format": "meaf_workshop_link"}).json()
    assert link["content"].startswith("https://meaf.us/rps2/?workshop=")
