"""The HTTP surface: metadata, engines, evaluation, guards and events."""

from __future__ import annotations

import json
import time

import httpx
from conftest import ADMIN_TOKEN, GOLDEN, add_player, make_settings, serve
from fastapi.testclient import TestClient
from test_ratings import arena_game

from arena.api.app import create_app
from arena.core.game import initial_board_list
from arena.db import store
from arena.engines import HEAD_SPECS


def test_meta_and_health(client) -> None:
    meta = client.get("/api/meta").json()
    assert meta["name"] == "Intransitive arena"
    assert meta["frames"] == ["meaf", "henhen"]
    assert meta["anchor"] == "alpha" and meta["sims"] == 4
    health = client.get("/healthz").json()
    assert health["ok"] and health["db"] and health["players"] == 2
    assert health["jobs_running"] == 0
    assert client.get("/api/scheduler").json()["slots"] == 1


def test_engines_list_and_heads(client) -> None:
    engines = client.get("/api/engines").json()
    assert {engine["id"] for engine in engines} == {"alpha", "beta"}
    engine = engines[0]
    assert engine["kind"] == "sq" and engine["analysable"] is True
    assert engine["params"] == 0 and engine["run"] == engine["id"]
    assert [head["id"] for head in engine["heads"]] == [head["id"] for head in HEAD_SPECS]
    assert client.get("/api/engines/alpha/heads").json()[0]["id"] == "policy"
    assert client.get("/api/engines/strongest").json()["id"] in {"alpha", "beta"}
    assert client.get("/api/engines/nobody/heads").status_code == 404


def test_eval_through_the_fake_bot_builds_every_head(client, conn) -> None:
    body = {
        "position": {"board": initial_board_list(), "to_move": "blue"},
        "engine": "alpha",
        "budget": "standard",
        "multipv": 3,
    }
    response = client.post("/api/eval", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["engine"] == "alpha" and payload["to_move"] == "blue"
    assert payload["terminal"] is None
    assert len(payload["legal"]) == 36
    assert payload["value"] == {"blue": 0.0, "mover": 0.0, "source": "search"}
    assert payload["search"]["sims"] == 128
    assert len(payload["search"]["lines"]) == 3
    line = payload["search"]["lines"][0]
    assert set(line) == {"move", "pv", "q", "pi", "visits", "rank"}
    assert line["rank"] == 1 and line["pv"][0] == line["move"]
    heads = payload["heads"]
    assert set(heads) == {head["id"] for head in HEAD_SPECS}
    for name in ("policy", "search", "q", "draw"):
        assert heads[name]["kind"] == "move_scalar"
        assert len(heads[name]["moves"]) == 36
        assert set(heads[name]["moves"][0]) == {"from", "to", "value"}
        assert len(heads[name]["range"]) == 2
    assert heads["policy"]["render"] == "arrows" and heads["q"]["render"] == "tint"
    assert heads["value"]["kind"] == "scalar"
    assert heads["value"]["items"][0]["label"] == "Policy-weighted Q"
    assert heads["plies_to_end"]["kind"] == "histogram"
    assert len(heads["plies_to_end"]["bins"]) == 4
    assert heads["plies_to_end"]["expectation"] == 40.0

    key = payload["position_key"]
    assert store.cached_analysis(conn, key, "alpha", 128) is not None
    cached = client.get(f"/api/positions/{key}").json()["evals"]
    assert cached[0]["engine"] == "alpha" and cached[0]["sims"] == 128
    assert client.get("/api/positions/nothex").status_code == 400


def test_eval_rejects_deep_budgets_without_the_token(client, admin) -> None:
    body = {"position": {"board": initial_board_list(), "to_move": "blue"}, "budget": "deep"}
    assert client.post("/api/eval", json=body).status_code == 401
    assert client.post("/api/eval", json=body, headers=admin).status_code == 200


def test_terminal_positions_short_circuit(client) -> None:
    board = [0] * 81
    board[80] = 1
    board[0] = 4
    response = client.post("/api/eval", json={"position": {"board": board, "to_move": "red"}})
    payload = response.json()
    assert payload["terminal"] == {"winner": "blue", "reason": "corner"}
    assert payload["legal"] == [] and payload["heads"] == {}
    assert payload["value"] == {"blue": 1.0, "mover": -1.0, "source": "network"}


def test_admin_and_lan_guards(settings, conn) -> None:
    add_player(settings, conn, "alpha")
    store.save_arena_game(conn, arena_game(1, "alpha", "alpha", None), live=False)
    app = create_app(settings, database_path=str(settings.database_path))
    with TestClient(app) as anonymous:
        assert anonymous.delete("/api/games/arena-1-0-0").status_code == 401
        assert anonymous.post("/api/players/alpha/retire").status_code == 401
    with TestClient(app, client=("192.168.1.7", 5)) as lan:
        assert lan.post("/api/players/alpha/retire").json() == {"name": "alpha", "retired": True}
        assert lan.post("/api/players/alpha/unretire").status_code == 200
    with TestClient(app) as authorised:
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        assert authorised.post("/api/players/alpha/retire", headers=headers).status_code == 200
        assert authorised.delete("/api/games/arena-1-0-0", headers=headers).json() == {"deleted": True}
        assert authorised.delete("/api/games/arena-1-0-0", headers=headers).status_code == 404


def test_public_mode_requires_the_token_for_writes(root, conn) -> None:
    settings = make_settings(root, public=True)
    add_player(settings, conn, "alpha")
    app = create_app(settings, database_path=str(settings.database_path))
    with TestClient(app) as anonymous:
        assert anonymous.get("/api/ratings").status_code == 200
        assert anonymous.post("/api/players/alpha/retire").status_code == 401
        limited = anonymous.post("/api/games/import", json={"source": "meaf_line", "payload": "1. E3-F3"})
        assert limited.status_code == 200


def test_import_rate_limit(client) -> None:
    body = {"source": "nonsense", "payload": "x"}
    codes = [client.post("/api/games/import", json=body).status_code for _ in range(13)]
    assert codes[:12] == [400] * 12
    assert codes[12] == 429
    assert client.post("/api/games/import", json=body).json()["error"]["code"] == "import_rate_limited"


def test_analysis_has_no_per_client_rate_limit(client) -> None:
    body = {"position": {"board": initial_board_list(), "to_move": "blue"}, "budget": "quick"}
    for _ in range(35):
        response = client.post("/api/eval", json=body)
        assert response.status_code == 200, response.text
    for _ in range(6):
        response = client.post("/api/games/missing/analyse", json={"engine": "alpha"})
        assert response.status_code == 404, response.text


def test_upload_installs_a_player(client, settings, admin, tmp_path) -> None:
    model = tmp_path / "net.onnx"
    model.write_bytes(b"onnx")
    meta = tmp_path / "net.onnx.json"
    meta.write_text("{}", encoding="utf-8")
    files = [("files", ("net.onnx", model.read_bytes())), ("files", ("net.onnx.json", meta.read_bytes()))]
    response = client.post("/api/players", data={"name": "gamma", "spec": "conv:net.onnx"}, files=files, headers=admin)
    assert response.status_code == 200, response.text
    assert (settings.players_dir / "gamma" / "player.json").is_file()
    names = {entry["name"] for entry in client.get("/api/players").json()["players"]}
    assert names == {"alpha", "beta", "gamma"}
    rejected = client.post(
        "/api/players", data={"name": "bad name", "spec": "conv:net.onnx"}, files=files, headers=admin
    )
    assert rejected.status_code == 400
    assert (
        client.post(
            "/api/players", data={"name": "delta", "spec": "conv:missing.onnx"}, files=files, headers=admin
        ).status_code
        == 400
    )


def test_live_events_stream_a_move(settings, conn) -> None:
    add_player(settings, conn, "alpha")
    app = create_app(settings, database_path=str(settings.database_path))
    move = {"type": "move", "game_id": "arena-1-0-0", "ply": 3, "move": {"from": 13, "to": 5}, "stats": {"sims": 4}}
    with serve(app) as base:
        with httpx.stream("GET", f"{base}/api/live/events", timeout=20.0) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            lines = response.iter_lines()
            assert next(lines) == ": open"
            app.state.broker.publish("live", move)
            payload = next(line for line in lines if line.startswith("data: "))
        assert '"type":"move"' in payload and '"ply":3' in payload
        assert httpx.get(f"{base}/api/live").json() == {"jobs": []}


def test_review_runs_through_the_pool(client, admin) -> None:
    text = (GOLDEN / "meaf_line.txt").read_text(encoding="utf-8").strip()
    game_id = client.post("/api/games/import", json={"source": "meaf_line", "payload": text}).json()["game_ids"][0]
    started = client.post(f"/api/games/{game_id}/analyse", json={"engine": "alpha", "budget": "standard"})
    assert started.status_code == 202
    job_id = started.json()["job_id"]
    assert started.json()["reused"] is False
    deadline = time.monotonic() + 60.0
    status = "queued"
    while time.monotonic() < deadline and status not in {"done", "failed"}:
        status = client.get(f"/api/jobs/{job_id}").json()["status"]
        time.sleep(0.1)
    assert status == "done"
    review = client.get(f"/api/games/{game_id}/review").json()
    assert review["engine"] == "alpha" and review["sims"] == 128
    assert len(review["plies"]) == 4
    first = review["plies"][0]
    assert set(first) >= {"ply", "move", "loss", "label", "best", "lines", "played_evidence"}
    assert first["played_evidence"]["source"] == "search"
    assert review["accuracy"]["blue"] >= 0.0
    assert set(review["agreement"]) == {"blue", "red"}
    page = client.get(f"/api/games/{game_id}").json()
    assert page["review_status"] == "done" and page["review_job"] is None
    assert client.get("/api/jobs/missing").status_code == 404
    del admin


def test_review_stream_returns_completion_when_connecting_after_the_job_ended(client, conn) -> None:
    store.create_review_job(conn, "finished", "game", "alpha", 0)
    store.update_review_job(conn, "finished", status="done", progress=1.0)
    response = client.get("/api/jobs/finished/events")
    assert response.status_code == 200
    payload = next(line[6:] for line in response.text.splitlines() if line.startswith("data: "))
    assert json.loads(payload)["status"] == "done"
    assert client.get("/api/jobs/missing/events").status_code == 404


def test_review_stream_subscribes_before_reading_its_snapshot(client, conn, monkeypatch) -> None:
    store.create_review_job(conn, "race", "game", "alpha", 0)
    original = store.get_review_job

    def finish_during_snapshot(db, job_id):
        snapshot = original(db, job_id)
        client.app.state.broker.publish(f"job:{job_id}", {"id": job_id, "status": "done", "progress": 1})
        return snapshot

    monkeypatch.setattr(store, "get_review_job", finish_during_snapshot)
    response = client.get("/api/jobs/race/events")
    statuses = [json.loads(line[6:])["status"] for line in response.text.splitlines() if line.startswith("data: ")]
    assert statuses == ["queued", "done"]


def test_the_pool_restarts_a_dead_process(client) -> None:
    body = {"position": {"board": initial_board_list(), "to_move": "blue"}, "engine": "beta", "budget": "quick"}
    assert client.post("/api/eval", json=body).status_code == 200
    pool = client.app.state.pool
    process = pool.processes["beta"]
    assert process.alive()
    process.proc.kill()
    process.proc.wait()
    assert client.post("/api/eval", json={**body, "budget": {"sims": 8}}).status_code == 200
    assert pool.processes["beta"].alive()
    assert pool.alive()
