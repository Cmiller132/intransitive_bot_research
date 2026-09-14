"""Stream parsing, statistics conversion and one real fake-bot job."""

from __future__ import annotations

import json
import time

from conftest import add_player

from arena.core.game import replay
from arena.db import store
from arena.events import Broker
from arena.scheduler import Scheduler, convert_stats

INFO = {
    "sims": 32,
    "value": 0.5,
    "plies_left": 41.5,
    "q": 0.25,
    "pi": 0.5,
    "exact_win": False,
    "top": [{"move": "e2-f1", "visits": 20, "q": 0.25}],
}


def test_stats_are_converted_to_blue_point_of_view() -> None:
    blue = convert_stats(INFO, "blue")
    assert blue["value"] == 0.5 and blue["q"] == 0.25
    assert blue["top"][0]["move"] == {"from": 13, "to": 5}
    assert blue["top"][0]["q"] == 0.25
    red = convert_stats(INFO, "red")
    assert red["value"] == -0.5 and red["q"] == -0.25
    assert red["top"][0]["q"] == -0.25
    assert red["pi"] == 0.5 and red["sims"] == 32
    assert convert_stats(None, "blue") is None


def test_stream_events_create_update_and_finalise_a_game(settings, conn) -> None:
    add_player(settings, conn, "alpha")
    add_player(settings, conn, "beta")
    broker = Broker()
    scheduler = Scheduler(settings, broker)
    scheduler.fit()
    subscription = broker.subscribe("live")
    job = scheduler.start_job("alpha", "beta")
    scheduler.stop.set()
    job.proc.kill()
    job.reader.join(timeout=10)
    job.records.clear()
    scheduler._apply(
        job, {"event": "start", "pair": 0, "game": 0, "first": "alpha", "second": "beta", "opening": ["e3-f3", "e7-d7"]}
    )
    game_id = f"arena-{job.id}-0-0"
    row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    assert row["live"] == 1 and row["ply_count"] == 2 and row["blue"] == "alpha"
    assert conn.execute("SELECT count(*) FROM plies WHERE game_id = ?", (game_id,)).fetchone()[0] == 0

    scheduler._apply(job, {"event": "move", "pair": 0, "game": 0, "ply": 2, "move": "f3-g3", "info": INFO})
    record = store.get_game(conn, game_id)
    assert len(record.moves) == 3
    assert record.moves[2].stats["value"] == 0.5
    assert record.live is True

    scheduler._apply(job, {"event": "move", "pair": 0, "game": 0, "ply": 3, "move": "d7-d6", "info": INFO})
    record = store.get_game(conn, game_id)
    assert record.moves[3].stats["value"] == -0.5

    scheduler._apply(job, {"event": "end", "pair": 0, "game": 0, "winner": 0, "end": "Goal", "plies": 4})
    row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    assert row["live"] == 0 and row["result_winner"] == "blue"
    assert row["result_reason"] == "corner"
    assert conn.execute("SELECT count(*) FROM plies WHERE game_id = ?", (game_id,)).fetchone()[0] == 5
    record = store.get_game(conn, game_id)
    assert record.meta["job"] == job.id and record.sims == settings.sims
    assert record.players["blue"].rating_before == 0.0
    assert len(replay(record)) == 5

    kinds = []
    while True:
        event = subscription.get(0.1)
        if event is None:
            break
        kinds.append(event["type"])
    assert kinds == ["job_start", "game_start", "move", "move", "game_end"]
    scheduler.shutdown()


def test_a_full_job_against_the_fake_bot(settings, conn) -> None:
    add_player(settings, conn, "alpha")
    add_player(settings, conn, "beta")
    scheduler = Scheduler(settings, Broker())
    scheduler.scan()
    scheduler.fit()
    scheduler.schedule()
    assert len(scheduler.running) == 1
    deadline = time.monotonic() + 180.0
    while scheduler.running and time.monotonic() < deadline:
        if not scheduler.poll():
            time.sleep(0.2)
    assert not scheduler.running
    job = conn.execute("SELECT * FROM jobs").fetchone()
    assert job["ok"] == 1 and {job["a"], job["b"]} == {"alpha", "beta"}
    assert job["seed"] == job["id"]
    rows = conn.execute("SELECT * FROM games ORDER BY id").fetchall()
    assert len(rows) == 2 * settings.pairs
    for row in rows:
        assert row["live"] == 0
        assert row["result_reason"] in {"corner", "no_pieces", "no_moves", "stagnation"}
        record = json.loads(row["record_json"])
        assert record["moves"][-1]["stats"]["sims"] == settings.sims
        assert {record["players"]["blue"]["name"], record["players"]["red"]["name"]} == {"alpha", "beta"}
    assert conn.execute("SELECT count(*) FROM plies").fetchone()[0] > 0
    report = scheduler.fit(publish=True)
    assert report["players"][0]["games"] == 2 * settings.pairs
    assert conn.execute("SELECT count(*) FROM ratings").fetchone()[0] == 2
    scheduler.shutdown()


def test_players_the_api_registers_are_scheduled(settings, conn) -> None:
    """A player row the API wrote (an upload, a replacement, a retirement)
    makes the next scan report a change, so the report is refitted without
    waiting for a job to end."""
    add_player(settings, conn, "alpha")
    scheduler = Scheduler(settings, Broker())
    assert scheduler.scan() and scheduler.fit()["players"]
    assert not scheduler.scan()
    add_player(settings, conn, "beta")  # the directory and the row, as the upload endpoint writes them
    assert scheduler.scan()
    scheduler.fit()
    assert {entry["name"] for entry in scheduler.eligible()} == {"alpha", "beta"} and scheduler.has_work()
    store.set_player_flag(conn, "beta", "retired", True)
    assert scheduler.scan()
    scheduler.fit()
    assert {entry["name"] for entry in scheduler.eligible()} == {"alpha"}
    assert not scheduler.scan()


def test_pairings_follow_the_katago_modes(settings, conn) -> None:
    for name in ("sq_g128", "conv_g128_0", "conv_g128_10", "conv_g128_20", "guest"):
        add_player(settings, conn, name)
    scheduler = Scheduler(settings, Broker())
    scheduler.scan()
    scheduler.fit()
    pool = scheduler.eligible()
    seen = set()
    for _ in range(200):
        pairing = scheduler.pair(pool)
        assert pairing is not None
        a, b = pairing
        assert a != b and {a, b} <= {entry["name"] for entry in pool}
        seen.add(pairing)
    assert any(b == "sq_g128" for _, b in seen)  # the anchor mode
    # a settled player or one over budget is never the candidate
    for entry in pool:
        entry["settled"] = entry["name"] != "guest"
    for _ in range(50):
        assert scheduler.pair(pool)[0] == "guest"
    for entry in pool:
        entry["settled"] = False
        entry["games"] = settings.max_games
    assert scheduler.pair(pool) is None and not scheduler.has_work()
