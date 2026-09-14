"""Fixtures that run the whole service against the fake bot."""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from arena.api.app import create_app  # noqa: E402
from arena.db import store  # noqa: E402
from arena.ratings import now_iso  # noqa: E402
from arena.settings import Settings  # noqa: E402

FAKE_BOT = [sys.executable, str(BACKEND / "arena" / "fake_bot.py")]
GOLDEN = Path(__file__).parent / "golden"
ADMIN_TOKEN = "test-token"


def make_settings(root: Path, **overrides) -> Settings:
    """Build settings rooted at a temporary directory, driving the fake bot."""
    values = dict(
        root=root,
        admin_token=ADMIN_TOKEN,
        public=False,
        jobs=1,
        target=30.0,
        sims=4,
        pairs=1,
        anchor="alpha",
        analysis_procs=2,
        bot=list(FAKE_BOT),
        scheduler=False,
    )
    values.update(overrides)
    return Settings(**values)


def add_player(settings: Settings, conn, name: str, spec: str = "sq:model.onnx") -> None:
    """Create a player directory and its database row."""
    directory = settings.players_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.onnx").write_bytes(b"onnx")
    (directory / "model.onnx.json").write_text("{}", encoding="utf-8")
    added = now_iso()
    (directory / "player.json").write_text(json.dumps({"name": name, "spec": spec, "added": added}), encoding="utf-8")
    store.upsert_player(conn, name, spec, added)


@contextmanager
def serve(app) -> Iterator[str]:
    """Run one application under uvicorn on a free port and yield its base URL."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True, name="test-uvicorn")
    thread.start()
    deadline = time.monotonic() + 30.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("the test server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=30.0)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A temporary runtime root with a players directory."""
    target = tmp_path / "arena"
    (target / "players").mkdir(parents=True)
    return target


@pytest.fixture
def settings(root: Path) -> Settings:
    """Settings for one isolated service instance."""
    return make_settings(root)


@pytest.fixture
def conn(settings: Settings):
    """A connection to the instance's database."""
    connection = store.connect(settings.database_path)
    yield connection
    connection.close()


@pytest.fixture
def client(settings: Settings, conn):
    """A test client with two players registered."""
    add_player(settings, conn, "alpha")
    add_player(settings, conn, "beta")
    app = create_app(settings, database_path=str(settings.database_path))
    with TestClient(app) as running:
        yield running


@pytest.fixture
def admin() -> dict[str, str]:
    """Headers that authenticate an administrative request."""
    return {"X-Admin-Token": ADMIN_TOKEN}
