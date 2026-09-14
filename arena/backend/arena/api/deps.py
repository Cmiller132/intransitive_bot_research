"""FastAPI dependencies shared by every router."""

from __future__ import annotations

import hmac
import ipaddress
import sqlite3
from collections.abc import Iterator
from typing import Any

from fastapi import Header, HTTPException, Request, status

from arena.analysis.review import ReviewWorker
from arena.db import store
from arena.engines import AnalysisPool
from arena.events import Broker
from arena.ratings import build_report
from arena.settings import Settings, get_settings

LAN_NETWORKS = (ipaddress.ip_network("192.168.0.0/16"), ipaddress.ip_network("127.0.0.0/8"))


def request_settings(request: Request) -> Settings:
    """The settings this application instance was built with."""
    configured = getattr(request.app.state, "settings", None)
    return configured if configured is not None else get_settings()


def database_path(request: Request) -> str:
    """The sqlite path this application instance uses."""
    override = getattr(request.app.state, "database_path", None)
    return str(override) if override else str(request_settings(request).database_path)


def get_db(request: Request) -> Iterator[sqlite3.Connection]:
    """Give each request its own connection and transaction boundary."""
    conn = store.connect(database_path(request))
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def client_host(request: Request) -> str:
    """The requesting client's address, or `unknown`."""
    return request.client.host if request.client else "unknown"


def is_lan(request: Request) -> bool:
    """Whether the client is on the local network."""
    try:
        address = ipaddress.ip_address(client_host(request))
    except ValueError:
        return False
    return any(address in network for network in LAN_NETWORKS)


def has_admin_token(request: Request, supplied: str | None) -> bool:
    """Whether a supplied token matches the configured admin token."""
    expected = request_settings(request).admin_token
    if not expected or supplied is None:
        return False
    return hmac.compare_digest(supplied, expected)


def is_admin(request: Request, supplied: str | None) -> bool:
    """Whether a request may perform administrative actions."""
    return has_admin_token(request, supplied) or is_lan(request)


def admin_required(request: Request, x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")) -> None:
    """Reject a request that is neither token-authenticated nor from the LAN."""
    if is_lan(request):
        return
    if not request_settings(request).admin_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "admin_not_configured", "message": "ARENA_ADMIN_TOKEN is not set"},
        )
    if not has_admin_token(request, x_admin_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "admin_required", "message": "A valid X-Admin-Token is required"},
        )


def api_error(code: str, message: str, detail: Any = None, status_code: int = 400) -> HTTPException:
    """Build an HTTPException carrying the public error envelope."""
    payload: dict[str, Any] = {"code": code, "message": message}
    if detail is not None:
        payload["detail"] = detail
    return HTTPException(status_code=status_code, detail=payload)


def get_broker(request: Request) -> Broker:
    """The application's event broker."""
    broker = getattr(request.app.state, "broker", None)
    if broker is None:
        broker = Broker()
        request.app.state.broker = broker
    return broker


def get_pool(request: Request) -> AnalysisPool:
    """The application's analysis process pool."""
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        pool = AnalysisPool(request_settings(request))
        request.app.state.pool = pool
    return pool


def get_reviewer(request: Request) -> ReviewWorker:
    """The application's review worker."""
    reviewer = getattr(request.app.state, "reviewer", None)
    if reviewer is None:
        reviewer = ReviewWorker(database_path(request), get_pool(request), get_broker(request))
        request.app.state.reviewer = reviewer
    return reviewer


def ratings_report(request: Request, conn: sqlite3.Connection) -> dict[str, Any]:
    """The current ratings report."""
    settings = request_settings(request)
    return build_report(conn, settings.target, settings.anchor)


def strongest_name(report: dict[str, Any]) -> str | None:
    """The highest-rated settled player, falling back to the highest rated."""
    settled = [entry for entry in report["players"] if entry["settled"] and not entry["retired"]]
    pool = settled or [entry for entry in report["players"] if entry["games"] and not entry["retired"]]
    pool = pool or report["players"]
    return pool[0]["name"] if pool else None


def resolve_engine(request: Request, conn: sqlite3.Connection, engine: str | None) -> tuple[str, str]:
    """Resolve an engine id, or `strongest`, into a player name and its spec."""
    name = (engine or "strongest").strip()
    if name in ("", "strongest"):
        chosen = strongest_name(ratings_report(request, conn))
        if chosen is None:
            raise api_error("engine_not_found", "the arena has no players yet", status_code=404)
        name = chosen
    row = store.get_player(conn, name)
    if row is None:
        raise api_error("engine_not_found", f"unknown engine {name}", status_code=404)
    return name, str(row["spec"])
