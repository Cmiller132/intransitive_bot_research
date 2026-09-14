"""FastAPI application backbone, guards and the shared error envelope."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from arena.analysis.review import ReviewWorker
from arena.db import store
from arena.engines import AnalysisPool
from arena.events import Broker
from arena.scheduler import Scheduler
from arena.settings import Settings, get_settings

from .deps import is_admin
from .routers import admin, analysis, engines, export, games, jobs, live, meta, positions, ratings

LOG = logging.getLogger("arena.api")
IMPORTS_PER_MINUTE = 12
ANALYSE_PATH = re.compile(r"^/api/games/[^/]+/analyse$")
PUBLIC_POSTS = ("/api/eval", "/api/export", "/api/games/import")


class RateLimiter:
    """A per-client sliding-window request limiter."""

    def __init__(self, per_window: int, window_seconds: float = 60.0):
        self.limit = max(1, int(per_window))
        self.window_seconds = float(window_seconds)
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        """Record one call and report whether it stays inside the window."""
        stamp = time.monotonic() if now is None else now
        cutoff = stamp - self.window_seconds
        with self._lock:
            calls = self._calls[client]
            while calls and calls[0] <= cutoff:
                calls.popleft()
            if len(calls) >= self.limit:
                return False
            calls.append(stamp)
            return True


def error_response(status_code: int, code: str, message: str, detail: Any = None) -> JSONResponse:
    """Build the public error envelope."""
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    if detail is not None:
        payload["error"]["detail"] = detail
    return JSONResponse(status_code=status_code, content=payload)


def _http_error(_request: Request, exc: Exception) -> JSONResponse:
    """Render an HTTPException in the public envelope."""
    assert isinstance(exc, StarletteHTTPException)
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        return error_response(exc.status_code, str(detail["code"]), str(detail["message"]), detail.get("detail"))
    return error_response(exc.status_code, f"http_{exc.status_code}", str(detail))


def _validation_error(_request: Request, exc: Exception) -> JSONResponse:
    """Render a request validation failure in the public envelope."""
    assert isinstance(exc, RequestValidationError)
    return error_response(422, "validation_error", "Request validation failed", str(exc.errors()))


def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Render an unhandled error in the public envelope."""
    LOG.exception("unhandled API error for %s", request.url.path, exc_info=exc)
    return error_response(500, "internal_error", "Internal server error")


def create_app(settings: Settings | None = None, *, database_path: str | None = None) -> FastAPI:
    """Build the application; injectable state keeps tests isolated."""
    configured = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Own the database, the analysis pool, the reviewer and the scheduler."""
        path = str(application.state.database_path or configured.database_path)
        application.state.database_path = path
        store.connect(path).close()
        application.state.broker = Broker()
        application.state.pool = AnalysisPool(configured)
        application.state.reviewer = ReviewWorker(path, application.state.pool, application.state.broker)
        application.state.scheduler = None
        if configured.scheduler:
            scheduler = Scheduler(configured, application.state.broker)
            application.state.scheduler = scheduler
            scheduler.start()
        try:
            yield
        finally:
            if application.state.scheduler is not None:
                application.state.scheduler.shutdown()
                application.state.scheduler = None
            application.state.reviewer.shutdown()
            application.state.pool.close()

    application = FastAPI(title=configured.name, version=configured.version, lifespan=lifespan)
    application.state.settings = configured
    application.state.database_path = database_path
    application.state.broker = Broker()
    application.state.pool = None
    application.state.reviewer = None
    application.state.scheduler = None
    application.state.import_limiter = RateLimiter(IMPORTS_PER_MINUTE)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def guards(request: Request, call_next):
        """Apply the public rate limits and the administrative token guard."""
        path = request.url.path
        method = request.method
        privileged = is_admin(request, request.headers.get("X-Admin-Token"))
        client = request.client.host if request.client else "unknown"
        if not privileged and method == "POST":
            if path == "/api/games/import" and not request.app.state.import_limiter.allow(client):
                return error_response(429, "import_rate_limited", "Game import rate limit exceeded")
        mutating = method not in {"GET", "HEAD", "OPTIONS"}
        exempt = method == "POST" and (path in PUBLIC_POSTS or bool(ANALYSE_PATH.match(path)))
        settings_now: Settings = request.app.state.settings
        if settings_now.public and mutating and not exempt and not privileged:
            if not settings_now.admin_token:
                return error_response(503, "admin_not_configured", "ARENA_ADMIN_TOKEN is not set")
            return error_response(401, "admin_required", "A valid X-Admin-Token is required")
        return await call_next(request)

    application.add_exception_handler(StarletteHTTPException, _http_error)
    application.add_exception_handler(RequestValidationError, _validation_error)
    application.add_exception_handler(Exception, _unexpected_error)

    for router in (
        meta.router,
        engines.router,
        positions.router,
        games.router,
        jobs.router,
        export.router,
        analysis.router,
        ratings.router,
        live.router,
        admin.router,
    ):
        application.include_router(router)
    return application


app = create_app()
