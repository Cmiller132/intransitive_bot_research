"""Live games, the live event stream and the scheduler's own state."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from arena.api.deps import get_broker, request_settings
from arena.events import sse_stream

router = APIRouter(prefix="/api", tags=["live"])
SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}


@router.get("/live")
def live(request: Request) -> dict[str, Any]:
    """The running jobs and the games they are playing."""
    scheduler = getattr(request.app.state, "scheduler", None)
    return scheduler.live() if scheduler else {"jobs": []}


@router.get("/live/events")
def live_events(request: Request) -> StreamingResponse:
    """Stream job, game, move and fit events as they happen."""
    subscription = get_broker(request).subscribe("live")
    return StreamingResponse(sse_stream(subscription), media_type="text/event-stream", headers=SSE_HEADERS)


@router.get("/scheduler")
def scheduler_status(request: Request) -> dict[str, Any]:
    """The scheduler's slots, target and next candidate."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        settings = request_settings(request)
        return {
            "jobs_running": 0,
            "slots": settings.jobs,
            "target": settings.target,
            "has_work": False,
            "next_candidate": None,
        }
    return scheduler.status()
