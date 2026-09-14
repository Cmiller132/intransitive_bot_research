"""Review job status and its event stream."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from arena.api.deps import api_error, get_broker, get_db
from arena.db import store
from arena.events import sse_stream

router = APIRouter(prefix="/api/jobs", tags=["jobs"])
Db = Annotated[sqlite3.Connection, Depends(get_db)]
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.get("/{job_id}")
def job_status(job_id: str, db: Db) -> dict[str, Any]:
    """The status of one review job."""
    job = store.get_review_job(db, job_id)
    if job is None:
        raise api_error("job_not_found", f"unknown job {job_id}", status_code=404)
    return job


@router.get("/{job_id}/events")
def job_events(job_id: str, request: Request, db: Db) -> StreamingResponse:
    """Stream one review job's progress as server-sent events."""
    subscription = get_broker(request).subscribe(f"job:{job_id}")
    job = store.get_review_job(db, job_id)
    if job is None:
        subscription.close()
        raise api_error("job_not_found", f"unknown job {job_id}", status_code=404)
    return StreamingResponse(sse_stream(subscription, initial=job), media_type="text/event-stream", headers=SSE_HEADERS)
