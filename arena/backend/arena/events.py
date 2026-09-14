"""In-process publish/subscribe used by the server-sent-event endpoints."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator
from typing import Any

HEARTBEAT_SECONDS = 15.0
POLL_SECONDS = 1.0
QUEUE_MAX = 1000


class Subscription:
    """One subscriber's bounded queue of events on a topic."""

    def __init__(self, broker: Broker, topic: str):
        self.broker = broker
        self.topic = topic
        self.items: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=QUEUE_MAX)
        self._lock = threading.Lock()

    def put(self, payload: dict[str, Any]) -> None:
        """Offer an event; replace an overflowed backlog and signal live clients to resync."""
        with self._lock:
            if self.items.full():
                while True:
                    try:
                        self.items.get_nowait()
                    except queue.Empty:
                        break
                if self.topic == "live":
                    self.items.put_nowait({"type": "resync"})
            self.items.put_nowait(payload)

    def get(self, timeout: float) -> dict[str, Any] | None:
        """Wait for the next event, or None when the wait times out."""
        try:
            return self.items.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        """Stop receiving events."""
        self.broker.unsubscribe(self)


class Broker:
    """Fan out events to every subscriber of a topic."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[Subscription]] = {}

    def subscribe(self, topic: str) -> Subscription:
        """Register a new subscriber on a topic."""
        subscription = Subscription(self, topic)
        with self._lock:
            self._subscribers.setdefault(topic, []).append(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        """Remove one subscriber."""
        with self._lock:
            listeners = self._subscribers.get(subscription.topic, [])
            if subscription in listeners:
                listeners.remove(subscription)
            if not listeners:
                self._subscribers.pop(subscription.topic, None)

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Send one event to every subscriber of a topic."""
        with self._lock:
            listeners = list(self._subscribers.get(topic, ()))
        for listener in listeners:
            listener.put(payload)


async def sse_stream(
    subscription: Subscription, heartbeat: float = HEARTBEAT_SECONDS, *, initial: dict[str, Any] | None = None
) -> AsyncIterator[str]:
    """Yield text/event-stream frames for a subscription until the client leaves."""
    waited = 0.0
    try:
        yield ": open\n\n"
        while True:
            slice_seconds = min(POLL_SECONDS, heartbeat)
            if initial is not None:
                payload, initial = initial, None
            else:
                payload = await asyncio.to_thread(subscription.get, slice_seconds)
            if payload is None:
                waited += slice_seconds
                if waited >= heartbeat:
                    waited = 0.0
                    yield ": heartbeat\n\n"
                continue
            waited = 0.0
            body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            yield f"data: {body}\n\n"
            if payload.get("type") in ("job_done", "job_failed") or payload.get("status") in ("done", "failed"):
                return
    finally:
        subscription.close()
