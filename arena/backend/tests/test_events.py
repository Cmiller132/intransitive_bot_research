"""Slow stream consumers recover from bounded-queue overflow."""

from arena import events


def test_live_overflow_requests_resync_before_the_latest_move(monkeypatch) -> None:
    monkeypatch.setattr(events, "QUEUE_MAX", 2)
    broker = events.Broker()
    subscription = broker.subscribe("live")
    for ply in range(5):
        broker.publish("live", {"type": "move", "ply": ply})
    assert subscription.get(0) == {"type": "resync"}
    assert subscription.get(0) == {"type": "move", "ply": 4}
    assert subscription.get(0) is None


def test_job_overflow_retains_the_final_status(monkeypatch) -> None:
    monkeypatch.setattr(events, "QUEUE_MAX", 2)
    broker = events.Broker()
    subscription = broker.subscribe("job:1")
    broker.publish("job:1", {"type": "job", "progress": 0.1})
    broker.publish("job:1", {"type": "job", "progress": 0.5})
    broker.publish("job:1", {"type": "job_done", "progress": 1})
    assert subscription.get(0) == {"type": "job_done", "progress": 1}
