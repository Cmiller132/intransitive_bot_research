"""On-demand game reviews driven by the analysis pool."""

from __future__ import annotations

import logging
import queue
import secrets
import threading
from typing import Any

from arena.core.game import GameRecord, PositionState, replay
from arena.db import store
from arena.engines import AnalysisPool, EngineError, evaluate, played_evidence
from arena.events import Broker

from .quality import accuracy, label

LOG = logging.getLogger("arena.review")
REVIEW_MULTIPV = 128
REVIEW_SIMS = {"standard": 128, "deep": 800}


def _move_key(move: Any) -> tuple[int, int]:
    """The (from, to) pair of a move given as a dict or a record."""
    if isinstance(move, dict):
        return int(move["from"]), int(move["to"])
    return int(move.from_), int(move.to)


def _phase(state: PositionState, ply: int) -> str:
    """The game phase a ply belongs to."""
    if ply < 20:
        return "opening"
    return "endgame" if sum(cell != 0 for cell in state.board) <= 10 else "middlegame"


def _mean(values: list[float]) -> float:
    """The mean of a list, or zero when it is empty."""
    return round(sum(values) / len(values), 2) if values else 0.0


def build_review(record: GameRecord, positions: list[PositionState], summary: dict[str, Any]) -> dict[str, Any]:
    """Build the public review from one response per analysed ply."""
    payloads = {int(item["ply"]): item["response"] for item in summary.get("plies", [])}
    reviewed: list[dict[str, Any]] = []
    by_side: dict[str, list[float]] = {"blue": [], "red": []}
    by_phase: dict[str, dict[str, list[float]]] = {
        phase: {"blue": [], "red": []} for phase in ("opening", "middlegame", "endgame")
    }
    agreement = {"blue": [0, 0], "red": [0, 0]}
    book_plies = int(
        record.meta.get("opening_plies") or record.meta.get("book_plies") or record.tags.get("BookPlies") or 0
    )

    for ply, move in enumerate(record.moves):
        response = payloads.get(ply)
        if response is None:
            continue
        lines = list((response.get("search") or {}).get("lines") or [])
        if not lines:
            continue
        played = _move_key(move)
        best_line = lines[0]
        best = _move_key(best_line["move"])
        best_q = float(best_line.get("q", 0.0))
        evidence = played_evidence(response, {"from": played[0], "to": played[1]})
        if evidence is None:
            continue
        played_q = float(evidence["q"])
        best_score = (1.0 + best_q) / 2.0
        played_score = (1.0 + played_q) / 2.0
        loss = round(max(0.0, min(1.0, best_score - played_score)), 4)
        second_loss = None
        if len(lines) > 1:
            second_loss = max(0.0, best_score - (1.0 + float(lines[1].get("q", 0.0))) / 2.0)
        side = positions[ply].to_move
        quality = label(
            loss,
            played_is_top=played == best,
            best_score=best_score,
            played_score=played_score,
            second_loss=second_loss,
            book=ply < book_plies,
        )
        score = accuracy(loss)
        phase = _phase(positions[ply], ply)
        by_side[side].append(score)
        by_phase[phase][side].append(score)
        agreement[side][1] += 1
        agreement[side][0] += int(played == best)
        value_before = float(response.get("value", {}).get("blue", 0.0))
        if ply + 1 in payloads:
            value_after = float(payloads[ply + 1].get("value", {}).get("blue", value_before))
        elif positions[ply + 1].result is not None:
            winner = positions[ply + 1].result.winner
            value_after = 1.0 if winner == "blue" else (-1.0 if winner == "red" else 0.0)
        else:
            value_after = value_before
        reviewed.append(
            {
                "ply": ply,
                "move": {"from": played[0], "to": played[1]},
                "loss": loss,
                "label": quality,
                "best": {"from": best[0], "to": best[1]},
                "played_evidence": evidence,
                "lines": lines[:4],
                "value_before": round(value_before, 4),
                "value_after": round(value_after, 4),
            }
        )

    key_moments = [
        item["ply"]
        for item in sorted(reviewed, key=lambda item: item["loss"], reverse=True)[:8]
        if item["loss"] >= 0.05
    ]
    key_moments.sort()
    return {
        "game_id": record.id,
        "engine": summary.get("engine", "unknown"),
        "budget": summary.get("budget", "standard"),
        "sims": int(summary.get("sims", 0)),
        "plies": reviewed,
        "accuracy": {
            "blue": _mean(by_side["blue"]),
            "red": _mean(by_side["red"]),
            "by_phase": {
                phase: {side: _mean(values) for side, values in sides.items()} for phase, sides in by_phase.items()
            },
        },
        "key_moments": key_moments,
        "agreement": {
            side: round(matches / total, 4) if total else 0.0 for side, (matches, total) in agreement.items()
        },
    }


class ReviewWorker:
    """A single background thread that reviews games one ply at a time."""

    def __init__(self, database_path: str, pool: AnalysisPool, broker: Broker):
        self.database_path = database_path
        self.pool = pool
        self.broker = broker
        self.queue: queue.Queue[tuple[str, str, str, str, str, int] | None] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.stopping = threading.Event()

    def start(self) -> None:
        """Start the worker thread."""
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True, name="arena-review")
            self.thread.start()

    def submit(self, conn: Any, game_id: str, engine: str, spec: str, budget: str, sims: int) -> str:
        """Queue a review and return its job id."""
        job_id = secrets.token_hex(6)
        store.create_review_job(conn, job_id, game_id, engine, sims)
        self.queue.put((job_id, game_id, engine, spec, budget, sims))
        self.start()
        return job_id

    def _publish(self, job_id: str, status: str, progress: float, detail: str = "") -> None:
        """Announce a job's progress to its subscribers."""
        self.broker.publish(
            f"job:{job_id}",
            {
                "type": "job_done" if status == "done" else ("job_failed" if status == "failed" else "job"),
                "id": job_id,
                "status": status,
                "progress": round(progress, 4),
                "detail": detail,
            },
        )

    def _run(self) -> None:
        """Consume queued reviews until the worker is stopped."""
        while not self.stopping.is_set():
            item = self.queue.get()
            if item is None:
                return
            job_id, game_id, engine, spec, budget, sims = item
            conn = store.connect(self.database_path)
            try:
                self._review(conn, job_id, game_id, engine, spec, budget, sims)
            except (EngineError, LookupError, ValueError) as error:
                LOG.warning("review %s failed: %s", job_id, error)
                store.update_review_job(conn, job_id, status="failed", detail=str(error))
                self._publish(job_id, "failed", 0.0, str(error))
            finally:
                conn.close()

    def _review(self, conn: Any, job_id: str, game_id: str, engine: str, spec: str, budget: str, sims: int) -> None:
        """Analyse every ply of one game and store the review."""
        record = store.get_game(conn, game_id)
        if record is None:
            raise LookupError(f"unknown game {game_id}")
        positions = replay(record)
        store.update_review_job(conn, job_id, status="running", progress=0.0)
        self._publish(job_id, "running", 0.0)
        plies = []
        total = max(1, len(record.moves))
        for ply in range(len(record.moves)):
            response = evaluate(
                self.pool, conn, engine, spec, {"game_id": game_id, "ply": ply}, {"sims": sims}, REVIEW_MULTIPV
            )
            plies.append({"ply": ply, "response": response})
            progress = (ply + 1) / total
            store.update_review_job(conn, job_id, progress=progress, detail=f"ply {ply + 1}/{total}")
            self._publish(job_id, "running", progress, f"ply {ply + 1}/{total}")
        review = build_review(record, positions, {"engine": engine, "budget": budget, "sims": sims, "plies": plies})
        store.save_review(conn, game_id, review)
        store.update_review_job(conn, job_id, status="done", progress=1.0, detail="")
        self._publish(job_id, "done", 1.0)

    def shutdown(self) -> None:
        """Stop the worker thread."""
        self.stopping.set()
        self.queue.put(None)
        if self.thread is not None:
            self.thread.join(timeout=10.0)
            self.thread = None
