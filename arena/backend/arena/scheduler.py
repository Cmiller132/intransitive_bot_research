"""The job loop: bot processes with --stream, per-move stats and database writes."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
from typing import Any

from arena.core.game import GameRecord, MoveRecord, PlayerRef, Result
from arena.core.notation import parse_token
from arena.db import store
from arena.events import Broker
from arena.players import resolve_spec, scan_players
from arena.ratings import build_report, expected_score, now_iso, save_history
from arena.settings import Settings

LOG = logging.getLogger("arena.scheduler")
JOB_TIMEOUT = 1800.0
FAIL_STREAK = 3
IDLE_SLEEP = 2.0
# The KataGo server's rating-game pairer: opponents within this many Elo,
# the candidate's rating jittered by its uncertainty times this scale.
OPPONENT_RANGE = 1200.0
VARIABILITY = 1.0
# Weight of the anchor mode against the server's three modes at 1 each.
ANCHOR_SHARE = 0.5


def half_width_of(entry: dict[str, Any]) -> float:
    return math.inf if entry["half_width"] is None else float(entry["half_width"])


OPENING_PLIES = 8
END_REASONS = {
    "Goal": "corner",
    "Elimination": "no_pieces",
    "Stalemate": "no_moves",
    "CaptureClock": "stagnation",
    "Forfeit": "forfeit",
}


def convert_stats(info: dict[str, Any] | None, mover: str) -> dict[str, Any] | None:
    """Re-express the bot's search info with absolute moves from Blue's view."""
    if not isinstance(info, dict):
        return None
    sign = 1.0 if mover == "blue" else -1.0
    top = []
    for line in info.get("top") or []:
        origin, target = parse_token(str(line["move"]))
        top.append(
            {
                "move": {"from": origin, "to": target},
                "visits": int(line.get("visits", 0)),
                "q": round(sign * float(line.get("q", 0.0)), 4),
            }
        )
    plies_left = info.get("plies_left")
    return {
        "sims": int(info.get("sims", 0)),
        "value": round(sign * float(info.get("value", 0.0)), 4),
        "plies_left": None if plies_left is None else round(float(plies_left), 2),
        "q": round(sign * float(info.get("q", 0.0)), 4),
        "pi": round(float(info.get("pi", 0.0)), 4),
        "exact_win": bool(info.get("exact_win", False)),
        "top": top,
    }


def kill_process(proc: subprocess.Popen[bytes], hard: bool) -> None:
    """Signal a job's whole process group where the platform supports it."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL if hard else signal.SIGTERM)
        elif hard:
            proc.kill()
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        pass


class Job:
    """One running bot process comparing a candidate against a reference."""

    def __init__(
        self,
        ident: int,
        a: str,
        b: str,
        seed: int,
        proc: subprocess.Popen[bytes],
        err: Any,
        ratings_before: dict[str, float],
    ):
        self.id = ident
        self.a = a
        self.b = b
        self.seed = seed
        self.proc = proc
        self.err = err
        self.ratings_before = ratings_before
        self.started = time.time()
        self.started_iso = now_iso()
        self.killed = False
        self.records: dict[tuple[int, int], GameRecord] = {}
        self.finished_games = 0
        self.reader: threading.Thread | None = None

    def stderr_tail(self) -> str:
        """The last 2000 characters the process wrote to stderr."""
        self.err.seek(0)
        return self.err.read().decode("utf-8", "replace")[-2000:]

    def close(self) -> None:
        """Close the captured stderr stream."""
        self.err.close()

    def summary(self) -> dict[str, Any]:
        """The live view of this job and its games."""
        return {
            "id": self.id,
            "a": self.a,
            "b": self.b,
            "seed": self.seed,
            "started": self.started_iso,
            "games": [
                {
                    "game_id": record.id,
                    "pair": key[0],
                    "game": key[1],
                    "ply_count": len(record.moves),
                    "live": record.live,
                }
                for key, record in sorted(self.records.items())
            ],
        }


class Scheduler:
    """Chooses pairings, runs bot processes and records their games."""

    def __init__(self, settings: Settings, broker: Broker):
        self.settings = settings
        self.broker = broker
        self.root = settings.root
        self.db_lock = threading.Lock()
        self.conn = store.connect(settings.database_path)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.running: list[Job] = []
        self.present: set[str] = set()
        self.known: dict[str, tuple[str, int, int]] = {}  # the players table as of the last scan
        self.fails: dict[str, int] = {}
        self.report: dict[str, Any] = {"fitted": now_iso(), "target": settings.target, "anchor": None, "players": []}
        self.random = random.Random()
        self.thread: threading.Thread | None = None
        settings.players_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "records").mkdir(parents=True, exist_ok=True)
        self.load_fails()
        self.drop_abandoned()

    def drop_abandoned(self) -> None:
        """Delete the games a previous service run left live."""
        with self.db_lock:
            rows = self.conn.execute("SELECT id FROM games WHERE source = 'arena' AND live = 1").fetchall()
            for row in rows:
                store.delete_game(self.conn, row["id"])
            self.conn.commit()
        if rows:
            LOG.info("dropped %d abandoned live games", len(rows))

    def load_fails(self) -> None:
        """Rebuild each player's consecutive job-failure count from the jobs table."""
        with self.db_lock:
            players = [row["name"] for row in self.conn.execute("SELECT name FROM players")]
            for name in players:
                rows = self.conn.execute(
                    "SELECT ok FROM jobs WHERE (a = ? OR b = ?) AND finished IS NOT NULL ORDER BY id DESC LIMIT ?",
                    (name, name, FAIL_STREAK),
                ).fetchall()
                streak = 0
                for row in rows:
                    if row["ok"]:
                        break
                    streak += 1
                self.fails[name] = streak

    def scan(self) -> bool:
        """Register new and refreshed player directories; True when the
        players table differs from the last scan, whether through this scan
        or through the API (uploads, replacements, retirements)."""
        present = set()
        for meta in scan_players(self.settings.players_dir):
            present.add(meta["name"])
            with self.db_lock:
                row = store.get_player(self.conn, meta["name"])
                if row is None or meta["added"] > row["added"]:
                    store.upsert_player(self.conn, meta["name"], meta["spec"], meta["added"])
                    LOG.info(
                        "player %s %s spec=%s", "added" if row is None else "refreshed", meta["name"], meta["spec"]
                    )
        self.present = present
        with self.db_lock:
            rows = self.conn.execute("SELECT name, added, broken, retired FROM players").fetchall()
        players = {row["name"]: (row["added"], row["broken"], row["retired"]) for row in rows}
        for name, (added, _, _) in players.items():
            previous = self.known.get(name)
            if previous is not None and previous[0] != added:
                self.fails[name] = 0
        for name in present:
            self.fails.setdefault(name, 0)
        changed = players != self.known
        self.known = players
        return changed

    def fit(self, publish: bool = False) -> dict[str, Any]:
        """Refit the ratings, optionally storing history and announcing the result."""
        with self.db_lock:
            report = build_report(self.conn, self.settings.target, self.settings.anchor)
            if publish:
                save_history(self.conn, report)
        self.report = report
        if publish:
            self.broker.publish(
                "live",
                {
                    "type": "fit",
                    "fitted": report["fitted"],
                    "anchor": report["anchor"],
                    "players": [
                        {
                            "name": entry["name"],
                            "rating": entry["rating"],
                            "half_width": entry["half_width"],
                            "games": entry["games"],
                            "settled": entry["settled"],
                        }
                        for entry in report["players"]
                    ],
                },
            )
        return report

    def eligible(self) -> list[dict[str, Any]]:
        """The players that may be scheduled right now."""
        return [
            entry
            for entry in self.report["players"]
            if entry["name"] in self.present and not entry["broken"] and not entry["retired"]
        ]

    def wants_games(self, entry: dict[str, Any]) -> bool:
        """An unsettled player under its game budget (`Settings.max_games`)."""
        return not entry["settled"] and entry["games"] < self.settings.max_games

    def has_work(self) -> bool:
        """Whether any eligible player still needs games."""
        pool = self.eligible()
        if len(pool) < 2:
            return False
        return any(self.wants_games(entry) for entry in pool)

    def weighted_pick(self, ordered: list[dict[str, Any]]) -> dict[str, Any] | None:
        """One of the first ten of `ordered`, the front favoured by the server's
        exponential weighting with scale 2."""
        front = ordered[:10]
        if not front:
            return None
        weights = [math.exp(-index / 2.0) for index in range(len(front))]
        return self.random.choices(front, weights=weights, k=1)[0]

    def choose_candidate(self, pool: list[dict[str, Any]]) -> dict[str, Any] | None:
        """The unsettled player to rate next, by one of the server's three
        modes at equal weight (plus the anchor mode, see `pair`): the player
        that may be the strongest (highest upper confidence bound), the most
        uncertain, or the one with the fewest games."""
        unsettled = [entry for entry in pool if self.wants_games(entry)]
        if not unsettled:
            return None
        self.random.shuffle(unsettled)  # random tie-breaking, as the server's "?" ordering
        mode = self.random.choices(("high_elo", "high_uncertainty", "low_data"), k=1)[0]
        if mode == "high_elo":
            ordered = sorted(unsettled, key=lambda entry: -(entry["rating"] + half_width_of(entry)))
        elif mode == "high_uncertainty":
            ordered = sorted(unsettled, key=lambda entry: -half_width_of(entry))
        else:
            ordered = sorted(unsettled, key=lambda entry: entry["games"])
        return self.weighted_pick(ordered)

    def choose_opponent(self, candidate: dict[str, Any], pool: list[dict[str, Any]]) -> dict[str, Any] | None:
        """The server's opponent choice: the candidate's rating is jittered by
        its own uncertainty, the players within 1200 Elo of that are weighted
        by the variance of the predicted result, `p * (1 - p)`; with fewer than
        four of them, one of the two nearest weaker and two nearest stronger
        players at random."""
        others = [entry for entry in pool if entry["name"] != candidate["name"]]
        if not others:
            return None
        spread = 0.0 if candidate["se"] is None else float(candidate["se"])
        centre = candidate["rating"] + self.random.gauss(0.0, 1.0) * spread * VARIABILITY
        nearby = [entry for entry in others if abs(entry["rating"] - centre) <= OPPONENT_RANGE]
        if len(nearby) < 4:
            weaker = sorted((e for e in others if e["rating"] <= centre), key=lambda e: -e["rating"])[:2]
            stronger = sorted((e for e in others if e["rating"] >= centre), key=lambda e: e["rating"])[:2]
            nearby = list({entry["name"]: entry for entry in weaker + stronger}.values())
            return self.random.choice(nearby) if nearby else None
        weights = []
        for entry in nearby:
            gap = (entry["rating"] - centre) / VARIABILITY
            prob = expected_score(0.0, gap)
            weights.append(prob * (1.0 - prob))
        return self.random.choices(nearby, weights=weights, k=1)[0]

    def pair(self, pool: list[dict[str, Any]]) -> tuple[str, str] | None:
        """The next pairing: with weight `ANCHOR_SHARE` against the server's
        three modes, the candidate with the fewest games against the anchor
        plays the anchor; otherwise a candidate and its chosen opponent."""
        anchor = self.report.get("anchor")
        by_name = {entry["name"]: entry for entry in pool}
        unsettled = [entry for entry in pool if self.wants_games(entry) and entry["name"] != anchor]
        if anchor in by_name and unsettled and self.random.random() < ANCHOR_SHARE / (3.0 + ANCHOR_SHARE):
            self.random.shuffle(unsettled)
            candidate = min(unsettled, key=lambda entry: entry["opponents"].get(anchor, {}).get("games", 0))
            return candidate["name"], anchor
        candidate = self.choose_candidate(pool)
        if candidate is None:
            return None
        opponent = self.choose_opponent(candidate, pool)
        if opponent is None:
            return None
        return candidate["name"], opponent["name"]

    def allocate_job(self, a: str, b: str) -> int:
        """Reserve a job id in the database."""
        with self.db_lock:
            cursor = self.conn.execute("INSERT INTO jobs (a, b, seed, started) VALUES (?, ?, 0, ?)", (a, b, now_iso()))
            ident = int(cursor.lastrowid)
            self.conn.execute("UPDATE jobs SET seed = ? WHERE id = ?", (ident, ident))
            self.conn.commit()
        return ident

    def start_job(self, a: str, b: str) -> Job:
        """Launch one bot process comparing players a and b."""
        ident = self.allocate_job(a, b)
        with self.db_lock:
            specs = {name: store.get_player(self.conn, name)["spec"] for name in (a, b)}
        command = [
            *self.settings.bot_command(),
            "eval",
            "--candidate",
            resolve_spec(specs[a], self.settings.players_dir / a),
            "--reference",
            resolve_spec(specs[b], self.settings.players_dir / b),
            "--pairs",
            str(self.settings.pairs),
            "--sims",
            str(self.settings.sims),
            "--threads",
            "1",
            "--seed",
            str(ident),
            "--opening-plies",
            str(OPENING_PLIES),
            "--stream",
        ]
        environment = dict(os.environ, LD_LIBRARY_PATH=str(self.root))
        err = tempfile.TemporaryFile()
        proc = subprocess.Popen(
            command, cwd=str(self.root), stdout=subprocess.PIPE, stderr=err, env=environment, start_new_session=True
        )
        ratings = {entry["name"]: entry["rating"] for entry in self.report["players"]}
        job = Job(ident, a, b, ident, proc, err, {a: ratings.get(a, 0.0), b: ratings.get(b, 0.0)})
        job.reader = threading.Thread(target=self._read, args=(job,), daemon=True, name=f"arena-job-{ident}")
        with self.lock:
            self.running.append(job)
        job.reader.start()
        LOG.info("job %s start %s vs %s", ident, a, b)
        self.broker.publish(
            "live", {"type": "job_start", "job": ident, "a": a, "b": b, "seed": ident, "started": job.started_iso}
        )
        return job

    def _read(self, job: Job) -> None:
        """Consume one job's event stream until the process closes it."""
        stream = job.proc.stdout
        if stream is None:
            return
        with stream:
            for raw in stream:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    LOG.warning("job %s wrote an unparseable line", job.id)
                    continue
                try:
                    self._apply(job, event)
                except (KeyError, ValueError, TypeError, sqlite3.Error):
                    LOG.exception("job %s event %r failed", job.id, event.get("event"))

    def _apply(self, job: Job, event: dict[str, Any]) -> None:
        """Apply one stream event to the database and the live feed."""
        kind = event.get("event")
        if kind == "start":
            self._on_start(job, event)
        elif kind == "move":
            self._on_move(job, event)
        elif kind == "end":
            self._on_end(job, event)

    def _on_start(self, job: Job, event: dict[str, Any]) -> None:
        """Create the live games row for one game of a pair."""
        pair, index = int(event["pair"]), int(event["game"])
        first, second = (job.a, job.b) if index % 2 == 0 else (job.b, job.a)
        with self.db_lock:
            specs = {name: store.get_player(self.conn, name) for name in (job.a, job.b)}
        record = GameRecord(
            id=f"arena-{job.id}-{pair}-{index}",
            source="arena",
            frame="meaf",
            players={
                "blue": PlayerRef(
                    first,
                    kind="engine",
                    rating_before=job.ratings_before[first],
                    spec=specs[first]["spec"] if specs[first] else None,
                ),
                "red": PlayerRef(
                    second,
                    kind="engine",
                    rating_before=job.ratings_before[second],
                    spec=specs[second]["spec"] if specs[second] else None,
                ),
            },
            moves=[MoveRecord(*parse_token(token)) for token in event.get("opening", [])],
            result=Result(None, "unknown", ""),
            started_at=now_iso(),
            meta={
                "job": job.id,
                "pair": pair,
                "game": index,
                "seed": job.seed,
                "opening_plies": len(event.get("opening", [])),
            },
            live=True,
            sims=self.settings.sims,
        )
        job.records[(pair, index)] = record
        with self.db_lock:
            store.save_arena_game(self.conn, record, live=True)
        self.broker.publish(
            "live",
            {
                "type": "game_start",
                "job": job.id,
                "pair": pair,
                "game": index,
                "game_id": record.id,
                "blue": first,
                "red": second,
                "ply_count": len(record.moves),
            },
        )

    def _on_move(self, job: Job, event: dict[str, Any]) -> None:
        """Append one played move and its statistics to a live game."""
        key = (int(event["pair"]), int(event["game"]))
        record = job.records.get(key)
        if record is None:
            return
        ply = int(event["ply"])
        origin, target = parse_token(str(event["move"]))
        mover = "blue" if ply % 2 == 0 else "red"
        stats = convert_stats(event.get("info"), mover)
        if ply < len(record.moves):
            record.moves[ply].stats = stats
        elif ply == len(record.moves):
            record.moves.append(MoveRecord(origin, target, stats=stats))
        else:
            LOG.warning("job %s game %s skipped to ply %s", job.id, record.id, ply)
            return
        with self.db_lock:
            store.save_arena_game(self.conn, record, live=True)
        self.broker.publish(
            "live",
            {
                "type": "move",
                "job": job.id,
                "game_id": record.id,
                "ply": ply,
                "move": {"from": origin, "to": target},
                "stats": stats,
            },
        )

    def _on_end(self, job: Job, event: dict[str, Any]) -> None:
        """Finalise one game with its result."""
        key = (int(event["pair"]), int(event["game"]))
        record = job.records.get(key)
        if record is None:
            return
        winner = event.get("winner")
        record.result = Result(
            None if winner is None else ("blue" if int(winner) == 0 else "red"),
            END_REASONS.get(str(event.get("end")), "unknown"),
            str(event.get("end", "")),
        )
        record.ended_at = now_iso()
        record.live = False
        job.finished_games += 1
        with self.db_lock:
            store.save_arena_game(self.conn, record, live=False)
        self.broker.publish(
            "live",
            {
                "type": "game_end",
                "job": job.id,
                "game_id": record.id,
                "winner": record.result.winner,
                "reason": record.result.reason,
                "ply_count": len(record.moves),
            },
        )

    def finish(self, job: Job) -> bool:
        """Record a finished job; returns True when it produced every game."""
        if job.reader is not None:
            job.reader.join(timeout=30.0)
        code = job.proc.returncode
        expected = 2 * self.settings.pairs
        ok = not job.killed and code == 0 and job.finished_games >= expected
        with self.db_lock:
            self.conn.execute(
                "UPDATE jobs SET finished = ?, ok = ?, stderr = ? WHERE id = ?",
                (now_iso(), int(ok), None if ok else job.stderr_tail(), job.id),
            )
            for record in job.records.values():
                if record.live:
                    record.live = False
                    record.result = Result(None, "forfeit", "abandoned")
                    store.save_arena_game(self.conn, record, live=False)
            self.conn.commit()
        if ok:
            LOG.info("job %s done %s vs %s in %.1fs", job.id, job.a, job.b, time.time() - job.started)
            self.fails[job.a] = 0
            self.fails[job.b] = 0
        else:
            LOG.warning("job %s failed %s vs %s exit=%s killed=%s", job.id, job.a, job.b, code, job.killed)
            for name in (job.a, job.b):
                self.fails[name] = self.fails.get(name, 0) + 1
                if self.fails[name] >= FAIL_STREAK:
                    with self.db_lock:
                        store.set_player_flag(self.conn, name, "broken", True)
                    LOG.warning("player broken %s", name)
        job.close()
        self.broker.publish("live", {"type": "job_end", "job": job.id, "ok": ok})
        return ok

    def poll(self) -> bool:
        """Reap finished jobs and kill the ones that ran too long."""
        ended = False
        for job in list(self.running):
            if job.proc.poll() is None:
                if time.time() - job.started > JOB_TIMEOUT:
                    job.killed = True
                    kill_process(job.proc, hard=True)
                    job.proc.wait()
                else:
                    continue
            with self.lock:
                if job in self.running:
                    self.running.remove(job)
            self.finish(job)
            ended = True
        return ended

    def schedule(self) -> None:
        """Fill the free job slots with new pairings."""
        while len(self.running) < self.settings.jobs and self.has_work():
            pool = self.eligible()
            pairing = self.pair(pool)
            if pairing is None:
                return
            self.start_job(*pairing)

    def terminate(self) -> None:
        """Kill every running job without recording it."""
        for job in list(self.running):
            kill_process(job.proc, hard=False)
        for job in list(self.running):
            try:
                job.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                kill_process(job.proc, hard=True)
                job.proc.wait()
            if job.reader is not None:
                job.reader.join(timeout=5.0)
            job.close()
        with self.lock:
            self.running = []

    def run(self) -> None:
        """Run the scheduling loop until the stop event is set."""
        dirty = True
        while not self.stop.is_set():
            changed = self.scan()
            if changed or dirty:
                self.fit()
                dirty = False
            self.schedule()
            if self.poll():
                self.fit(publish=True)
                dirty = False
                continue
            if not changed:
                self.stop.wait(IDLE_SLEEP)
        self.terminate()
        LOG.info("scheduler stopped")

    def start(self) -> None:
        """Start the scheduling loop in a background thread."""
        self.thread = threading.Thread(target=self.run, daemon=True, name="arena-scheduler")
        self.thread.start()

    def shutdown(self) -> None:
        """Stop the loop, kill running jobs and close the database."""
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=30.0)
            self.thread = None
        else:
            self.terminate()
        self.conn.close()

    def live(self) -> dict[str, Any]:
        """The live view of every running job."""
        return {"jobs": [job.summary() for job in list(self.running)]}

    def status(self) -> dict[str, Any]:
        """The scheduler's own state, as served by /api/scheduler."""
        candidate = self.choose_candidate(self.eligible())
        return {
            "jobs_running": len(self.running),
            "slots": self.settings.jobs,
            "target": self.settings.target,
            "has_work": self.has_work(),
            "next_candidate": candidate["name"] if candidate else None,
        }
