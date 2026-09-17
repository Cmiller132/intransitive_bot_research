"""Runs the quickstart's steps from the dashboard: the same commands models/nnue/quickstart.sh issues, one job at a
time, with pauses, resumption and an automatic mode that follows the decisions.

A job is one of
* pipeline   check, generate, import, encode, train, evaluate for one run (any finished step is skipped);
* eval       one evaluation of a run's best.nnue with its own settings;
* build      cargo build --release -p cli.

Only one job runs at a time; a paused job does not hold the slot, so an evaluation can run while training waits between
epochs. Every step can be interrupted and continued: the generator publishes its finished games, the trainer resumes
from its last epoch checkpoint, the sequential test from its journal (a fixed-count evaluation starts again).

State lives in runs/nnue_dashboard/state.json (the quickstart ignores names starting with nnue_) and every step's
output in runs/nnue_dashboard/logs/<job>/<step>.log. After a restart of the server nothing starts on its own: running
and queued jobs come back paused and the automatic mode off.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from pathlib import Path

from . import analysis
from .sprt import JournalReader

PIPELINE_STEPS = ["check", "generate", "import", "encode", "train", "evaluate"]
NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")
TERMINAL = {"done", "failed", "cancelled"}
STEP_WORDS = {"check": "Checking", "generate": "Generating games", "import": "Importing", "encode": "Encoding",
              "train": "Training", "evaluate": "Evaluating", "build": "Building the bot"}
STEP_NOUNS = {"check": "the check", "generate": "game generation", "import": "the import", "encode": "encoding",
              "train": "training", "evaluate": "the evaluation", "build": "the build"}


def default_settings(threads: int) -> dict:
    return {
        "start": "models/nnue/examples/example.nnue",
        "name": "",
        "games": analysis.PLAN["ladder"][0],
        "nodes": analysis.PLAN["nodes"],
        "threads": threads,
        "hash": 64,
        "multipv": 1,
        "quiet": False,
        "epochs": "auto",       # auto: sized from the training steps (see epoch_plan); or a number
        "patience": None,       # None: never stop early; N: stop after N epochs without a better objective
        "passes": 4.0,
        "batch": 8192,          # the recipe's; "auto" (experimental) halves it on small data for more updates
        "lr": 0.0001,
        "ema": "auto",          # auto | off | a decay in (0, 1)
        "device": "cpu",
        "version": "auto",      # auto | 9
        "extra": [],            # [{"set": name, "share": 1.0}]
        "seed": None,           # None: the manifest's, else derived from the name as quickstart.sh does
        "steps": None,          # None: sized from the rows and passes
        "warmup": None,
        "pause_every_epoch": False,
        "eval": {"mode": "sprt", "sims": analysis.PLAN["decision_sims"], "pairs": analysis.PLAN["decision_pairs"],
                 "target": analysis.PLAN["sprt_targets"][0],  # a new line is generation 0
                 "cap": analysis.sprt_cap(analysis.PLAN, analysis.PLAN["ladder"][0])},
    }


def posix_cksum(data: bytes) -> int:
    """POSIX cksum, so a name gives the same seed here as in quickstart.sh."""
    table = []
    for i in range(256):
        c = i << 24
        for _ in range(8):
            c = ((c << 1) ^ 0x04C11DB7) if c & 0x80000000 else (c << 1)
        table.append(c & 0xFFFFFFFF)
    crc = 0
    for b in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[((crc >> 24) ^ b) & 0xFF]
    n = len(data)
    while n:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[((crc >> 24) ^ (n & 0xFF)) & 0xFF]
        n >>= 8
    return (~crc) & 0xFFFFFFFF


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class StepFailed(Exception):
    pass


class Interrupted(Exception):
    pass


class Paused(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason, self.detail = reason, detail


class Runner:
    def __init__(self, repo: Path, python: str, bot: str | None = None, plan: dict | None = None):
        self.repo = repo.resolve()
        self.runs = self.repo / "runs"
        self.python = python
        self.bot = bot or "target/release/bot"
        self.plan = plan or {}
        self.home = self.runs / "nnue_dashboard"
        self.state_file = self.home / "state.json"
        self.probe_file = Path(__file__).with_name("probe.py")
        self.threads = os.cpu_count() or 8
        self.lock = threading.RLock()
        self.jobs: list[dict] = []
        self.auto = {"enabled": False, "max_generations": 3, "max_batches": 8, "pause_on_problems": True,
                     "pause_every_epoch": False, "adopted": 0, "batches": 0, "message": ""}
        self.current: str | None = None
        self.proc: subprocess.Popen | None = None
        self.interrupts: dict[str, str] = {}   # job id -> "pause" | "cancel"
        self.tails: dict[str, deque] = {}
        self.journals: dict[str, JournalReader] = {}
        self.eval_games: dict[str, int] = {}
        self.version = 0
        self.closing = False
        self.load()
        threading.Thread(target=self.schedule, name="scheduler", daemon=True).start()

    # ------------------------------------------------------------------ state
    def load(self) -> None:
        data = analysis.read_json(self.state_file) or {}
        self.jobs = data.get("jobs", [])
        self.auto.update({k: v for k, v in (data.get("auto") or {}).items() if k in self.auto})
        self.auto["enabled"] = False
        for job in self.jobs:
            if job["state"] in ("running", "queued"):
                job["state"] = "paused"
                job["pause"] = {"reason": "restart", "detail": "The dashboard was restarted; continue when ready."}
                for step in job["steps"].values():
                    if step["state"] == "running":
                        step["state"] = "paused"

    def save(self) -> None:
        with self.lock:
            self.version += 1
            self.home.mkdir(parents=True, exist_ok=True)
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"jobs": self.jobs[-200:], "auto": self.auto}, indent=1, default=str))
            tmp.replace(self.state_file)

    def job(self, job_id: str) -> dict:
        for job in self.jobs:
            if job["id"] == job_id:
                return job
        raise KeyError(f"no job {job_id}")

    def log_path(self, job: dict, step: str) -> Path:
        path = self.home / "logs" / job["id"] / f"{step}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------ creating jobs
    def new_job(self, kind: str, run: str | None, settings: dict, auto: bool = False) -> dict:
        steps = PIPELINE_STEPS if kind == "pipeline" else ["evaluate"] if kind == "eval" else ["build"]
        now = time.time()
        job = {"id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4], "kind": kind, "run": run,
               "settings": settings, "state": "queued", "queued_at": now, "created": now, "updated": now,
               "step": steps[0], "steps": {s: {"state": "pending", "note": "", "started": None, "ended": None} for s in steps},
               "pause": None, "error": None, "auto": auto, "result": {}, "info": {}, "commands": []}
        with self.lock:
            self.jobs.append(job)
            self.save()
        return job

    def validate_pipeline(self, raw: dict) -> dict:
        s = default_settings(self.threads)
        for k, v in (raw or {}).items():
            if k == "eval":
                s["eval"].update({kk: vv for kk, vv in (v or {}).items() if kk in s["eval"]})
            elif k in s or k in ("restart_training",):
                s[k] = v
        s["games"] = int(s.get("games") or 0)
        name = str(s["name"]).strip()
        if not NAME.fullmatch(name) or name.startswith("nnue_") or ".replaced_" in name:
            raise ValueError("a name of letters, digits, '.', '_' and '-' that does not start with nnue_")
        s["name"] = name
        s["start"] = str(s["start"]).strip()
        if not s["start"]:
            raise ValueError("a starting network is required")
        if s["batch"] != "auto":
            s["batch"] = int(s["batch"])
            if s["batch"] < 1:
                raise ValueError("batch must be auto or at least 1")
        for key, low in (("games", 0), ("nodes", 1), ("threads", 1), ("hash", 1)):
            s[key] = int(s[key])
            if s[key] < low:
                raise ValueError(f"{key} must be at least {low}")
        if s["epochs"] != "auto":
            s["epochs"] = int(s["epochs"])
            if s["epochs"] < 1:
                raise ValueError("epochs must be auto or at least 1")
        if s.get("patience") not in (None, "", 0, "0"):
            s["patience"] = int(s["patience"])
            if s["patience"] < 1:
                raise ValueError("early stopping needs at least 1 epoch of patience")
        else:
            s["patience"] = None
        s["passes"], s["lr"] = float(s["passes"]), float(s["lr"])
        if s["games"] < 1 and not s["extra"]:
            raise ValueError("games must be at least 1, or 0 with earlier batches to train on")
        if s["multipv"] not in (1, 2):
            s["multipv"] = 2 if str(s["multipv"]) == "2" else 1
        if s["ema"] not in ("auto", "off"):
            s["ema"] = float(s["ema"])
            if not 0 < s["ema"] < 1:
                raise ValueError("EMA must be auto, off or a decay between 0 and 1")
        if str(s["version"]) not in ("auto", "8", "9"):
            raise ValueError("format must be auto, 8 or 9")
        own = f"selfplay_{name}"
        s["extra"] = [{"set": str(e["set"]), "share": float(e.get("share", 1.0))} for e in s["extra"]
                      if str(e.get("set", "")) and (int(s.get("games") or 0) == 0 or not str(e["set"]).startswith(own))]
        self.validate_eval(s["eval"])
        with self.lock:
            busy = [j for j in self.jobs if j["run"] == name and j["kind"] == "pipeline" and j["state"] not in TERMINAL]
        if busy:
            raise ValueError(f"{name} already has an unfinished job; continue or cancel it first")
        return s

    @staticmethod
    def validate_eval(e: dict) -> dict:
        e["mode"] = "sprt" if e.get("mode") == "sprt" else "fixed"
        e["sims"] = int(e.get("sims") or 16)
        e["pairs"] = int(e.get("pairs") or 100)
        e["target"] = float(e.get("target") or 0.52)
        e["cap"] = int(e.get("cap") or 3008)
        if e["sims"] < 1 or e["pairs"] < 1:
            raise ValueError("the evaluation needs positive sims and pairs")
        if e["mode"] == "sprt" and (not 0.5 < e["target"] < 1 or e["cap"] < 128 or e["cap"] % 16):
            raise ValueError("the sequential test needs a target between .5 and 1 and a cap that is a multiple of 16, at least 128")
        return e

    def create_pipeline(self, raw: dict, auto: bool = False) -> dict:
        s = self.validate_pipeline(raw)
        return self.new_job("pipeline", s["name"], s, auto)

    def create_eval(self, run: str, raw: dict, auto: bool = False) -> dict:
        if not (self.runs / run / "best.nnue").is_file():
            raise ValueError(f"runs/{run}/best.nnue does not exist yet")
        e = self.validate_eval(dict(raw or {}))
        reference = str((raw or {}).get("reference") or "").strip()
        if reference and not (self.repo / reference).is_file():
            raise ValueError(f"no such network: {reference}")
        e["reference"] = reference or None
        e["threads"] = int(raw.get("threads") or self.threads)
        e["hash"] = int(raw.get("hash") or 64)
        return self.new_job("eval", run, {"eval": e}, auto)

    def create_retrain(self, run: str, raw: dict) -> dict:
        """Train `run` again on exactly the batches it was trained on, with other settings. No games are played or
        imported (the mixture is passed as it was, in its order); the old run moves aside when training starts. Jobs of
        the run that are queued or paused are cancelled; a running one has to be paused first."""
        saved = analysis.read_json(self.runs / run / "config.json")
        if not saved:
            raise ValueError(f"runs/{run} has not been trained yet, so there is nothing to retrain")
        cfg = saved.get("config") or {}
        with self.lock:
            if any(j["run"] == run and j["state"] == "running" for j in self.jobs):
                raise ValueError(f"{run} has a running job; pause or cancel it first")
            for job in self.jobs:
                if job["run"] == run and job["state"] in ("queued", "paused"):
                    job.update(state="cancelled", updated=time.time(),
                               error="replaced by retraining with other settings")
            self.save()
        gen = ((analysis.read_json(self.runs / "nnue_selfplay" / run / "manifest.json") or {}).get("settings")) or {}
        raw = dict(raw or {})
        settings = {
            "start": cfg.get("init"), "name": run, "games": 0, "nodes": gen.get("nodes") or analysis.PLAN["nodes"],
            "threads": int(raw.pop("threads", 0) or cfg.get("threads") or self.threads), "hash": 64,
            "multipv": gen.get("multipv", 1),
            "extra": [{"set": name, "share": float(share)} for name, share in saved.get("data", [])],
            "seed": cfg.get("seed"), "epochs": "auto", "passes": 4.0, "batch": cfg.get("batch") or 8192,
            "lr": cfg.get("lr") or 0.0001, "ema": cfg["ema"] if cfg.get("ema") else "off", "device": cfg.get("device") or "cpu",
            "version": "9" if cfg.get("version") == 9 else "auto", "patience": None,
            "restart_training": True,
        }
        for key in ("epochs", "passes", "batch", "lr", "ema", "device", "version", "patience", "pause_every_epoch", "seed"):
            if key in raw:
                settings[key] = raw[key]
        if raw.get("eval"):
            settings["eval"] = raw["eval"]
        if not settings["start"]:
            raise ValueError(f"runs/{run}/config.json does not name its starting network")
        return self.create_pipeline(settings)

    def restore_training(self, run: str, earlier: str) -> dict:
        """Put a moved-aside training back as the run: the current training moves aside in its place, so nothing is
        lost and the swap can be undone the same way."""
        if not re.fullmatch(rf"{re.escape(run)}\.replaced_\d{{8}}_\d{{6}}", earlier or ""):
            raise ValueError(f"{earlier} is not an earlier training of {run}")
        source = self.runs / earlier
        if not (source / "config.json").is_file():
            raise ValueError(f"runs/{earlier} holds no training")
        with self.lock:
            live = [j for j in self.jobs if j["run"] == run and j["state"] in ("running", "queued", "paused")]
            if live:
                raise ValueError(f"{run} has an unfinished job ({live[-1]['state']}); finish or cancel it first")
            current = self.runs / run
            moved = None
            if current.exists():
                stamp = time.strftime("%Y%m%d_%H%M%S")
                moved = f"{run}.replaced_{stamp}"
                while (self.runs / moved).exists():
                    time.sleep(1.1)
                    moved = f"{run}.replaced_{time.strftime('%Y%m%d_%H%M%S')}"
                current.rename(self.runs / moved)
            source.rename(current)
            self.version += 1
        return {"restored": earlier, "moved_aside": moved}

    def create_build(self) -> dict:
        return self.new_job("build", None, {})

    # ------------------------------------------------------------------ actions
    def action(self, job_id: str, action: str, payload: dict | None = None) -> dict:
        payload = payload or {}
        with self.lock:
            job = self.job(job_id)
            state = job["state"]
            if action == "pause":
                if state == "running":
                    self.interrupt(job, "pause")
                elif state == "queued":
                    job["state"], job["pause"] = "paused", {"reason": "user", "detail": "Paused before it started."}
            elif action in ("continue", "run_to_end", "end_training"):
                job.pop("pause_after_epoch", None)
                if state not in ("paused", "failed"):
                    raise ValueError(f"{job_id} is {state}")
                if action == "run_to_end":
                    job["settings"]["pause_every_epoch"] = False
                if action == "end_training":
                    job["settings"]["end_training_early"] = True
                job.update(state="queued", queued_at=time.time(), pause=None, error=None)
            elif action == "pause_every_epoch":
                job["settings"]["pause_every_epoch"] = bool(payload.get("on", True))
            elif action == "pause_after_epoch":
                if job["step"] != "train" or state != "running":
                    raise ValueError("pausing after an epoch applies while training runs")
                job["pause_after_epoch"] = True
            elif action == "next_epoch":
                if state != "paused" or job["step"] != "train":
                    raise ValueError("the job is not paused between epochs")
                job["pause_after_epoch"] = True
                job.update(state="queued", queued_at=time.time(), pause=None, error=None)
            elif action == "restart_training":
                if state == "running":
                    raise ValueError("pause the job first")
                if job["kind"] != "pipeline" or state not in ("paused", "failed") or job["step"] != "train":
                    # anything else (an evaluation job, a cancelled or finished pipeline, a pause during the evaluation)
                    # retrains from the run's saved configuration as a new job
                    if not job["run"]:
                        raise ValueError("this job has no run to retrain")
                    return self.create_retrain(job["run"], payload)
                for key in ("epochs", "passes", "batch", "lr", "ema", "device", "version", "steps", "warmup", "pause_every_epoch"):
                    if key in payload:
                        job["settings"][key] = payload[key]
                if "eval" in payload:
                    job["settings"]["eval"].update(payload["eval"])
                job["settings"]["restart_training"] = True
                job["settings"].pop("end_training_early", None)
                for step in ("train", "evaluate"):
                    job["steps"][step].update(state="pending", note="")
                job.update(step="train", state="queued", queued_at=time.time(), pause=None, error=None)
            elif action == "decide_now":
                info = job.get("info") or {}
                if job["step"] != "evaluate" or job["settings"]["eval"]["mode"] != "sprt" or not info.get("journal"):
                    raise ValueError("only a sequential test can be stopped and decided")
                reader = JournalReader(self.repo / info["journal"])
                seq = reader.read() or {}
                need = analysis.PLAN["decision_pairs"]
                if seq.get("pairs", 0) < need:
                    raise ValueError(f"a test stopped by hand decides from {need} pairs; it has {seq.get('pairs', 0)}")
                if state == "running":
                    self.interrupt(job, "decide")
                elif state == "paused":
                    self.report_from_journal(job)
                    job["steps"]["evaluate"].update(state="done", ended=time.time(),
                                                    note=f"stopped by hand at {seq['pairs']:,} pairs ({info['eval_file']})")
                    job.update(state="done", step=None, pause=None)
                    threading.Thread(target=self.after, args=(job,), daemon=True).start()
                else:
                    raise ValueError(f"{job_id} is {state}")
            elif action == "cancel":
                if state == "running":
                    self.interrupt(job, "cancel")
                elif state not in TERMINAL:
                    job["state"] = "cancelled"
            elif action == "remove":
                if state not in TERMINAL and state != "paused":
                    raise ValueError("only finished, failed, cancelled or paused jobs can be removed")
                self.jobs.remove(job)
            else:
                raise ValueError(f"unknown action {action}")
            job["updated"] = time.time()
            self.save()
            return job

    def set_auto(self, cfg: dict) -> dict:
        with self.lock:
            was = self.auto["enabled"]
            for key in ("max_generations", "max_batches"):
                if key in cfg:
                    self.auto[key] = max(0, int(cfg[key]))
            for key in ("pause_on_problems", "pause_every_epoch", "enabled"):
                if key in cfg:
                    self.auto[key] = bool(cfg[key])
            if self.auto["enabled"] and not was:
                self.auto.update(adopted=0, batches=0, message="")
            self.save()
            return self.auto

    def start_auto_from(self, run: str) -> dict:
        """Turn the automatic mode on and queue what the run's decision asks for."""
        self.set_auto({"enabled": True})
        job = self.follow(run)
        if job is None and not self.busy():
            with self.lock:
                self.auto["enabled"] = False
                self.save()
        return self.auto

    def busy(self) -> bool:
        with self.lock:
            return any(j["state"] in ("running", "queued") for j in self.jobs)

    def interrupt(self, job: dict, why: str) -> None:
        self.interrupts[job["id"]] = why
        proc = self.proc
        if not proc or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            return

        def escalate():
            for sig, wait in ((signal.SIGTERM, 120), (signal.SIGKILL, 30)):
                try:
                    proc.wait(timeout=wait)
                    return
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, sig)
                    except (ProcessLookupError, PermissionError):
                        return

        threading.Thread(target=escalate, daemon=True).start()

    def shutdown(self) -> None:
        self.closing = True
        with self.lock:
            job = next((j for j in self.jobs if j["id"] == self.current), None)
        if job:
            self.interrupt(job, "pause")
            deadline = time.time() + 90
            while self.current and time.time() < deadline:
                time.sleep(0.2)
        self.save()

    # ------------------------------------------------------------------ scheduling
    def schedule(self) -> None:
        while not self.closing:
            with self.lock:
                if self.current is None:
                    queued = sorted((j for j in self.jobs if j["state"] == "queued"), key=lambda j: j["queued_at"])
                    if queued:
                        job = queued[0]
                        self.current = job["id"]
                        job.update(state="running", updated=time.time())
                        self.save()
                        threading.Thread(target=self.execute, args=(job,), name=f"job-{job['id']}", daemon=True).start()
            time.sleep(0.4)

    def execute(self, job: dict) -> None:
        self.tails[job["id"]] = deque(maxlen=60)
        try:
            if job["kind"] == "build":
                self.run_step(job, "build", self.step_build)
            elif job["kind"] == "eval":
                self.run_step(job, "evaluate", self.step_evaluate)
            else:
                start = PIPELINE_STEPS.index(job["step"]) if job["step"] in PIPELINE_STEPS else 0
                for name in PIPELINE_STEPS[start:]:
                    if job["steps"][name]["state"] in ("done", "skipped") and name != "check":
                        continue
                    self.run_step(job, name, getattr(self, f"step_{name}"))
            with self.lock:
                job.update(state="done", step=None, updated=time.time())
        except Paused as p:
            with self.lock:
                job.update(state="paused", pause={"reason": p.reason, "detail": p.detail}, updated=time.time())
        except Interrupted:
            with self.lock:
                why = self.interrupts.get(job["id"], "pause")
                if why == "decide":
                    try:
                        pairs = self.report_from_journal(job)
                        job["steps"]["evaluate"].update(state="done", ended=time.time(),
                                                        note=f"stopped by hand at {pairs:,} pairs ({job['info']['eval_file']})")
                        job.update(state="done", step=None, pause=None, updated=time.time())
                    except (OSError, ValueError) as e:
                        job.update(state="failed", error=f"the test could not be decided from its journal: {e}", updated=time.time())
                elif why == "cancel":
                    job.update(state="cancelled", updated=time.time())
                else:
                    job.update(state="paused", updated=time.time(),
                               pause={"reason": "user", "detail": "Paused during " + STEP_NOUNS.get(job["step"], job["step"])
                                      + ". Continue picks it up where it stopped."})
        except StepFailed as e:
            with self.lock:
                job.update(state="failed", error=str(e), updated=time.time())
        except Exception as e:  # a bug here must not take the scheduler down
            with self.lock:
                job.update(state="failed", error=f"{type(e).__name__}: {e}", updated=time.time())
            self.tails[job["id"]].append(traceback.format_exc())
        finally:
            with self.lock:
                self.interrupts.pop(job["id"], None)
                self.proc = None
                self.current = None
                self.save()
        if job["state"] == "done" and not self.closing:
            try:
                self.after(job)
            except Exception as e:
                with self.lock:
                    self.auto.update(enabled=False, message=f"The automatic mode stopped on an error: {e}")
                    self.save()

    def run_step(self, job: dict, name: str, fn) -> None:
        step = job["steps"][name]
        with self.lock:
            job["step"] = name
            step.update(state="running", started=step["started"] or time.time(), note=step["note"] if step["state"] == "paused" else "")
            self.save()
        try:
            note = fn(job)
        except Paused:
            with self.lock:
                step["state"] = "paused"
            raise
        except Interrupted:
            with self.lock:
                step["state"] = "paused"
            raise
        except Exception:
            with self.lock:
                step.update(state="failed", ended=time.time())
            raise
        with self.lock:
            skipped = isinstance(note, tuple)
            step.update(state="skipped" if skipped else "done", ended=time.time(), note=note[0] if skipped else (note or ""))
            self.save()

    # ------------------------------------------------------------------ processes
    def command(self, job: dict, step: str, cmd: list[str], stdout=None) -> str:
        """Run one command from the workspace root; its output goes to the step's log and the job's tail. `stdout`, when
        given, receives stdout line by line instead (the evaluation stream) and stderr alone goes to the log."""
        if self.interrupts.get(job["id"]):
            raise Interrupted()
        shown = " ".join(c if re.fullmatch(r"[\w@%+=:,./-]+", c) else json.dumps(c) for c in cmd)
        with self.lock:
            job["commands"].append({"step": step, "command": shown, "at": time.time()})
            self.save()
        log = self.log_path(job, step)
        tail = self.tails.setdefault(job["id"], deque(maxlen=60))
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        with log.open("a", encoding="utf-8") as out:
            out.write(f"\n$ {shown}\n")
            out.flush()
            tail.append(f"$ {shown}")
            try:
                proc = subprocess.Popen(cmd, cwd=self.repo, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE if stdout else subprocess.STDOUT,
                                        text=True, errors="replace", bufsize=1, env=env, start_new_session=True)
            except OSError as e:
                raise StepFailed(f"{cmd[0]} could not start: {e}")
            with self.lock:
                self.proc = proc

            def pump(stream, into_stdout):
                for line in stream:
                    if into_stdout:
                        stdout(line)
                        continue
                    out.write(line)
                    out.flush()
                    tail.append(line.rstrip("\n")[:400])

            readers = []
            if stdout:
                readers.append(threading.Thread(target=pump, args=(proc.stderr, False), daemon=True))
                readers[-1].start()
                pump(proc.stdout, True)
            else:
                pump(proc.stdout, False)
            code = proc.wait()
            for r in readers:
                r.join(timeout=5)
        if self.interrupts.get(job["id"]):
            raise Interrupted()
        if code != 0:
            last = next((l for l in reversed(tail) if l.strip() and not l.startswith("$ ")), "")
            raise StepFailed(f"{step} failed (exit {code}): {last[:300]}")
        return shown

    def probe(self, *args: str) -> dict:
        try:
            done = subprocess.run([self.python, str(self.probe_file), *args], cwd=self.repo, capture_output=True, text=True,
                                  timeout=900)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise StepFailed(f"the training interpreter {self.python} could not be run: {e}")
        lines = [l for l in done.stdout.splitlines() if l.strip()]
        if done.returncode != 0 or not lines:
            err = (done.stderr.strip().splitlines() or ["no output"])[-1]
            if "No module named" in err:
                err += f" (install the nnue package for {self.python}: python3 -m pip install -e 'models/nnue/py[dev]')"
            raise StepFailed(err)
        return json.loads(lines[-1])

    # ------------------------------------------------------------------ steps
    def step_build(self, job: dict):
        if not shutil.which("cargo"):
            raise StepFailed("cargo is not on the PATH; install Rust with rustup")
        self.command(job, "build", ["cargo", "build", "--release", "-p", "cli"])
        return "target/release/bot built"

    def own_set(self, s: dict) -> str:
        return f"selfplay_{s['name']}" + ("_quiet" if s.get("quiet") else "")

    def step_check(self, job: dict):
        s = job["settings"]
        if not (self.repo / self.bot).is_file():
            raise StepFailed(f"{self.bot} is missing: build it first (the Build button, or cargo build --release -p cli)")
        if not (self.repo / s["start"]).is_file():
            raise StepFailed(f"no such network: {s['start']}")
        net = self.probe("net", s["start"])
        caps = self.probe("caps")
        version = 9 if str(s["version"]) == "9" or (s["version"] == "auto" and net["version"] == 9) else 8
        if net["version"] == 9 and version != 9:
            raise StepFailed(f"{s['start']} is format 9 and trains only as format 9")
        if version == 9 and not caps["version9"]:
            raise StepFailed("format 9 needs a newer trainer (git pull, then reinstall models/nnue/py)")
        if s.get("quiet") and not caps["quiet"]:
            raise StepFailed("the quiet filter needs a newer importer (git pull)")
        if s["ema"] != "off" and not caps["ema"]:
            s["ema"] = "off"
        missing = [n for n, v in self.probe("sets", *[e["set"] for e in s["extra"]]).items() if not v["exists"]] if s["extra"] else []
        if missing:
            raise StepFailed("earlier batches not found under runs/nnue_data: " + ", ".join(missing))
        foreign = self.runs / s["name"]
        if foreign.is_dir() and any(foreign.iterdir()) and not (foreign / "config.json").is_file():
            raise StepFailed(f"runs/{s['name']} exists but is not a training run; choose another name")
        manifest = analysis.read_json(self.runs / "nnue_selfplay" / s["name"] / "manifest.json")
        seed = s["seed"]
        if manifest:
            if manifest.get("model_sha256") != net["sha256"] and not manifest.get("complete"):
                raise StepFailed(f"runs/nnue_selfplay/{s['name']} is an unfinished batch from another network; use a new name")
            if seed is None:
                seed = (manifest.get("settings") or {}).get("seed")
        if seed is None:
            seed = posix_cksum(s["name"].encode()) % 1000000 + 1
        with self.lock:
            job["info"] = {"net": net, "caps": caps, "version": version, "seed": int(seed),
                           "manifest": (manifest or {}).get("settings")}
        return f"H{net['hidden']}, format {net['version']}, seed {seed}"

    def step_generate(self, job: dict):
        s, info = job["settings"], job["info"]
        if s["games"] == 0:
            return ("no new games: training on earlier batches",)
        folder = self.runs / "nnue_selfplay" / s["name"]
        manifest = analysis.read_json(folder / "manifest.json")
        if manifest and manifest.get("complete"):
            return (f"{sum(sh['counts']['games'] for sh in manifest.get('shards', [])):,} games already played",)
        m = (manifest or {}).get("settings")
        if m:  # a batch on disk keeps its own settings, or it cannot resume
            games, nodes, threads, hash_mib, seed = m["games"], m["nodes"], m["threads"], m["hash_mib"], m["seed"]
            mpv = ["--multipv", str(m.get("multipv", 1)), "--multipv-margin", str(m.get("multipv_margin", 60)),
                   "--multipv-prob", str(m.get("multipv_prob", 50)), "--multipv-nodes", str(m.get("multipv_nodes", 25)),
                   "--pv-labels", str(m.get("pv_labels", 0))]
        else:
            games, nodes, threads, hash_mib, seed = s["games"], s["nodes"], s["threads"], s["hash"], info["seed"]
            mpv = ["--multipv", "2", "--multipv-margin", "30", "--multipv-prob", "50", "--multipv-nodes", "25"] \
                if s["multipv"] == 2 else ["--multipv", "1"]
        cmd = [self.bot, "selfplay", "--player", f"nnue:{s['start']}?hash={hash_mib}", "--nodes", str(nodes), "--games",
               str(games), "--threads", str(threads), "--seed", str(seed), "--opening-plies", "8", "--random-moves", "2",
               "--random-from", "8", "--random-to", "40", *mpv, "--records", f"runs/nnue_selfplay/{s['name']}"]
        if manifest:
            cmd.append("--resume")
        self.command(job, "generate", cmd)
        manifest = analysis.read_json(folder / "manifest.json") or {}
        if not manifest.get("complete"):
            raise StepFailed("the generator stopped before the batch was complete; continue to resume it")
        return f"{games:,} games at {nodes:,} nodes"

    def step_import(self, job: dict):
        s = job["settings"]
        if s["games"] == 0:
            return ("nothing to import",)
        own = self.own_set(s)
        if (self.runs / "nnue_data" / own / "provenance.json").is_file():
            return ("already imported",)
        shutil.rmtree(self.runs / "nnue_data" / f"{own}.tmp", ignore_errors=True)
        cmd = [self.python, "-m", "nnue.importer", "selfplay", "--records", f"runs/nnue_selfplay/{s['name']}",
               "--out", own, "--min-ply", "16"]
        if s.get("quiet"):
            cmd.append("--quiet")
        self.command(job, "import", cmd)
        rows = (analysis.read_json(self.runs / "nnue_data" / own / "provenance.json") or {}).get("rows", 0)
        return f"{rows:,} rows"

    def mixture(self, s: dict) -> list[tuple[str, float]]:
        parts = [(self.own_set(s), 1.0)] if s["games"] > 0 else []
        return parts + [(e["set"], e["share"]) for e in s["extra"]]

    def step_encode(self, job: dict):
        sets = self.probe("sets", *[n for n, _ in self.mixture(job["settings"])])
        todo = [n for n, v in sets.items() if not v["cache_ok"]]
        if not todo:
            return ("every batch already has its feature-id cache",)
        for name in todo:
            self.command(job, "encode", [self.python, "-m", "nnue.data", "encode", name])
        return f"encoded {', '.join(t.removeprefix('selfplay_') for t in todo)}"

    def train_args(self, job: dict) -> tuple[list[str], dict]:
        s, info = job["settings"], job["info"]
        parts = self.mixture(s)
        sets = self.probe("sets", *[n for n, _ in parts])
        rows = sum(sets[n]["rows"] for n, _ in parts)
        batch = self.batch_plan(rows, s)
        s = {**s, "batch": batch}
        epochs = self.epoch_plan(rows, s)
        steps = int(s["steps"]) if s.get("steps") else max(10, math.ceil(rows * float(s["passes"]) / (batch * epochs)))
        warmup = int(s["warmup"]) if s.get("warmup") else max(1, min(200, steps * epochs // 10))
        if s["ema"] == "auto":
            ema = f"{1 - 1 / steps:.6f}" if steps >= 2 else None
        elif s["ema"] == "off":
            ema = None
        else:
            ema = f"{float(s['ema']):g}"
        args = ["--run", s["name"], "--init", s["start"]]
        for name, share in parts:
            args += ["--data", f"{name}:{share:g}"]
        args += ["--hidden", str(info["net"]["hidden"]), "--version", str(info["version"]), "--batch", str(s["batch"]),
                 "--steps_per_epoch", str(steps), "--epochs", str(epochs), "--lr", f"{s['lr']:g}",
                 "--weight_decay", "1e-5", "--warmup_steps", str(warmup), "--lr_floor", "0.15", "--qat_start_epoch", "1",
                 "--raw_weight", "0.02", "--symmetry_weight", "0.2", "--grad_clip", "5", "--patience", str(s.get("patience") or epochs),
                 "--val_rows", "8192", "--threads", str(s["threads"]), "--device", s["device"], "--seed", str(info["seed"])]
        if ema:
            args += ["--ema", ema]
        return args, {"rows": rows, "steps": steps, "warmup": warmup, "ema": ema, "epochs": epochs, "batch": batch}

    @staticmethod
    def batch_plan(rows: int, s: dict) -> int:
        """The training batch. The passes set how often each row is seen; the batch sets how many optimiser updates
        that is. A 3,200-game batch at 8,192 is about 340 updates, few enough that the network barely leaves its start.
        `auto` halves the batch (8,192, 4,096, 2,048) until training makes at least 1,000 updates, keeping the passes,
        the learning rate and so the risk of memorising where they were. A large mixture keeps 8,192. Experimental: in
        a direct match a first batch retrained this way (2,048, about 1,350 updates) played about 18 Elo weaker than at
        8,192, likely because more updates pull a small continuation further from its start; 8,192 is the default."""
        if s["batch"] != "auto":
            return int(s["batch"])
        samples = rows * float(s["passes"])
        batch = 8192
        while batch > 2048 and samples / batch < 1000:
            batch //= 2
        return batch

    @staticmethod
    def epoch_plan(rows: int, s: dict) -> int:
        """Epochs are how often the trainer validates, checkpoints and exports best.nnue; the amount of training is
        the passes over the rows. `auto` keeps an epoch at least 100 steps (fewer launches and a steadier validation
        on a small batch) and at most 20 epochs (the research loop's count, 1,000 steps each on its large mixtures)."""
        if s["epochs"] != "auto":
            return int(s["epochs"])
        total = math.ceil(rows * float(s["passes"]) / s["batch"])
        return max(4, min(20, total // 100))

    def step_train(self, job: dict):
        s = job["settings"]
        out = self.runs / s["name"]
        if s.get("end_training_early") and (out / "best.nnue").is_file():
            done = self.epochs_done(out)
            return (f"ended by hand after epoch {done}",)
        if "net" not in job["info"]:
            self.step_check(job)
        args, sizing = self.train_args(job)
        with self.lock:
            job["info"]["sizing"] = sizing
        state = self.probe("train-state", *args)
        if state["state"] == "foreign":
            raise StepFailed(f"runs/{s['name']} exists but is not a training run")
        if state["state"] == "mismatch":
            if not s.get("restart_training"):
                raise StepFailed(f"runs/{s['name']} was trained with other settings ({state.get('why', '')}). "
                                 "Restart training moves the old run aside and trains again on the same games.")
            aside = self.runs / f"{s['name']}.replaced_{time.strftime('%Y%m%d_%H%M%S')}"
            (self.runs / s["name"]).rename(aside)
            with self.lock:
                s["restart_training"] = False
                job["steps"]["train"]["note"] = f"the old run moved to runs/{aside.name}"
            state = {"state": "fresh", "epochs_done": 0}
        done = state["epochs_done"] if state["state"] == "resume" else 0
        epochs = sizing["epochs"]
        # One epoch per launch (--stop_epoch, then --resume): the trainer checkpoints at every epoch's end, so the job can
        # pause after any epoch without losing work, and the schedule and random streams continue exactly.
        while done < epochs:
            cmd = [self.python, "-m", "nnue.train", *args]
            if done > 0 or state["state"] == "resume":
                cmd += ["--resume", f"runs/{s['name']}/latest.pt"]
            cmd += ["--stop_epoch", str(done + 1)]
            self.command(job, "train", cmd)
            now = self.epochs_done(out)
            if now <= done:
                raise StepFailed(f"the trainer finished without writing epoch {done + 1}")
            done = now
            if s.get("patience") and done < epochs and self.stale_epochs(out) >= s["patience"]:
                return f"stopped early after epoch {done} of {epochs}: {s['patience']} epochs without a better objective"
            with self.lock:
                wants = s.get("pause_every_epoch") or job.pop("pause_after_epoch", False)
            if wants and done < epochs:
                raise Paused("epoch", f"Paused after epoch {done} of {epochs}; best.nnue holds the best epoch so far.")
        return f"{done} epochs of {sizing['steps']} steps over {sizing['rows']:,} rows"

    @staticmethod
    def stale_epochs(out: Path) -> int:
        log = analysis.read_log(out / "log.csv")
        rows = (log or {}).get("rows") or []
        values = [r.get("objective") for r in rows]
        best = min(range(len(values)), key=lambda i: values[i] if isinstance(values[i], (int, float)) else float("inf"), default=0)
        return len(rows) - 1 - best if rows else 0

    @staticmethod
    def epochs_done(out: Path) -> int:
        try:
            with (out / "log.csv").open() as f:
                return max(0, sum(1 for _ in f) - 1)
        except OSError:
            return 0

    def step_evaluate(self, job: dict):
        e = job["settings"]["eval"]
        run = job["run"]
        out = self.runs / run
        best = out / "best.nnue"
        if not best.is_file():
            raise StepFailed(f"runs/{run}/best.nnue does not exist yet")
        config = (analysis.read_json(out / "config.json") or {}).get("config") or {}
        reference = e.get("reference") or job["settings"].get("start") or config.get("init")
        anchor = bool(e.get("reference")) and e["reference"] != config.get("init")
        if not reference:
            raise StepFailed("the run's starting network is unknown (no config.json)")
        hash_mib = job["settings"].get("hash") or e.get("hash") or 64
        threads = int(job["settings"].get("threads") or e.get("threads") or self.threads)
        seed = (job.get("info") or {}).get("seed") or config.get("seed") or 1
        sha = sha256_file(best)[:10]
        k = f"{e['sims'] * 2500 // 1000}k"
        epochs = config.get("epochs") or 0
        done = self.epochs_done(out)
        early = f"e{done}_" if epochs and done < epochs else ""
        versus = ""
        if anchor:  # measured against another network: kept apart from the decision (analysis reads eval_vs_*)
            slug = re.sub(r"[^A-Za-z0-9]+", "_", Path(reference).parent.name if Path(reference).name == "best.nnue" else Path(reference).stem)
            versus = f"vs_{slug}_{sha256_file(self.repo / reference)[:6]}_"
        if e["mode"] == "sprt":
            threads = min(threads, analysis.PLAN["eval_threads_max"])
            target, cap = f"{e['target']:g}", str(e["cap"])
            journal = f"runs/{run}/sprt_{versus}{sha}_{k}_{target}_{cap}.jsonl"
            tag = ("" if abs(e["target"] - 0.52) < 1e-9 else f"_t{round(e['target'] * 1000):03d}") + ("" if e["cap"] == 3008 else f"_c{e['cap']}")
            file = f"eval_{versus}{early}sprt_{k}{tag}.json"  # a retest at another target never overwrites a result
            with self.lock:
                job["info"].update(journal=journal, eval_file=file, candidate=f"nnue:runs/{run}/best.nnue?hash={hash_mib}",
                                   reference=f"nnue:{reference}?hash={hash_mib}", sims=e["sims"])
        else:
            file = f"eval_{versus}{early}{e['pairs']}p_{k}.json"
        cmd = [self.bot, "eval", "--candidate", f"nnue:runs/{run}/best.nnue?hash={hash_mib}",
               "--reference", f"nnue:{reference}?hash={hash_mib}", "--sims", str(e["sims"]), "--threads", str(threads),
               "--seed", str(seed)]
        if e["mode"] == "sprt":
            cmd += ["--sprt", journal]
            if abs(e["target"] - 0.52) > 1e-9:
                cmd += ["--sprt-target", target]
            if e["cap"] != 3008:
                cmd += ["--sprt-cap", cap]
            if (self.repo / journal).is_file():
                cmd.append("--resume")
        else:
            cmd += ["--pairs", str(e["pairs"])]
        cmd.append("--stream")
        stream = self.log_path(job, "evaluate").with_suffix(".jsonl")
        report: dict = {}
        self.eval_games[job["id"]] = 0
        with stream.open("a", encoding="utf-8") as sink:
            def on_line(line: str) -> None:
                if re.search(r'"event":\s*"move"', line):
                    return
                sink.write(line)
                sink.flush()
                if re.search(r'"event":\s*"end"', line):
                    self.eval_games[job["id"]] = self.eval_games.get(job["id"], 0) + 1
                elif re.search(r'"event":\s*"report"', line):
                    report.update(json.loads(line))

            self.command(job, "evaluate", cmd, stdout=on_line)
        if not report:
            raise StepFailed("the evaluation ended without a report")
        report.pop("event", None)
        tmp = out / (file + ".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(out / file)
        games = report["wins"] + report["draws"] + report["losses"]
        score = (report["wins"] + report["draws"] / 2) / games if games else 0.5
        with self.lock:
            job["result"]["eval_file"] = file
        stop = (report.get("sequential") or {}).get("stop_reason")
        return f"score {score:.3f} over {games:,} games" + (f", test {stop}" if stop else "") + f" ({file})"

    def report_from_journal(self, job: dict) -> int:
        """The evaluation's report from the pairs its journal holds, for a test stopped by hand: the same fields as
        bot eval's report, with the stop recorded as `stopped` so the decision reads it like a test stopped at its cap."""
        info = job["info"]
        reader = JournalReader(self.repo / info["journal"])
        seq = reader.read()
        if not seq or not seq["pairs"]:
            raise ValueError("no finished pairs")
        n = seq["pairs"]
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        mean = sum(c * v for c, v in zip(seq["counts"], values)) / n
        sd = math.sqrt(sum(c * (v - mean) ** 2 for c, v in zip(seq["counts"], values)) / max(1, n - 1))
        margin, half = 2 * mean - 1, 1.96 * 2 * sd / math.sqrt(n)
        report = {
            "candidate": info["candidate"], "reference": info["reference"], "pairs": n, "complete_pairs": n,
            "sims": info["sims"], "reference_sims": info["sims"], "move_ms": None, "reference_move_ms": None, "forfeits": 0,
            "wins": reader.wdl[0], "draws": reader.wdl[1], "losses": reader.wdl[2], "margin": margin,
            "interval": [margin - half, margin + half], "bootstrap_interval": [margin - half, margin + half],
            "mean_plies": reader.plies / (2 * n),
            "sequential": {"protocol": {"test": {"s0": seq["s0"], "s1": seq["s1"], "cap": seq["cap"], "batch": 16, "first_check": 128}},
                           "counts": seq["counts"], "llr": seq["llr"], "stop_reason": "stopped", "error": None,
                           "intervals_descriptive_only": True},
        }
        out = self.runs / job["run"] / info["eval_file"]
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(out)
        return n

    # ------------------------------------------------------------------ the automatic mode
    def after(self, job: dict) -> None:
        if not self.auto["enabled"] or job["kind"] not in ("pipeline", "eval") or not job["run"]:
            return
        if self.busy():
            return
        self.follow(job["run"])

    def follow(self, run_name: str) -> dict | None:
        data = analysis.load_runs(self.runs, self.plan)
        run = next((r for r in data["runs"] if r["name"] == run_name), None)
        if not run:
            return self.stop_auto(f"{run_name} is not in the runs directory")
        d = run["decision"]
        nxt = d.get("next")
        if not nxt:
            return self.stop_auto(f"{run_name}: {d['title'].lower()}; nothing to do automatically")
        bad = [c for c in run["checks"] if c["status"] == "bad" and c["group"] in ("Training", "Data")]
        if self.auto["pause_on_problems"] and bad and d["verdict"] != "reeval":
            return self.stop_auto(f"{run_name} has training or data problems ({', '.join(c['label'] for c in bad)}); "
                                  "look at them, then start the automatic mode again")
        with self.lock:
            if d["verdict"] == "adopt":
                if self.auto["adopted"] >= self.auto["max_generations"]:
                    n = self.auto["max_generations"]
                    return self.stop_auto(f"{n} new generation{'s' if n != 1 else ''} done, the limit; {run_name} is adopted and waits")
                self.auto["adopted"] += 1
            if nxt["kind"] == "pipeline":
                if self.auto["batches"] >= self.auto["max_batches"]:
                    n = self.auto["max_batches"]
                    return self.stop_auto(f"{n} batch{'es' if n != 1 else ''} played, the limit")
                self.auto["batches"] += 1
        try:
            if nxt["kind"] == "pipeline":
                settings = {**nxt["settings"], "threads": self.threads, "pause_every_epoch": self.auto["pause_every_epoch"]}
                job = self.create_pipeline(settings, auto=True)
            elif nxt["kind"] == "eval":
                job = self.create_eval(nxt["run"], nxt["settings"], auto=True)
            else:
                paused = next((j for j in reversed(self.jobs) if j["run"] == run_name and j["state"] in ("paused", "failed")), None)
                if not paused:
                    return self.stop_auto(f"{run_name} stopped outside the dashboard; continue it by hand")
                job = self.action(paused["id"], "continue")
        except ValueError as e:
            return self.stop_auto(str(e))
        with self.lock:
            self.auto["message"] = f"{d['title']}: queued {job['kind']} {job['run']}"
            self.save()
        return job

    def stop_auto(self, message: str) -> None:
        with self.lock:
            self.auto.update(enabled=False, message="Auto stopped: " + message + ("" if message.endswith(".") else "."))
            self.save()
        return None

    # ------------------------------------------------------------------ reading
    def progress(self, job: dict) -> dict:
        step = job["step"]
        s = job["settings"]
        out = {"step": step, "words": STEP_WORDS.get(step, step or ""), "done": None, "total": None, "unit": "",
               "eta_seconds": None}
        run = job["run"]
        if step == "generate" and run:
            gen = analysis.generation_summary(self.runs, run)
            if gen:
                out.update(done=gen["games_done"], total=gen["games_target"], unit="games", in_flight=gen["in_flight"])
                if gen["games_per_hour"] and gen["games_target"] and job["state"] == "running":
                    out["eta_seconds"] = (gen["games_target"] - gen["games_done"]) / gen["games_per_hour"] * 3600
        elif step == "train" and run:
            log = analysis.read_log(self.runs / run / "log.csv")
            rows = (log or {}).get("rows") or []
            total = ((job.get("info") or {}).get("sizing") or {}).get("epochs") or (s.get("epochs") if s.get("epochs") != "auto" else None)
            out.update(done=len(rows), total=total, unit="epochs")
            secs = [r.get("seconds") for r in rows if isinstance(r.get("seconds"), (int, float))]
            if secs and total and job["state"] == "running":
                out["eta_seconds"] = (total - len(rows)) * sum(secs) / len(secs)
        elif step == "evaluate":
            e = s["eval"]
            out.update(done=self.eval_games.get(job["id"], 0), unit="games",
                       total=None if e["mode"] == "sprt" else 2 * e["pairs"], cap=2 * e["cap"] if e["mode"] == "sprt" else None)
            journal = (job.get("info") or {}).get("journal")
            if e["mode"] == "sprt" and journal:
                reader = self.journals.get(job["id"])
                if reader is None or reader.path != self.repo / journal:
                    reader = self.journals[job["id"]] = JournalReader(self.repo / journal)
                seq = reader.read()
                if seq:
                    # the bar is how close the test is to stopping: a bound or the cap, whichever is nearer
                    seq["decide_from"] = analysis.PLAN["decision_pairs"]
                    out.update(sequential=seq, done=seq["pairs"], unit="pairs", total=None,
                               eta_seconds=seq["eta_seconds"] if job["state"] == "running" else None)
        return out

    def snapshot(self) -> dict:
        with self.lock:
            jobs = []
            for job in self.jobs[-60:]:
                view = {k: v for k, v in job.items() if k != "commands"}
                view["commands"] = job["commands"][-12:]
                view["progress"] = self.progress(job) if job["state"] in ("running", "paused") else None
                view["tail"] = list(self.tails.get(job["id"], []))[-40:] if job["state"] in ("running", "paused", "failed") else []
                if not view["tail"] and job["state"] in ("paused", "failed") and job["step"]:
                    log = self.home / "logs" / job["id"] / f"{job['step']}.log"
                    try:
                        view["tail"] = [l[:400] for l in log.read_text(errors="replace").splitlines()[-40:]]
                    except OSError:
                        pass
                jobs.append(view)
            return {"jobs": jobs, "auto": dict(self.auto), "current": self.current, "version": self.version,
                    "defaults": default_settings(self.threads), "bot_exists": (self.repo / self.bot).is_file(),
                    "python": self.python, "repo": str(self.repo)}

    def overlay(self, runs: list[dict]) -> None:
        """Replace the file-based guess of a run's state with what the runner knows."""
        with self.lock:
            latest = {}
            for job in self.jobs:
                if job["run"] and job["state"] not in TERMINAL:
                    latest[job["run"]] = job
        for run in runs:
            job = latest.get(run["name"])
            if not job:
                run["status"]["source"] = "files"
                continue
            p = self.progress(job)
            fraction = p["done"] / p["total"] if p.get("total") else None
            state = {"running": "running", "queued": "queued", "paused": "paused"}[job["state"]]
            detail = p["words"]
            if p.get("total"):
                detail += f": {p['done']:,} of {p['total']:,} {p['unit']}"
            elif p.get("done"):
                detail += f": {p['done']:,} {p['unit']}"
            if job["state"] == "paused" and job["pause"]:
                detail = job["pause"]["detail"]
            run["status"] = {"state": state, "step": job["step"], "fraction": fraction, "eta_seconds": p["eta_seconds"],
                             "detail": detail, "source": "runner", "job": job["id"]}
            if job["state"] in ("running", "queued") or (job["state"] == "paused" and job["step"] != "evaluate"):
                run["decision"] = {**run["decision"], "verdict": "running",
                                   "title": {"running": STEP_WORDS.get(job["step"], "Running"), "queued": "Queued",
                                             "paused": "Paused"}[job["state"]],
                                   "summary": detail.capitalize() + ("" if detail.endswith(".") else "."),
                                   "reasons": ["The decision waits for this job's evaluation."], "next": None}
