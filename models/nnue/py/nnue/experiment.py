"""One preregistered experiment from one file: trains its arms, exports their
endpoints, plays its matches through `bot eval` and writes `verdict.json`.
Every step is skipped when its artifact exists and is bound to the manifest,
so a relaunch resumes. Heavy phases run one at a time at below-normal
priority on the declared logical CPUs, under the workspace's compute lock.

    python -m nnue.experiment pin <experiment.json>
    python -m nnue.experiment run <experiment.json> [--only train|match]
    python -m nnue.experiment verdict <experiment.json>
    python -m nnue.experiment lock status | acquire <owner> <phase> [<mask>] | release <owner> [<pid>]

The file (schema 2) sits in its experiment directory, where the match
reports, journals, timings and `verdict.json` are written:

    {"schema": 2, "name": "long_training",
     "bot": "runs/nnue_bins/accepted16.exe", "network": "runs/nnue_candidates/mb_b.nnue",
     "affinity": "16-31",
     "identities": {"runs/nnue_bins/accepted16.exe": "<sha256>", ...},
     "arms": [{"run": "long60_s2",
               "train": {"init": "runs/nnue_candidates/mb_b.nnue",
                         "config": {"epochs": 60, "seed": 2, ...},
                         "data": [["selfplay_gen3", 0.347], ...]},
               "stops": [20],
               "endpoints": {"epoch20": "latest@20", "epoch60": "latest", "best": "best"}}],
     "matches": [{"tag": "s2_60_vs_20_50", "candidate": "long60_s2:epoch60",
                  "reference": "long60_s2:epoch20", "move_ms": 50, "pairs": 200,
                  "seed": 2026091521, "player_threads": 1, "concurrent": 8},
                 {"tag": "qtt_50", "candidate": "runs/nnue_bins/qtt.exe",
                  "reference": "runs/nnue_bins/base.exe", "move_ms": 50,
                  "sprt": true, "seed": 2026091601}],
     "rules": [{"name": "promotion", "type": "lower_bound_above", "threshold": 0.5,
                "matches": ["s2_60_vs_mb_b_50", "s2_60_vs_mb_b_100"]},
               {"name": "patch", "type": "sprt_accept", "matches": ["qtt_50"]}]}

`identities` pins every fixed input by SHA-256: the bot, the network, the
`.exe` and `.nnue` sides, the arms' `init` files and the provenance file of
every dataset. `pin` fills it from the files as they are when the experiment
is preregistered; `run` refuses a file that differs.

An arm without `train` names an existing run. `stops` are epoch counts at
which training pauses (`--stop_epoch`), the checkpoint is retained as
`<run>/epoch<N>.pt` and resumed exactly; the checkpoints' own epoch counts
are the authority, never the log. An endpoint `latest@N` is that
checkpoint's export, `latest` the final checkpoint's, `best` the trainer's
`best.nnue`. A match side is `<run>:<endpoint>` or a `.nnue` path, played by
the preregistered `bot`; sides that are `.exe` paths are search builds
(`rpsi:` seats with the match's player threads, both sides so the process
overhead is equal) playing the experiment's `network`. A match with
`"sprt": true` is the sequential test of `bot eval --sprt` (journal
`<tag>.jsonl`, resumed when it exists, stopped by its own rule, streamed so
that `<tag>.timing.json` records each invocation's pairs, wall time and the
batch-barrier idle share); every other match plays `pairs` openings and
records its games. A played match is bound to its manifest entry, the
affinity, the hashes of the files it used and of its own report and journal
or records (`<tag>.match.json`); a report whose binding differs is played
again and judged invalid. An existing endpoint is reused only when its
sidecar carries the arm's config, the epoch its source names and the file's
hash. Rule types: `lower_bound_above` and
`score_at_least` (fixed-count matches only: the paired bootstrap lower
bound above, or the score at least, the threshold in every named match) and
`sprt_accept` (sequential matches only: every named test accepted). The
verdict verifies the pinned inputs and records numbers, rule outcomes and
an audit of every match's recorded games reconciled with its report (the
journal's protocol, digest and provenance against the report, the manifest
and the pinned files; pairs, seats, openings, endings, the candidate's
results; how the games ended, the first mover's score, distinct openings,
mean plies) and, for a finished, bound and audited sequential test, C1's
own replay of the journal; a match whose records, binding or replay
disagree is invalid. Every `bot eval` runs in a job object that is
terminated when the invocation ends, however it ends; the runner keeps the
lock when a member of a job survives its termination.
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from ctypes import wintypes
from pathlib import Path

import psutil
import torch

from . import export, paths
from .model import NNUE
from .paths import data_dir, run_dir, sha256

LOCK = "runs/nnue_plan/compute_lock"
TERMINAL = ("accept", "reject", "inconclusive", "invalid")  # the stop reasons of `bot eval --sprt`
FIXED_RULES = ("lower_bound_above", "score_at_least")
SEQUENTIAL_RULES = ("sprt_accept",)


def workspace_root() -> Path:
    return paths.workspace_root()


def absolute(spec: str) -> Path:
    """A manifest path as an absolute path (lexically, so a junction stays the path the manifest names)."""
    return Path(os.path.abspath(workspace_root() / spec))


def relative(path: Path) -> str:
    """The manifest key of a file: its path from the workspace root."""
    return Path(os.path.relpath(path, workspace_root())).as_posix()


def cpus(mask: str) -> list[int]:
    out: list[int] = []
    for part in mask.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def nice(mask: str) -> None:
    """Below-normal priority and the affinity mask for this process and its children."""
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(cpus(mask))


# The compute lock: one heavy phase at a time; created atomically, never taken from a live owner. A lock
# whose owner died (a hard kill skips the release) is released by hand once no child of it survives.


def lock_path() -> Path:
    return workspace_root() / LOCK


def lock_read() -> dict | None:
    try:
        return json.loads(lock_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def lock_acquire(owner: str, phase: str, mask: str, pid: int | None = None) -> dict:
    record = {
        "owner": owner,
        "pid": os.getpid() if pid is None else pid,
        "phase": phase,
        "mask": mask,
        "since": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        fd = os.open(lock_path(), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        held = lock_read() or {}
        raise RuntimeError(
            f"compute lock held by {held.get('owner')} (pid {held.get('pid')}, "
            f"{'alive' if psutil.pid_exists(held.get('pid', -1)) else 'stale'}): {held.get('phase')}"
        ) from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(record, f)
    return record


def lock_release(owner: str, pid: int | None = None) -> dict:
    """Releases the lock of `owner` (and, when given, of exactly that pid)."""
    held = lock_read()
    if held is None or held["owner"] != owner or (pid is not None and held["pid"] != pid):
        raise RuntimeError(f"compute lock not held by {owner}{'' if pid is None else f' (pid {pid})'}: {held}")
    lock_path().unlink()
    return held


# Identities: every fixed input pinned by its hash.


def fixed_inputs(experiment: dict) -> list[Path]:
    """The files the experiment must find unchanged: the bot, the network, the
    `.exe`/`.nnue` sides, the arms' inits and every dataset's provenance."""
    files = [absolute(experiment["bot"])]
    if experiment.get("network"):
        files.append(absolute(experiment["network"]))
    for arm in experiment.get("arms", []):
        spec = arm.get("train", {})
        if spec.get("init"):
            files.append(absolute(spec["init"]))
        for name, _ in spec.get("data", []):
            files.append(data_dir(name) / "provenance.json")
    for match in experiment.get("matches", []):
        for role in ("candidate", "reference"):
            if match[role].endswith((".nnue", ".exe")):
                files.append(absolute(match[role]))
    return sorted(set(files), key=relative)


def identities(experiment: dict) -> dict[str, str]:
    return {relative(file): sha256(file) for file in fixed_inputs(experiment)}


def verify(experiment: dict, path: Path) -> Path:
    """`path` after checking it against its pinned identity."""
    key = relative(path)
    pinned = experiment.get("identities", {}).get(key)
    if pinned is None:
        raise RuntimeError(f"{key} is not pinned in the manifest (python -m nnue.experiment pin)")
    if sha256(path) != pinned:
        raise RuntimeError(f"{key} differs from its pinned identity")
    return path


# Training arms.


def checkpoint_epochs(path: Path) -> int:
    """Epochs completed by a checkpoint (its `epoch` is the last finished one)."""
    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"]) + 1


def epochs_done(run: str) -> int:
    latest = run_dir(run) / "latest.pt"
    return checkpoint_epochs(latest) if latest.is_file() else 0


def train(experiment: dict, arm: dict, stop: int | None) -> None:
    spec = arm["train"]
    command = [sys.executable, "-m", "nnue.train", "--run", arm["run"]]
    for key, value in spec.get("config", {}).items():
        command += [f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)]
    if spec.get("init"):
        command += ["--init", str(verify(experiment, absolute(spec["init"])))]
    for name, share in spec["data"]:
        command += ["--data", f"{name}:{share}"]
    latest = run_dir(arm["run"]) / "latest.pt"
    if latest.is_file():
        command += ["--resume", str(latest)]
    if stop is not None:
        command += ["--stop_epoch", str(stop)]
    logs = run_dir(arm["run"]).parent / "nnue_plan"
    logs.mkdir(parents=True, exist_ok=True)
    with (
        (logs / f"train_{arm['run']}.log").open("a", encoding="utf-8") as out,
        (logs / f"train_{arm['run']}.err").open("a", encoding="utf-8") as err,
    ):
        result = subprocess.run(command, cwd=workspace_root(), stdout=out, stderr=err, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"training {arm['run']} failed (runs/nnue_plan/train_{arm['run']}.err)")


def snapshot(checkpoint: Path, target: Path) -> dict:
    """Export a checkpoint exactly as the trainer would export it at that
    epoch, with the run's binding (data shares, dataset hashes, init hash)."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    model = NNUE(config["hidden"], config["version"])
    model.load_state_dict(saved["model"])
    metadata = {"checkpoint": str(checkpoint), "epoch": saved["epoch"], "step": saved["step"], "config": config}
    metadata.update({key: saved[key] for key in ("data", "datasets", "init_sha256")})
    return export.export(model, target, metadata)


def endpoint_file(run: str, name: str) -> Path:
    return run_dir(run) / ("best.nnue" if name == "best" else f"{name}.nnue")


def endpoint_problem(arm: dict, name: str) -> str | None:
    """Why an existing endpoint file cannot stand for this arm (None when it
    can): its sidecar must hash the file and carry the checkpoint epoch its
    source names and, for an arm the runner trains, the arm's config, data
    shares, dataset hashes and init hash. An endpoint without that binding is
    refused; there is no fallback."""
    file = endpoint_file(arm["run"], name)
    sidecar = file.with_suffix(".nnue.json")
    if not sidecar.is_file():
        return "no sidecar"
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    bound = ("data", "datasets", "init_sha256") if "train" in arm else ()  # an existing run has no spec to bind to
    for key in ("sha256", "epoch", "config", *bound):
        if key not in info:
            return f"the sidecar lacks {key}"
    if info["sha256"] != sha256(file):
        return "the file differs from its sidecar's hash"
    spec = arm.get("train", {})
    for key, value in spec.get("config", {}).items():
        if info["config"].get(key) != value:
            return f"config {key} is {info['config'].get(key)!r}, the arm says {value!r}"
    if "train" in arm:
        if [list(part) for part in info["data"]] != [list(part) for part in spec.get("data", [])]:
            return "trained on other data shares"
        expected = {name: sha256(data_dir(name) / "provenance.json") for name, _ in spec.get("data", [])}
        if info["datasets"] != expected:
            return "trained on other datasets"
        init = sha256(absolute(spec["init"])) if spec.get("init") else None
        if info["init_sha256"] != init:
            return "started from another init"
    source = arm.get("endpoints", {}).get(name, "")
    total = int(spec.get("config", {}).get("epochs", 20))
    epoch = info["epoch"]
    if source.startswith("latest@") and epoch != int(source[7:]) - 1:
        return f"epoch {epoch}, the endpoint needs {int(source[7:]) - 1}"
    if source == "latest" and "train" in arm and epoch != total - 1:
        return f"epoch {epoch}, the final checkpoint needs {total - 1}"
    if source == "best" and not (isinstance(epoch, int) and 0 <= epoch < total):
        return f"epoch {epoch} is outside the arm's {total} epochs"
    return None


def run_arm(experiment: dict, arm: dict) -> None:
    run, endpoints = arm["run"], arm.get("endpoints", {})
    for name in endpoints:
        if endpoint_file(run, name).is_file() and (problem := endpoint_problem(arm, name)):
            raise RuntimeError(f"{run}: endpoint {name} cannot stand for this arm ({problem})")
    if all(endpoint_file(run, name).is_file() for name in endpoints):
        return
    if "train" in arm:
        total = int(arm["train"]["config"].get("epochs", 20))
        for stop in sorted(arm.get("stops", [])):
            retained = run_dir(run) / f"epoch{stop}.pt"
            if not retained.is_file():
                if epochs_done(run) < stop:
                    train(experiment, arm, stop)
                if epochs_done(run) != stop:
                    raise RuntimeError(f"{run}: {epochs_done(run)} epochs where the retained checkpoint needs {stop}")
                shutil.copy2(run_dir(run) / "latest.pt", retained)
            if checkpoint_epochs(retained) != stop:
                raise RuntimeError(f"{run}: epoch{stop}.pt holds {checkpoint_epochs(retained)} epochs, not {stop}")
        if epochs_done(run) < total:
            train(experiment, arm, None)
        if epochs_done(run) != total:
            raise RuntimeError(f"{run}: {epochs_done(run)} of {total} epochs")
    for name, source in endpoints.items():
        target = endpoint_file(run, name)
        if target.is_file():
            continue
        if source == "best":
            raise RuntimeError(f"{run}: best.nnue missing")
        if source == "latest":
            snapshot(run_dir(run) / "latest.pt", target)
        elif source.startswith("latest@"):
            snapshot(run_dir(run) / f"epoch{int(source[7:])}.pt", target)
        else:
            raise ValueError(f"{run}: unknown endpoint source {source!r}")


# Matches.


def side(spec: str) -> Path:
    if spec.endswith((".nnue", ".exe")):
        return absolute(spec)
    run, _, endpoint = spec.partition(":")
    return endpoint_file(run, endpoint)


def seat(experiment: dict, match: dict, role: str) -> list[str]:
    """The `bot eval` arguments of one side: a network under the preregistered
    search, or a search build (`.exe`) playing the experiment's `network` with
    the match's player threads. The child's command is split on whitespace by
    `bot eval`, so paths with whitespace are refused; a sequential test also
    names the network so the journal hashes the shared weights."""
    file = side(match[role])
    if file.suffix != ".exe":
        return [f"--{role}", f"nnue:{file}"]
    network = verify(experiment, absolute(experiment["network"]))
    if any(ch.isspace() for ch in f"{file}{network}"):
        raise ValueError(f"{match['tag']}: an rpsi seat cannot hold a path with whitespace ({file}, {network})")
    out = [f"--{role}", f"rpsi:{file} rpsi --player nnue:{network} --threads {match.get('player_threads', 1)}"]
    if match.get("sprt"):
        out += [f"--{role}-network", str(network)]
    return out


def command_of(experiment: dict, directory: Path, match: dict) -> list[str]:
    command = [
        str(absolute(experiment["bot"])),
        "eval",
        *seat(experiment, match, "candidate"),
        *seat(experiment, match, "reference"),
        "--move-ms",
        str(match["move_ms"]),
        "--threads",
        str(match.get("concurrent", 8)),
        "--player-threads",
        str(match.get("player_threads", 1)),
        "--seed",
        str(match["seed"]),
        "--opening-plies",
        str(match.get("opening_plies", 8)),
    ]
    if match.get("sprt"):
        journal = directory / f"{match['tag']}.jsonl"
        command += ["--sprt", str(journal), *(["--resume"] if journal.is_file() else []), "--stream"]
    else:
        command += ["--pairs", str(match["pairs"]), "--records", str(directory / f"{match['tag']}.games.jsonl")]
    return command


def finished(match: dict, report: dict) -> bool:
    if match.get("sprt"):
        return report.get("sequential", {}).get("stop_reason") in TERMINAL
    return report.get("complete_pairs") == match["pairs"]


def binding(experiment: dict, match: dict) -> dict:
    """What a played match is bound to: its manifest entry, the affinity it ran
    under and the hashes of the files it used."""
    files = [absolute(experiment["bot"]), side(match["candidate"]), side(match["reference"])]
    if any(file.suffix == ".exe" for file in files[1:]):
        files.append(absolute(experiment["network"]))
    return {
        "match": match,
        "affinity": experiment.get("affinity", "16-31"),
        "identities": {relative(file): sha256(file) for file in files},
    }


def outputs(directory: Path, match: dict) -> dict:
    """The hashes of a match's report and of its journal or records as they are now."""
    kind, suffix = ("journal", "jsonl") if match.get("sprt") else ("records", "games.jsonl")
    files = {"report": directory / f"{match['tag']}.json", kind: directory / f"{match['tag']}.{suffix}"}
    return {key: sha256(file) if file.is_file() else None for key, file in files.items()}


def bound(experiment: dict, directory: Path, match: dict) -> str | None:
    """Why the match's report is not bound to this manifest entry, affinity,
    these files and the outputs on disk (None when it is)."""
    sidecar = directory / f"{match['tag']}.match.json"
    if not sidecar.is_file():
        return "no binding sidecar"
    stored = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = binding(experiment, match)
    if {k: stored.get(k) for k in expected} != expected:
        return "the report is not bound to this manifest entry, affinity and these files"
    if stored.get("outputs") != outputs(directory, match):
        return "the report, journal or records changed after the match was bound"
    return None


def journal_pairs(journal: Path) -> tuple[dict, list[list[dict]]]:
    """The header and the complete pairs of a sequential journal, in order: a
    header line `{"protocol": ..., "protocol_digest": ...}`, then `{"pair": n,
    "games": [candidate first, reference first]}` per pair. Only an
    unterminated last line is ignored (as `bot eval --resume` does); anything
    else malformed is an error."""
    lines = journal.read_text(encoding="utf-8").split("\n")[:-1]
    if not lines:
        raise ValueError(f"{journal}: no header")
    header = json.loads(lines[0])
    if not isinstance(header, dict) or not isinstance(header.get("protocol"), dict) or "protocol_digest" not in header:
        raise ValueError(f"{journal}: the first line is not the protocol header")
    pairs = []
    for n, line in enumerate(lines[1:]):
        entry = json.loads(line)
        games = entry.get("games") if isinstance(entry, dict) else None
        if (
            entry.get("pair") != n
            or not isinstance(games, list)
            or len(games) != 2
            or not all(isinstance(g, dict) and {"winner", "end", "plies", "moves"} <= g.keys() for g in games)
        ):
            raise ValueError(f"{journal}: malformed pair line {n + 1}")
        pairs.append(games)
    return header, pairs


SURVIVORS: set[int] = (
    set()
)  # members of a coordinator's job that outlived its termination; the lock stays until they are gone


class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("ro", "wo", "oo", "rt", "wt", "ot")]


class BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", BasicLimits),
        ("IoInfo", IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def running(pid: int) -> bool:
    """Whether a process still runs (a terminated one whose handle is held is not running)."""
    try:
        return psutil.Process(pid).status() != psutil.STATUS_DEAD
    except psutil.NoSuchProcess:
        return False


class Job:
    """A Windows job object containing the coordinator and every descendant
    (the rpsi seats inherit it): terminating the job kills them all, whether
    or not their parent still lives, and the job's member list names what
    survived without traversing a dead parent."""

    KILL_ON_CLOSE, EXTENDED_LIMITS, PROCESS_IDS, CREATE_SUSPENDED = 0x2000, 9, 3, 0x4

    def __init__(self):
        self.kernel = ctypes.windll.kernel32
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = self.KILL_ON_CLOSE
        if not self.kernel.SetInformationJobObject(
            wintypes.HANDLE(self.handle), self.EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

    def assign(self, process: subprocess.Popen) -> None:
        if not self.kernel.AssignProcessToJobObject(
            wintypes.HANDLE(self.handle), wintypes.HANDLE(int(process._handle))
        ):
            raise OSError(
                ctypes.get_last_error(),
                "AssignProcessToJobObject failed (is the runner in a job that forbids nesting?)",
            )

    def members(self) -> list[int]:
        """The pids still assigned to the job."""
        count = 1024
        fields = [("assigned", wintypes.DWORD), ("listed", wintypes.DWORD), ("ids", ctypes.c_size_t * count)]
        record = type("ProcessIds", (ctypes.Structure,), {"_fields_": fields})()
        if not self.kernel.QueryInformationJobObject(
            wintypes.HANDLE(self.handle), self.PROCESS_IDS, ctypes.byref(record), ctypes.sizeof(record), None
        ):
            raise OSError(ctypes.get_last_error(), "QueryInformationJobObject failed")
        return [int(record.ids[i]) for i in range(record.listed)]

    def reap(self, process: subprocess.Popen) -> list[int]:
        """Terminates the job and waits for its members; the pids that survived."""
        self.kernel.TerminateJobObject(wintypes.HANDLE(self.handle), 1)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        deadline = time.monotonic() + 10
        while (alive := [pid for pid in self.members() if pid != process.pid and running(pid)]) and (
            time.monotonic() < deadline
        ):
            time.sleep(0.05)
        self.kernel.CloseHandle(wintypes.HANDLE(self.handle))
        SURVIVORS.update(alive)
        return alive


def read_lines(stream, lines: queue.Queue) -> None:
    """Feeds a pipe's lines to a queue, then None (the pipe closed)."""
    try:
        for line in stream:
            lines.put(line)
    except (OSError, ValueError):
        pass
    lines.put(None)


def launch(command: list[str], err, events: list | None = None) -> tuple[dict | None, list[tuple[float, dict]], int]:
    """Runs `bot eval` inside its own job: its report (the last stdout line
    without an `event`), the streamed start/end events stamped with the
    monotonic time of their receipt (appended to `events` as they come, so an
    interruption leaves them with the caller), and the exit code. However the
    run ends, the job is terminated: an exception kills the tree before it
    propagates, and a coordinator that exits abruptly leaves no seat behind."""
    events = [] if events is None else events
    report = None
    job = Job()
    process = subprocess.Popen(
        command,
        cwd=workspace_root(),
        stdout=subprocess.PIPE,
        stderr=err,
        text=True,
        encoding="utf-8",
        creationflags=Job.CREATE_SUSPENDED,  # assigned to the job before it runs, so every seat inherits it
    )
    try:
        job.assign(process)
        psutil.Process(process.pid).resume()
        assert process.stdout is not None
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=read_lines, args=(process.stdout, lines), daemon=True).start()
        while True:
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                if process.poll() is None:
                    continue
                break  # the coordinator has exited; whatever still holds its pipe is an orphan for the reaper
            if line is None:
                break
            if not line.strip():
                continue
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise ValueError(f"bot eval printed {line.strip()[:80]!r}")
            if "event" not in entry:
                report = entry
            elif entry["event"] in ("start", "end"):
                events.append((time.monotonic(), entry))
        code = process.wait()
    finally:
        job.reap(process)
        if process.stdout is not None:
            process.stdout.close()
    return report, events, code


def occupancy(events: list[tuple[float, dict]], batch: int, concurrent: int) -> dict:
    """The pair workers' occupancy per stopping batch from one invocation's
    timestamped events: a pair is busy from its first game's start to its
    second game's end, a batch's idle share is 1 - busy / (concurrent x
    elapsed). The idle share counts fully observed batches only."""
    starts: dict[tuple[int, int], float] = {}
    ends: dict[tuple[int, int], float] = {}
    for at, event in events:
        (starts if event["event"] == "start" else ends)[(event["pair"], event["game"])] = at
    pairs = {p for p, g in starts if g == 0 and (p, 1) in ends}
    batches = []
    for b in sorted({p // batch for p in pairs}):
        members = [p for p in pairs if p // batch == b]
        busy = sum(ends[(p, 1)] - starts[(p, 0)] for p in members)
        elapsed = max(ends[(p, 1)] for p in members) - min(starts[(p, 0)] for p in members)
        batches.append(
            {
                "batch": b,
                "pairs": len(members),
                "complete": len(members) == batch,
                "elapsed_seconds": elapsed,
                "busy_seconds": busy,
                "idle_share": 1 - busy / (concurrent * elapsed) if elapsed > 0 else None,
            }
        )
    complete = [x for x in batches if x["complete"] and x["elapsed_seconds"] > 0]
    idle = (
        1 - sum(x["busy_seconds"] for x in complete) / (concurrent * sum(x["elapsed_seconds"] for x in complete))
        if complete
        else None
    )
    return {"observed_pairs": len(pairs), "idle_share": idle, "batches": batches}


def committed_pairs(journal: Path) -> int | None:
    """The complete pairs a journal holds: 0 without a journal, None when its header is unreadable."""
    if not journal.is_file():
        return 0
    try:
        return len(journal_pairs(journal)[1])
    except ValueError:
        return None


def record_timing(directory: Path, match: dict, report: dict | None, events: list, pairs: tuple, times: tuple, code):
    """Appends one invocation to `<tag>.timing.json`, finished, failed or
    interrupted: the launch time, pairs before, committed to the journal after
    and reported; process, startup, play (first to last event), active (to
    the process end when a game was still open) and finish seconds; new pairs
    per active second; the batch occupancy (batch size from the journal
    header); and the aggregate over every invocation, committed pairs over
    summed active seconds and over summed process seconds, never paused
    time. Written atomically."""
    initial, committed = pairs
    started, launched, ended = times
    journal = directory / f"{match['tag']}.jsonl"
    try:
        header = journal_pairs(journal)[0] if journal.is_file() else {}
    except ValueError:
        header = {}
    batch = int(header.get("protocol", {}).get("test", {}).get("batch", 16))
    stamps = [at for at, _ in events]
    first, last = (min(stamps), max(stamps)) if stamps else (None, None)
    completed = code == 0 and report is not None
    open_tail = bool(events) and events[-1][1]["event"] == "start"
    active = None if first is None else (ended - first if open_tail or not completed else last - first)
    new = None if committed is None or initial is None else committed - initial
    entry = {
        "launched_utc": started,
        "completed": completed,
        "interrupted": code is None,
        "returncode": code,
        "initial_pairs": initial,
        "committed_pairs": committed,
        "reported_pairs": None if report is None else report.get("pairs"),
        "process_seconds": ended - launched,
        "startup_seconds": None if first is None else first - launched,
        "play_seconds": None if first is None else last - first,
        "active_seconds": active,
        "finish_seconds": None if last is None or not completed else ended - last,
        "pairs_per_second": None if new is None or not active or active <= 0 else new / active,
        **occupancy(events, batch, int(match.get("concurrent", 8))),
    }
    path = directory / f"{match['tag']}.timing.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"invocations": []}
    data["invocations"].append(entry)
    counted = [i for i in data["invocations"] if i["committed_pairs"] is not None and i["initial_pairs"] is not None]
    total = sum(i["committed_pairs"] - i["initial_pairs"] for i in counted)
    active_sum = sum(i["active_seconds"] or 0.0 for i in data["invocations"])
    process_sum = sum(i["process_seconds"] for i in data["invocations"])
    data.update(
        committed_pairs=total,
        active_seconds=active_sum,
        process_seconds=process_sum,
        pairs_per_second=total / active_sum if active_sum > 0 else None,
        pairs_per_process_second=total / process_sum if process_sum > 0 else None,
    )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def play(experiment: dict, directory: Path, match: dict) -> dict:
    """The match's report: the one on disk when it is finished and bound to
    this manifest entry, these files and its own outputs, else played (a
    sequential test resumes its journal). A sequential invocation's timing is
    recorded however it ends, before the report is cached."""
    for role in ("candidate", "reference"):
        if match[role].endswith((".nnue", ".exe")):
            verify(experiment, side(match[role]))
    if (side(match["candidate"]).suffix == ".exe") != (side(match["reference"]).suffix == ".exe"):
        raise ValueError(f"{match['tag']}: a search build plays another search build on the shared network")
    verify(experiment, absolute(experiment["bot"]))
    if side(match["candidate"]).suffix == ".exe":
        verify(experiment, absolute(experiment["network"]))
    report_file = directory / f"{match['tag']}.json"
    if report_file.is_file() and bound(experiment, directory, match) is None:
        loaded = json.loads(report_file.read_text(encoding="utf-8"))
        if finished(match, loaded):
            return loaded
    command = command_of(experiment, directory, match)
    journal = directory / f"{match['tag']}.jsonl"
    initial = committed_pairs(journal) if match.get("sprt") else 0
    started, launched = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.monotonic()
    events: list = []
    report, code = None, None
    try:
        with (directory / f"{match['tag']}.err").open("a", encoding="utf-8") as err:
            report, _, code = launch(command, err, events)
    finally:
        if match.get("sprt"):
            times = (started, launched, time.monotonic())
            record_timing(directory, match, report, events, (initial, committed_pairs(journal)), times, code)
    if code != 0 or report is None:
        raise RuntimeError(f"bot eval failed for {match['tag']} (exit code {code}, see {match['tag']}.err)")
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sidecar = {**binding(experiment, match), "outputs": outputs(directory, match), "command": command}
    (directory / f"{match['tag']}.match.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return report


def replay(experiment: dict, directory: Path, match: dict, report: dict) -> list[str]:
    """C1's own validation of a finished sequential journal: `bot eval --resume`
    on it replays the pairs under the same protocol (so under the same
    affinity), plays nothing once the test has stopped and reports again; the
    problems are where that report differs from the cached one. Only a
    journal that exists, is bound and passed the audit is given to C1, so the
    replay can never start or continue a test."""
    if not (directory / f"{match['tag']}.jsonl").is_file():
        raise RuntimeError(f"{match['tag']}: no journal to replay")
    nice(experiment.get("affinity", "16-31"))
    with (directory / f"{match['tag']}.replay.err").open("a", encoding="utf-8") as err:
        again, _, code = launch(command_of(experiment, directory, match), err)
    if code != 0 or again is None:
        return [f"C1 refuses to replay the journal (exit code {code}, see {match['tag']}.replay.err)"]
    keys = ("wins", "draws", "losses", "pairs", "complete_pairs")
    problems = [f"C1's replay gives another {k}" for k in keys if again.get(k) != report.get(k)]
    mine, theirs = report.get("sequential") or {}, again.get("sequential") or {}
    keys = ("protocol_digest", "counts", "llr", "stop_reason")
    problems += [f"C1's replay gives another {k}" for k in keys if theirs.get(k) != mine.get(k)]
    return problems


# Verdict.


def pair_points(games: list[dict]) -> int:
    """The candidate's points (0..4) in a pair: first in the first game, second in the other."""
    points = 0
    for game, mine in zip(games, (0, 1), strict=True):
        points += 2 if game["winner"] == mine else 1 if game["winner"] is None else 0
    return points


ENDS = {"Goal", "Elimination", "Stalemate", "CaptureClock"}  # C1 censors Forfeit, PlyCap and Interrupted


def first_mover_points(game: dict) -> float:
    return 1.0 if game["winner"] == 0 else 0.5 if game["winner"] is None else 0.0


def game_result(game: dict, mine: int) -> str:
    return "draws" if game["winner"] is None else "wins" if game["winner"] == mine else "losses"


def check_pair(games: list, names: tuple, opening: int, problems: list[str], where: str) -> bool:
    """Two games of one opening with the seats swapped, the candidate first in the first game."""
    candidate, reference = names
    for n, (game, first, second) in enumerate(zip(games, (candidate, reference), (reference, candidate), strict=True)):
        if game.get("first") != first or game.get("second") != second:
            problems.append(f"{where}: game {n} has other seats than the pair's order")
            return False
        if game.get("end") not in ENDS:
            problems.append(f"{where}: game {n} ended by {game.get('end')}")
            return False
    if games[0]["moves"][:opening] != games[1]["moves"][:opening]:
        problems.append(f"{where}: the two games have different openings")
        return False
    return True


def expected_provenance(experiment: dict, match: dict) -> dict:
    """The file hashes C1's journal provenance must carry for this match: the
    coordinator and, per side, the binary and the network."""
    pinned = experiment.get("identities", {})
    out = {"coordinator": pinned.get(relative(absolute(experiment["bot"])))}
    for role in ("candidate", "reference"):
        file = side(match[role])
        if file.suffix == ".exe":
            network = relative(absolute(experiment["network"]))
            out[role] = {"binary": pinned.get(relative(file)), "network": pinned.get(network)}
        else:
            out[role] = {"binary": out["coordinator"], "network": pinned.get(relative(file)) or sha256(file)}
    return out


def protocol_problems(protocol: dict, match: dict, experiment: dict | None) -> list[str]:
    """The journal protocol's settings and provenance against the match and the pinned identities."""
    settings = protocol.get("settings") or {}
    expected = {
        "seed": match["seed"],
        "move_ms": match["move_ms"],
        "opening_plies": match.get("opening_plies", 8),
        "workers": match.get("concurrent", 8),
    }
    out = [
        f"protocol {k} is {settings.get(k)}, the match says {v}" for k, v in expected.items() if settings.get(k) != v
    ]
    provenance = protocol.get("provenance") or {}
    if provenance.get("player_threads") != match.get("player_threads", 1):
        out.append(f"protocol player threads are {provenance.get('player_threads')}")
    if experiment is not None:
        wanted = expected_provenance(experiment, match)
        if (provenance.get("coordinator") or {}).get("sha256") != wanted["coordinator"]:
            out.append("the journal's coordinator is not the pinned bot")
        for role in ("candidate", "reference"):
            recorded = provenance.get(role) or {}
            for kind in ("binary", "network"):
                if (recorded.get(kind) or {}).get("sha256") != wanted[role][kind]:
                    out.append(f"the journal's {role} {kind} is not the pinned file")
    return out


def audit(match: dict, report: dict, directory: Path, experiment: dict | None = None) -> dict:
    """The recorded games of a match reconciled with its report (protocol,
    pairs, seats, openings, endings, results) and their descriptive counts."""
    problems: list[str] = []
    names = (report.get("candidate"), report.get("reference"))
    opening = int(match.get("opening_plies", 8))
    games: list[dict] = []
    results: Counter = Counter()
    out: dict = {}
    try:
        if match.get("sprt"):
            journal = directory / f"{match['tag']}.jsonl"
            header, pairs = journal_pairs(journal) if journal.is_file() else ({}, [])
            sequential = report.get("sequential") or {}
            if header.get("protocol_digest") != sequential.get("protocol_digest") or header.get("protocol") != (
                sequential.get("protocol")
            ):
                problems.append("the journal header's protocol differs from the report's")
            problems += protocol_problems(header.get("protocol") or {}, match, experiment)
            counts = [0] * 5
            for n, pair in enumerate(pairs):
                if check_pair(pair, names, opening, problems, f"pair {n}"):
                    counts[pair_points(pair)] += 1
                    for game, mine in zip(pair, (0, 1), strict=True):
                        results[game_result(game, mine)] += 1
            games = [g for pair in pairs for g in pair]
            if len(pairs) != report.get("pairs") or len(pairs) != report.get("complete_pairs"):
                problems.append(
                    f"{len(pairs)} recorded pairs, the report says {report.get('pairs')} "
                    f"of which {report.get('complete_pairs')} complete"
                )
            if counts != list(sequential.get("counts") or []):
                problems.append("the recorded pairs give other pentanomial counts than the report")
            if sequential.get("stop_reason") in ("accept", "reject", "inconclusive") and sequential.get("error"):
                problems.append(f"a completed test reports an error: {sequential.get('error')}")
        else:
            records = directory / f"{match['tag']}.games.jsonl"
            if records.is_file():
                games = [json.loads(line) for line in records.read_text(encoding="utf-8").split("\n") if line]
            if len(games) % 2:
                problems.append("an odd number of recorded games")
            for n in range(0, len(games) - 1, 2):
                if check_pair(games[n : n + 2], names, opening, problems, f"pair {n // 2}"):
                    for game, mine in zip(games[n : n + 2], (0, 1), strict=True):
                        results[game_result(game, mine)] += 1
            if len(games) != 2 * (report.get("complete_pairs") or 0):
                problems.append(f"{len(games)} recorded games for {report.get('complete_pairs')} complete pairs")
        if any(results[k] != report.get(k) for k in ("wins", "draws", "losses")):
            problems.append("the recorded games give other results than the report")
        if report.get("forfeits"):
            problems.append(f"{report['forfeits']} forfeits")
        if games:
            out = {
                "ends": dict(Counter(g.get("end") for g in games)),
                "first_mover_score": sum(first_mover_points(g) for g in games) / len(games),
                "distinct_openings": len({tuple(g["moves"][:opening]) for g in games}),
                "mean_plies": sum(g["plies"] for g in games) / len(games),
            }
    except (TypeError, KeyError, AttributeError, ValueError) as error:
        problems.append(f"malformed records: {error}")
        games, out = [], {}
    return {"games": len(games), "consistent": not problems, "problems": problems, **out}


def summarise(match: dict, report: dict | None, directory: Path, experiment: dict | None = None) -> dict:
    if report is None:
        return {"tag": match["tag"], "played": False}
    games = report["wins"] + report["draws"] + report["losses"]
    lo, hi = report["bootstrap_interval"]
    records = audit(match, report, directory, experiment)
    problems = list(records["problems"])
    # C1 names a network side by its path and a search build by the engine's id; the builds are identified
    # by the journal provenance (sequential) and the binding sidecar (fixed count).
    for role in ("candidate", "reference"):
        if side(match[role]).suffix != ".exe" and str(side(match[role])) not in str(report.get(role, "")):
            problems.append(f"the report's {role} is not the manifest's")
    if report.get("move_ms") != match["move_ms"]:
        problems.append("the report's clock differs from the manifest")
    if match.get("sprt"):
        sequential = report.get("sequential", {})
        valid = sequential.get("stop_reason") in ("accept", "reject", "inconclusive")
    else:
        valid = report.get("complete_pairs") == match["pairs"] and report.get("forfeits", 0) == 0
    out = {
        "tag": match["tag"],
        "played": True,
        "candidate": match["candidate"],
        "reference": match["reference"],
        "move_ms": match["move_ms"],
        "games": games,
        "wins": report["wins"],
        "draws": report["draws"],
        "losses": report["losses"],
        "score": (report["wins"] + report["draws"] / 2) / games if games else None,
        "interval": [(1 + lo) / 2, (1 + hi) / 2],
        "valid": valid and not problems,
        "problems": problems,
        "records": records,
    }
    if match.get("sprt"):
        out.update(
            pairs=report.get("pairs"),
            stop_reason=sequential.get("stop_reason"),
            llr=sequential.get("llr"),
            counts=sequential.get("counts"),
            protocol_digest=sequential.get("protocol_digest"),
            protocol=sequential.get("protocol"),
            error=sequential.get("error"),
            interval_descriptive_only=sequential.get("intervals_descriptive_only", True),
        )
    return out


def validate(experiment: dict) -> None:
    """The manifest's shape: unique tags, a pair count exactly for fixed-count
    matches, rules of a known type naming matches of their kind."""
    if experiment.get("schema") != 2:
        raise ValueError("experiment.json must have schema 2")
    matches = {m["tag"]: m for m in experiment.get("matches", [])}
    if len(matches) != len(experiment.get("matches", [])):
        raise ValueError("match tags must be unique")
    for tag, match in matches.items():
        if bool(match.get("sprt")) == ("pairs" in match):
            raise ValueError(f"{tag}: a sequential match has no pair count, a fixed-count match needs one")
    for rule in experiment.get("rules", []):
        sequential = rule["type"] in SEQUENTIAL_RULES
        if not sequential and rule["type"] not in FIXED_RULES:
            raise ValueError(f"unknown rule type {rule['type']!r}")
        for tag in rule["matches"]:
            if tag not in matches:
                raise ValueError(f"rule {rule['name']} names no match {tag!r}")
            if bool(matches[tag].get("sprt")) != sequential:
                kind = "sequential" if sequential else "fixed-count"
                raise ValueError(f"rule {rule['name']}: {rule['type']} judges {kind} matches only")


def verdict(experiment: dict, directory: Path, replays: bool = True) -> dict:
    """The experiment judged from what is on disk: the pinned inputs verified,
    every endpoint and every match audited (a finished sequential journal
    replayed through C1), the rules decided."""
    validate(experiment)
    for path in fixed_inputs(experiment):
        verify(experiment, path)
    arms, broken = [], {}
    for arm in experiment.get("arms", []):
        endpoints = {}
        for name in arm.get("endpoints", {}):
            file = endpoint_file(arm["run"], name)
            sidecar = file.with_suffix(".nnue.json")
            if sidecar.is_file():
                info = json.loads(sidecar.read_text(encoding="utf-8"))
                kept = {k: info[k] for k in ("sha256", "epoch", "objective") if k in info}
                endpoints[name] = {"file": str(file), **kept, "problem": endpoint_problem(arm, name)}
                if endpoints[name]["problem"]:
                    broken[f"{arm['run']}:{name}"] = endpoints[name]["problem"]
            else:
                endpoints[name] = None
        arms.append({"run": arm["run"], "epochs": epochs_done(arm["run"]), "endpoints": endpoints})
    matches = {}
    for match in experiment.get("matches", []):
        report = directory / f"{match['tag']}.json"
        loaded = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else None
        summary = summarise(match, loaded, directory, experiment)
        if summary["played"]:
            problems = [
                f"{role} endpoint: {broken[match[role]]}"
                for role in ("candidate", "reference")
                if match[role] in broken
            ]
            problem = bound(experiment, directory, match)
            problems += [problem] if problem else []
            clean = not problems and not summary["problems"]
            if replays and match.get("sprt") and finished(match, loaded) and clean:
                problems += replay(experiment, directory, match, loaded)
            if problems:
                summary["problems"] += problems
                summary["valid"] = False
        matches[match["tag"]] = summary
    rules = []
    for rule in experiment.get("rules", []):
        named = [matches.get(tag) for tag in rule["matches"]]
        complete = all(m and m["played"] and m["valid"] for m in named)
        if not complete:
            outcome = None
        elif rule["type"] == "lower_bound_above":
            outcome = all(m["interval"][0] > rule["threshold"] for m in named)
        elif rule["type"] == "score_at_least":
            outcome = all(m["score"] >= rule["threshold"] for m in named)
        else:
            outcome = all(m.get("stop_reason") == "accept" for m in named)
        rules.append({**rule, "complete": complete, "met": outcome})
    out = {
        "experiment": experiment["name"],
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arms": arms,
        "matches": list(matches.values()),
        "rules": rules,
    }
    (directory / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def visible_devices(experiment: dict) -> str:
    """The GPU the trainers may see unless the environment says otherwise: none,
    or device 0 when an arm's config names cuda (the trainer itself hides CUDA
    by default, so a preregistered cuda arm must be given it here)."""
    devices = [str(arm.get("train", {}).get("config", {}).get("device", "cpu")) for arm in experiment.get("arms", [])]
    return "0" if any(device.startswith("cuda") for device in devices) else "-1"


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "lock":
        if argv[1] == "status":
            held = lock_read()
            alive = {"alive": psutil.pid_exists(held["pid"])} if held else {}
            print(json.dumps({"held": held is not None, **(held or {}), **alive}))
        elif argv[1] == "acquire":
            mask = argv[4] if len(argv) > 4 else "16-31"
            print(json.dumps(lock_acquire(argv[2], argv[3], mask, pid=os.getppid())))
        elif argv[1] == "release":
            print(json.dumps(lock_release(argv[2], int(argv[3]) if len(argv) > 3 else None)))
        else:
            raise SystemExit(__doc__)
        return 0
    if len(argv) < 2 or argv[0] not in ("pin", "run", "verdict"):
        raise SystemExit(__doc__)
    file = Path(argv[1]).resolve()
    experiment = json.loads(file.read_text(encoding="utf-8"))
    validate(experiment)
    directory = file.parent
    if argv[0] == "pin":
        experiment["identities"] = identities(experiment)
        file.write_text(json.dumps(experiment, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(experiment["identities"], indent=1))
        return 0
    if argv[0] == "verdict":
        print(json.dumps(verdict(experiment, directory), indent=2))
        return 0
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    for path in fixed_inputs(experiment):
        verify(experiment, path)
    nice(experiment.get("affinity", "16-31"))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", visible_devices(experiment))
    lock_acquire("experiment", f"{experiment['name']} ({only or 'all'})", experiment.get("affinity", "16-31"))
    try:
        if only in (None, "train"):
            for arm in experiment.get("arms", []):
                run_arm(experiment, arm)
                print(json.dumps({"event": "arm", "run": arm["run"], "epochs": epochs_done(arm["run"])}), flush=True)
        if only in (None, "match"):
            for match in experiment.get("matches", []):
                summary = summarise(match, play(experiment, directory, match), directory, experiment)
                print(json.dumps({"event": "match", **summary}), flush=True)
    finally:
        # The reservation outlives a failure that leaves a child (a seat's engine) alive: release it by hand
        # once every child is gone (`lock release experiment <pid>`).
        survivors = sorted(
            {p.pid for p in psutil.Process().children(recursive=True)} | {p for p in SURVIVORS if running(p)}
        )
        if survivors:
            print(json.dumps({"event": "lock_kept", "surviving_children": survivors}), flush=True)
        else:
            lock_release("experiment", os.getpid())
    print(json.dumps({"event": "verdict", **{k: v for k, v in verdict(experiment, directory).items() if k == "rules"}}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
