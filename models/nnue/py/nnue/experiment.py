"""One preregistered experiment from one file: trains its arms, exports their
endpoints, plays its matches through `bot eval` and writes `verdict.json`.
Every step is skipped when its artifact exists and is bound to the manifest,
so a relaunch resumes. Heavy phases run one at a time at below-normal
priority on the declared logical CPUs, under the workspace's compute lock.

    python -m nnue.experiment pin <experiment.json>
    python -m nnue.experiment run <experiment.json> [--only train|match]
    python -m nnue.experiment verdict <experiment.json>
    python -m nnue.experiment lock status | acquire <owner> <phase> [<mask>] | release <owner>

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
records its games. A played match is bound to its manifest entry and the
hashes of the files it used (`<tag>.match.json`); a report whose binding
differs is played again. Rule types: `lower_bound_above` and
`score_at_least` (fixed-count matches only: the paired bootstrap lower
bound above, or the score at least, the threshold in every named match) and
`sprt_accept` (sequential matches only: every named test accepted). The
verdict records numbers, rule outcomes and an audit of every match's
recorded games reconciled with its report (pairs, games, the candidate's
results; how the games ended, the first mover's score, distinct openings,
mean plies); a match whose records or binding disagree is invalid.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
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
    """Export a checkpoint exactly as the trainer would export it at that epoch."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    model = NNUE(config["hidden"], config["version"])
    model.load_state_dict(saved["model"])
    return export.export(
        model, target, {"checkpoint": str(checkpoint), "epoch": saved["epoch"], "step": saved["step"], "config": config}
    )


def endpoint_file(run: str, name: str) -> Path:
    return run_dir(run) / ("best.nnue" if name == "best" else f"{name}.nnue")


def run_arm(experiment: dict, arm: dict) -> None:
    run, endpoints = arm["run"], arm.get("endpoints", {})
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
    `bot eval`, so paths with spaces are refused; a sequential test also names
    the network so the journal hashes the shared weights."""
    file = side(match[role])
    if file.suffix != ".exe":
        return [f"--{role}", f"nnue:{file}"]
    network = verify(experiment, absolute(experiment["network"]))
    if " " in str(file) or " " in str(network):
        raise ValueError(f"{match['tag']}: an rpsi seat cannot hold a path with spaces ({file}, {network})")
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
    """What a played match is bound to: its manifest entry and the hashes of the files it used."""
    files = [absolute(experiment["bot"]), side(match["candidate"]), side(match["reference"])]
    if any(file.suffix == ".exe" for file in files[1:]):
        files.append(absolute(experiment["network"]))
    return {"match": match, "identities": {relative(file): sha256(file) for file in files}}


def journal_pairs(journal: Path) -> list[list[dict]]:
    """The complete pairs of a sequential journal, in order: a header line
    `{"protocol": ...}`, then `{"pair": n, "games": [candidate first,
    reference first]}` per pair. Only an unterminated last line is ignored
    (as `bot eval --resume` does); anything else malformed is an error."""
    lines = journal.read_text(encoding="utf-8").split("\n")[:-1]
    if not lines:
        raise ValueError(f"{journal}: no header")
    header = json.loads(lines[0])
    if not isinstance(header, dict) or "protocol" not in header:
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
    return pairs


def launch(command: list[str], err) -> tuple[dict | None, list[tuple[float, dict]]]:
    """Runs `bot eval`: its report (the last stdout line without an `event`)
    and the streamed start/end events, each stamped with the monotonic time
    of its receipt."""
    report, events = None, []
    with subprocess.Popen(
        command, cwd=workspace_root(), stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8"
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            if not line.strip():
                continue
            entry = json.loads(line)
            if "event" not in entry:
                report = entry
            elif entry["event"] in ("start", "end"):
                events.append((time.monotonic(), entry))
    if process.returncode != 0:
        raise RuntimeError(f"bot eval failed with code {process.returncode}")
    return report, events


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


def record_timing(
    directory: Path, match: dict, report: dict, events: list, initial: int, launched: float, ended: float
):
    """Appends one invocation's timing to `<tag>.timing.json`: pairs before and
    after, process, startup, play and finish seconds, pairs per second of the
    play time and the batch occupancy (the journal header's batch size)."""
    journal = directory / f"{match['tag']}.jsonl"
    header = json.loads(journal.read_text(encoding="utf-8").split("\n", 1)[0])
    batch = int(header["protocol"].get("test", {}).get("batch", 16))
    stamps = [at for at, _ in events]
    first, last = (min(stamps), max(stamps)) if stamps else (None, None)
    final = int(report.get("pairs") or 0)
    entry = {
        "launched": time.strftime("%Y-%m-%d %H:%M:%S"),
        "initial_pairs": initial,
        "final_pairs": final,
        "process_seconds": ended - launched,
        "startup_seconds": None if first is None else first - launched,
        "play_seconds": None if first is None else last - first,
        "finish_seconds": None if last is None else ended - last,
        "pairs_per_second": None if first is None or last <= first else (final - initial) / (last - first),
        **occupancy(events, batch, int(match.get("concurrent", 8))),
    }
    path = directory / f"{match['tag']}.timing.json"
    invocations = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    path.write_text(json.dumps([*invocations, entry], indent=2), encoding="utf-8")


def play(experiment: dict, directory: Path, match: dict) -> dict:
    """The match's report: the one on disk when it is finished and bound to
    this manifest entry and these files, else played (a sequential test
    resumes its journal)."""
    for role in ("candidate", "reference"):
        if match[role].endswith((".nnue", ".exe")):
            verify(experiment, side(match[role]))
    if (side(match["candidate"]).suffix == ".exe") != (side(match["reference"]).suffix == ".exe"):
        raise ValueError(f"{match['tag']}: a search build plays another search build on the shared network")
    verify(experiment, absolute(experiment["bot"]))
    bound = binding(experiment, match)
    report_file, sidecar = directory / f"{match['tag']}.json", directory / f"{match['tag']}.match.json"
    if report_file.is_file() and sidecar.is_file():
        loaded = json.loads(report_file.read_text(encoding="utf-8"))
        stored = json.loads(sidecar.read_text(encoding="utf-8"))
        if {k: stored.get(k) for k in bound} == bound and finished(match, loaded):
            return loaded
    command = command_of(experiment, directory, match)
    journal = directory / f"{match['tag']}.jsonl"
    initial = len(journal_pairs(journal)) if match.get("sprt") and journal.is_file() else 0
    launched = time.monotonic()
    with (directory / f"{match['tag']}.err").open("a", encoding="utf-8") as err:
        report, events = launch(command, err)
    ended = time.monotonic()
    if report is None:
        raise RuntimeError(f"bot eval printed no report for {match['tag']} ({match['tag']}.err)")
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sidecar.write_text(json.dumps({**bound, "command": command}, indent=2), encoding="utf-8")
    if match.get("sprt"):
        record_timing(directory, match, report, events, initial, launched, ended)
    return report


# Verdict.


def pair_points(games: list[dict]) -> int:
    """The candidate's points (0..4) in a pair: first in the first game, second in the other."""
    points = 0
    for game, mine in zip(games, (0, 1), strict=True):
        points += 2 if game["winner"] == mine else 1 if game["winner"] is None else 0
    return points


def audit(match: dict, report: dict, directory: Path) -> dict:
    """The recorded games of a match reconciled with its report (pairs, games,
    the candidate's results) and their descriptive counts."""
    problems: list[str] = []
    candidate = report.get("candidate")
    games: list[dict] = []
    if match.get("sprt"):
        journal = directory / f"{match['tag']}.jsonl"
        try:
            pairs = journal_pairs(journal) if journal.is_file() else []
        except ValueError as error:
            pairs, problems = [], [str(error)]
        games = [g for pair in pairs for g in pair]
        counts = [0] * 5
        for pair in pairs:
            counts[pair_points(pair)] += 1
        if len(pairs) != report.get("pairs"):
            problems.append(f"{len(pairs)} recorded pairs, the report says {report.get('pairs')}")
        if counts != list(report.get("sequential", {}).get("counts") or []):
            problems.append("the recorded pairs give other pentanomial counts than the report")
    else:
        records = directory / f"{match['tag']}.games.jsonl"
        try:
            if records.is_file():
                games = [json.loads(line) for line in records.read_text(encoding="utf-8").split("\n") if line]
        except ValueError as error:
            problems.append(f"malformed record: {error}")
        results: Counter = Counter()
        for g in games:
            mine = 0 if g.get("first") == candidate else 1 if g.get("second") == candidate else None
            if mine is None:
                problems.append("a recorded game without the candidate")
                break
            results["draws" if g["winner"] is None else "wins" if g["winner"] == mine else "losses"] += 1
        if len(games) != 2 * (report.get("complete_pairs") or 0):
            problems.append(f"{len(games)} recorded games for {report.get('complete_pairs')} complete pairs")
        elif any(results[k] != report.get(k) for k in ("wins", "draws", "losses")):
            problems.append("the recorded games give other results than the report")
    out = {"games": len(games), "consistent": not problems, "problems": problems}
    if games:
        out.update(
            ends=dict(Counter(g.get("end") for g in games)),
            first_mover_score=sum(1.0 if g["winner"] == 0 else 0.5 if g["winner"] is None else 0.0 for g in games)
            / len(games),
            distinct_openings=len({tuple(g["moves"][: match.get("opening_plies", 8)]) for g in games}),
            mean_plies=sum(g["plies"] for g in games) / len(games),
        )
    return out


def summarise(match: dict, report: dict | None, directory: Path) -> dict:
    if report is None:
        return {"tag": match["tag"], "played": False}
    games = report["wins"] + report["draws"] + report["losses"]
    lo, hi = report["bootstrap_interval"]
    records = audit(match, report, directory)
    problems = list(records["problems"])
    if (
        str(side(match["candidate"])) not in str(report.get("candidate", ""))
        or str(side(match["reference"])) not in str(report.get("reference", ""))
        or report.get("move_ms") != match["move_ms"]
    ):
        problems.append("the report's sides or clock differ from the manifest")
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


def verdict(experiment: dict, directory: Path) -> dict:
    validate(experiment)
    arms = []
    for arm in experiment.get("arms", []):
        endpoints = {}
        for name in arm.get("endpoints", {}):
            file = endpoint_file(arm["run"], name)
            sidecar = file.with_suffix(".nnue.json")
            if sidecar.is_file():
                info = json.loads(sidecar.read_text(encoding="utf-8"))
                kept = {k: info[k] for k in ("sha256", "epoch", "objective") if k in info}
                endpoints[name] = {"file": str(file), **kept}
            else:
                endpoints[name] = None
        arms.append({"run": arm["run"], "epochs": epochs_done(arm["run"]), "endpoints": endpoints})
    matches = {}
    for match in experiment.get("matches", []):
        report = directory / f"{match['tag']}.json"
        loaded = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else None
        summary = summarise(match, loaded, directory)
        if summary["played"]:
            bound = directory / f"{match['tag']}.match.json"
            try:
                expected = binding(experiment, match)
                stored = json.loads(bound.read_text(encoding="utf-8")) if bound.is_file() else {}
                if {k: stored.get(k) for k in expected} != expected:
                    raise ValueError("the report is not bound to this manifest entry and these files")
            except (OSError, ValueError) as error:
                summary["problems"].append(str(error))
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
            print(json.dumps(lock_release(argv[2])))
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
    devices = [str(arm.get("train", {}).get("config", {}).get("device", "cpu")) for arm in experiment.get("arms", [])]
    if not any(device.startswith("cuda") for device in devices):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    lock_acquire("experiment", f"{experiment['name']} ({only or 'all'})", experiment.get("affinity", "16-31"))
    try:
        if only in (None, "train"):
            for arm in experiment.get("arms", []):
                run_arm(experiment, arm)
                print(json.dumps({"event": "arm", "run": arm["run"], "epochs": epochs_done(arm["run"])}), flush=True)
        if only in (None, "match"):
            for match in experiment.get("matches", []):
                summary = summarise(match, play(experiment, directory, match), directory)
                print(json.dumps({"event": "match", **summary}), flush=True)
    finally:
        lock_release("experiment", os.getpid())
    print(json.dumps({"event": "verdict", **{k: v for k, v in verdict(experiment, directory).items() if k == "rules"}}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
