"""One preregistered experiment from one file: trains its arms, exports their
endpoints, plays its matches through `bot eval` and writes `verdict.json`.
Every step is skipped when its artifact exists, so a relaunch resumes. Heavy
phases run one at a time at below-normal priority on the declared logical
CPUs, under the workspace's compute lock.

    python -m nnue.experiment run <experiment.json> [--only train|match|verdict]
    python -m nnue.experiment verdict <experiment.json>
    python -m nnue.experiment lock status | acquire <owner> <phase> [<mask>] | release <owner>

The file (schema 2) sits in its experiment directory, where the match reports
and `verdict.json` are written:

    {"schema": 2, "name": "long_training",
     "bot": "runs/nnue_bins/accepted16.exe", "bot_sha256": "...",
     "affinity": "16-31",
     "arms": [{"run": "long60_s2",
               "train": {"init": "runs/nnue_candidates/mb_b.nnue",
                         "config": {"epochs": 60, "seed": 2, ...},
                         "data": [["selfplay_gen3", 0.347], ...]},
               "stops": [20],
               "endpoints": {"epoch20": "latest@20", "epoch60": "latest", "best": "best"}}],
     "matches": [{"tag": "s2_60_vs_20_50", "candidate": "long60_s2:epoch60",
                  "reference": "long60_s2:epoch20", "move_ms": 50, "pairs": 200,
                  "seed": 2026091521, "player_threads": 1, "concurrent": 8}],
     "rules": [{"name": "promotion", "type": "lower_bound_above", "threshold": 0.5,
                "matches": ["s2_60_vs_mb_b_50", "s2_60_vs_mb_b_100"]}]}

An arm without `train` names an existing run. `stops` are epoch counts at
which training pauses (`--stop_epoch`), the checkpoint is retained as
`<run>/epoch<N>.pt` and resumed exactly; an endpoint `latest@N` is that
checkpoint's export, `latest` the final checkpoint's, `best` the trainer's
`best.nnue`. A match side is `<run>:<endpoint>` or a `.nnue` path. Rule
types: `lower_bound_above` (every named match's paired lower bound above
the threshold) and `score_at_least` (every named match's score at least the
threshold). Intervals are the eval tool's opening-pair bootstrap on the
score scale; the verdict records numbers and rule outcomes, nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch

from . import export, paths
from .model import NNUE
from .paths import run_dir

LOCK = "runs/nnue_plan/compute_lock"


def workspace_root() -> Path:
    return paths.workspace_root()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cpus(mask: str) -> list[int]:
    out: list[int] = []
    for part in mask.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def pin(mask: str) -> None:
    """Below-normal priority and the affinity mask for this process and its children."""
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(cpus(mask))


# The compute lock: one heavy phase at a time; created atomically, never taken from a live owner.


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


def lock_release(owner: str) -> dict:
    held = lock_read()
    if held is None or held["owner"] != owner:
        raise RuntimeError(f"compute lock not held by {owner}: {held}")
    lock_path().unlink()
    return held


# Training arms.


def epochs_done(run: str) -> int:
    log = run_dir(run) / "log.csv"
    if not log.is_file():
        return 0
    with log.open(encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def train(arm: dict, stop: int | None) -> None:
    spec = arm["train"]
    command = [sys.executable, "-m", "nnue.train", "--run", arm["run"]]
    for key, value in spec.get("config", {}).items():
        command += [f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)]
    if spec.get("init"):
        command += ["--init", str(spec["init"])]
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
    model = NNUE(config["hidden"], config["buckets"], config["version"])
    model.load_state_dict(saved["model"])
    return export.export(
        model, target, {"checkpoint": str(checkpoint), "epoch": saved["epoch"], "step": saved["step"], "config": config}
    )


def endpoint_file(run: str, name: str) -> Path:
    return run_dir(run) / ("best.nnue" if name == "best" else f"{name}.nnue")


def run_arm(arm: dict) -> None:
    run, endpoints = arm["run"], arm.get("endpoints", {})
    if all(endpoint_file(run, name).is_file() for name in endpoints):
        return
    if "train" in arm:
        total = int(arm["train"]["config"].get("epochs", 20))
        for stop in sorted(arm.get("stops", [])):
            retained = run_dir(run) / f"epoch{stop}.pt"
            if retained.is_file():
                continue
            if epochs_done(run) < stop:
                train(arm, stop)
            if epochs_done(run) != stop:
                raise RuntimeError(f"{run}: {epochs_done(run)} epochs where the retained checkpoint needs {stop}")
            shutil.copy2(run_dir(run) / "latest.pt", retained)
        if epochs_done(run) < total:
            train(arm, None)
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
    if spec.endswith(".nnue"):
        return (workspace_root() / spec).resolve()
    run, _, endpoint = spec.partition(":")
    return endpoint_file(run, endpoint).resolve()


def play(experiment: dict, directory: Path, match: dict) -> dict:
    report = directory / f"{match['tag']}.json"
    if report.is_file():
        try:
            loaded = json.loads(report.read_text(encoding="utf-8"))
            if loaded.get("complete_pairs") == match["pairs"]:
                return loaded
        except json.JSONDecodeError:
            pass
    bot = workspace_root() / experiment["bot"]
    if sha256(bot) != experiment["bot_sha256"]:
        raise RuntimeError(f"{bot} is not the preregistered binary")
    command = [
        str(bot),
        "eval",
        "--candidate",
        f"nnue:{side(match['candidate'])}",
        "--reference",
        f"nnue:{side(match['reference'])}",
        "--move-ms",
        str(match["move_ms"]),
        "--pairs",
        str(match["pairs"]),
        "--threads",
        str(match.get("concurrent", 8)),
        "--player-threads",
        str(match.get("player_threads", 1)),
        "--seed",
        str(match["seed"]),
        "--opening-plies",
        str(match.get("opening_plies", 8)),
        "--records",
        str(directory / f"{match['tag']}.games.jsonl"),
    ]
    with (directory / f"{match['tag']}.err").open("w", encoding="utf-8") as err:
        result = subprocess.run(command, cwd=workspace_root(), stdout=subprocess.PIPE, stderr=err, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"bot eval failed for {match['tag']} ({match['tag']}.err)")
    loaded = json.loads(result.stdout.decode("utf-8"))
    report.write_text(json.dumps(loaded, indent=2), encoding="utf-8")
    return loaded


# Verdict.


def summarise(match: dict, report: dict | None) -> dict:
    if report is None:
        return {"tag": match["tag"], "played": False}
    games = report["wins"] + report["draws"] + report["losses"]
    lo, hi = report["bootstrap_interval"]
    return {
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
        "valid": report.get("complete_pairs") == match["pairs"] and report.get("forfeits", 0) == 0,
    }


def verdict(experiment: dict, directory: Path) -> dict:
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
        matches[match["tag"]] = summarise(match, loaded)
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
            raise ValueError(f"unknown rule type {rule['type']!r}")
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
    if len(argv) < 2 or argv[0] not in ("run", "verdict"):
        raise SystemExit(__doc__)
    file = Path(argv[1]).resolve()
    experiment = json.loads(file.read_text(encoding="utf-8"))
    if experiment.get("schema") != 2:
        raise SystemExit("experiment.json must have schema 2")
    directory = file.parent
    if argv[0] == "verdict":
        print(json.dumps(verdict(experiment, directory), indent=2))
        return 0
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    pin(experiment.get("affinity", "16-31"))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    lock_acquire("experiment", f"{experiment['name']} ({only or 'all'})", experiment.get("affinity", "16-31"))
    try:
        if only in (None, "train"):
            for arm in experiment.get("arms", []):
                run_arm(arm)
                print(json.dumps({"event": "arm", "run": arm["run"], "epochs": epochs_done(arm["run"])}), flush=True)
        if only in (None, "match"):
            for match in experiment.get("matches", []):
                summary = summarise(match, play(experiment, directory, match))
                print(json.dumps({"event": "match", **summary}), flush=True)
    finally:
        lock_release("experiment")
    print(json.dumps({"event": "verdict", **{k: v for k, v in verdict(experiment, directory).items() if k == "rules"}}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
