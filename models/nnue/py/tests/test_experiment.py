"""The experiment runner end to end on tiny synthetic sets: an arm trained
with a stop, its retained checkpoint and endpoints, a relaunch that skips
finished steps, matches through a scripted `bot eval`, and the verdict."""

import json
import subprocess
import sys

import numpy as np
import pytest

from nnue import experiment, export, paths, train

from .test_train import synthetic


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    synthetic(paths.data_dir("a"), 1, 400)
    synthetic(paths.data_dir("b"), 2, 200)
    bot = tmp_path / "bot.exe"
    bot.write_bytes(b"scripted")
    (tmp_path / "runs" / "nnue_plan").mkdir(parents=True)
    return tmp_path, bot


def scripted(calls: list, outcomes: dict):
    """A subprocess.run that trains in-process and answers `bot eval` with a canned report;
    any other command goes to the real function."""
    real = subprocess.run

    def run(command, *args, **kwargs):
        if not isinstance(command, list) or len(command) < 2:
            return real(command, *args, **kwargs)
        if command[1:3] == ["-m", "nnue.train"]:
            calls.append(command)
            train.train(*train.parse(command[3:]))
            return subprocess.CompletedProcess(command, 0)
        if command[1] != "eval":
            return real(command, *args, **kwargs)
        calls.append(command)

        def game(pair, first_wins):
            winner = 0 if first_wins else None
            return {
                "winner": winner,
                "end": "Goal" if first_wins else "CaptureClock",
                "plies": 40 + pair,
                "moves": [f"m{pair}"],
            }

        if "--sprt" in command:
            journal = command[command.index("--sprt") + 1]
            tag = journal.split("/")[-1].split("\\")[-1].removesuffix(".jsonl")
            counts, stop = outcomes[tag]
            wins, losses = 2 * counts[4] + counts[3], 2 * counts[0] + counts[1]
            draws = 2 * counts[2] + counts[1] + counts[3]
            pairs = sum(counts)
            with open(journal, "w", encoding="utf-8") as f:
                f.write(json.dumps({"protocol": {}}) + "\n")
                for pair in range(2):
                    f.write(json.dumps({"pair": pair, "games": [game(pair, True), game(pair, False)]}) + "\n")
                f.write('{"pair": 2, "games": [')  # a torn last line
            sequential = {"counts": counts, "llr": 3.0 if stop == "accept" else -3.0, "stop_reason": stop}
        else:
            records = command[command.index("--records") + 1]
            tag = records.split("/")[-1].split("\\")[-1].removesuffix(".games.jsonl")
            pairs = int(command[command.index("--pairs") + 1])
            wins, draws, losses = outcomes[tag]
            with open(records, "w", encoding="utf-8") as f:
                for pair in range(pairs):
                    f.write(json.dumps(game(pair, True)) + "\n" + json.dumps(game(pair, False)) + "\n")
            sequential = None
        report = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "pairs": pairs,
            "complete_pairs": pairs,
            "forfeits": 0,
            "bootstrap_interval": [-0.05, 0.25],
            **({"sequential": sequential} if sequential else {}),
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(report).encode())

    return run


def test_runner_trains_with_a_stop_plays_and_judges(workspace, monkeypatch):
    root, bot = workspace
    directory = root / "runs" / "nnue_gauntlets" / "tiny"
    directory.mkdir(parents=True)
    spec = {
        "schema": 2,
        "name": "tiny",
        "bot": "bot.exe",
        "bot_sha256": experiment.sha256(bot),
        "affinity": "0-1",
        "arms": [
            {
                "run": "arm",
                "train": {
                    "config": {
                        "hidden": 32,
                        "batch": 64,
                        "steps_per_epoch": 5,
                        "epochs": 4,
                        "warmup_steps": 2,
                        "threads": 2,
                        "val_rows": 64,
                        "patience": 4,
                    },
                    "data": [["a", 0.7], ["b", 0.3]],
                },
                "stops": [2],
                "endpoints": {"epoch2": "latest@2", "epoch4": "latest", "best": "best"},
            }
        ],
        "network": "net.nnue",
        "matches": [
            {"tag": "m50", "candidate": "arm:epoch4", "reference": "arm:epoch2", "move_ms": 50, "pairs": 3, "seed": 1},
            {"tag": "m100", "candidate": "arm:epoch4", "reference": "arm:best", "move_ms": 100, "pairs": 2, "seed": 2},
            {"tag": "q", "candidate": "patch.exe", "reference": "base.exe", "move_ms": 50, "sprt": True, "seed": 3},
        ],
        "rules": [
            {"name": "gain", "type": "lower_bound_above", "threshold": 0.5, "matches": ["m50", "m100"]},
            {"name": "bar", "type": "score_at_least", "threshold": 0.6, "matches": ["m50"]},
            {"name": "patch", "type": "sprt_accept", "matches": ["q"]},
        ],
    }
    (root / "net.nnue").write_bytes(b"weights")
    (root / "patch.exe").write_bytes(b"search build")
    (root / "base.exe").write_bytes(b"baseline build")
    file = directory / "experiment.json"
    file.write_text(json.dumps(spec), encoding="utf-8")
    calls: list = []
    outcomes = {"m50": (4, 1, 1), "m100": (1, 2, 1), "q": ([10, 30, 60, 40, 20], "accept")}
    monkeypatch.setattr(experiment.subprocess, "run", scripted(calls, outcomes))
    monkeypatch.setattr(experiment, "pin", lambda mask: None)
    monkeypatch.setattr(sys, "argv", ["nnue.experiment"])
    assert experiment.main(["run", str(file)]) == 0

    run = paths.run_dir("arm")
    assert (run / "epoch2.pt").is_file() and (run / "epoch2.nnue").is_file() and (run / "epoch4.nnue").is_file()
    assert experiment.epochs_done("arm") == 4
    trains = [c for c in calls if c[1:3] == ["-m", "nnue.train"]]
    assert len(trains) == 2
    assert "--stop_epoch" in trains[0] and "--resume" in trains[1] and "--stop_epoch" not in trains[1]
    # The retained checkpoint is the epoch-2 model: its export equals an export of the trainer's state at that epoch.
    saved = json.loads((run / "epoch2.nnue.json").read_text(encoding="utf-8"))
    assert saved["epoch"] == 1
    net = export.read(run / "epoch2.nnue")
    assert net["hidden"] == 32
    final = export.read(run / "epoch4.nnue")
    assert not np.array_equal(net["weights"], final["weights"])

    verdict = json.loads((directory / "verdict.json").read_text(encoding="utf-8"))
    by_tag = {m["tag"]: m for m in verdict["matches"]}
    assert by_tag["m50"]["score"] == pytest.approx(4.5 / 6) and by_tag["m50"]["interval"] == [0.475, 0.625]
    assert by_tag["m100"]["valid"] and by_tag["m100"]["games"] == 4
    assert by_tag["m50"]["records"] == {
        "games": 6,
        "ends": {"Goal": 3, "CaptureClock": 3},
        "first_mover_score": 0.75,
        "distinct_openings": 3,
        "mean_plies": 41.0,
    }
    assert by_tag["q"]["records"]["games"] == 4 and by_tag["q"]["records"]["distinct_openings"] == 2
    rules = {r["name"]: r for r in verdict["rules"]}
    assert rules["gain"]["complete"] and rules["gain"]["met"] is False
    assert rules["bar"]["met"] is True
    assert rules["patch"]["complete"] and rules["patch"]["met"] is True
    assert by_tag["q"]["stop_reason"] == "accept" and by_tag["q"]["pairs"] == 160 and by_tag["q"]["valid"]
    sprt = next(c for c in calls if "--sprt" in c)
    assert "--pairs" not in sprt and "--records" not in sprt and "--resume" not in sprt
    assert sprt[sprt.index("--candidate") + 1].startswith("rpsi:") and sprt[sprt.index("--candidate") + 1].endswith(
        "net.nnue"
    )
    assert sprt[sprt.index("--candidate-network") + 1].endswith("net.nnue")
    assert sprt[sprt.index("--reference") + 1].startswith("rpsi:") and "--reference-network" in sprt
    spec["matches"][2]["reference"] = "arm:best"
    with pytest.raises(ValueError):
        experiment.play(spec, directory, spec["matches"][2] | {"tag": "mixed"})
    assert verdict["arms"][0]["endpoints"]["best"]["sha256"]
    assert not experiment.lock_path().exists()

    # A relaunch trains nothing and replays nothing.
    calls.clear()
    assert experiment.main(["run", str(file)]) == 0
    assert calls == []

    # A sequential test that has not stopped is resumed from its journal.
    report = directory / "q.json"
    unfinished = json.loads(report.read_text(encoding="utf-8"))
    unfinished["sequential"]["stop_reason"] = "running"
    report.write_text(json.dumps(unfinished), encoding="utf-8")
    assert experiment.main(["run", str(file), "--only", "match"]) == 0
    assert len(calls) == 1 and "--resume" in calls[0]


def test_lock_refuses_a_second_owner_and_a_wrong_release(workspace):
    experiment.lock_acquire("one", "phase", "0-1")
    with pytest.raises(RuntimeError, match="held by one"):
        experiment.lock_acquire("two", "other", "0-1")
    with pytest.raises(RuntimeError, match="not held by two"):
        experiment.lock_release("two")
    assert experiment.lock_release("one")["phase"] == "phase"
    assert experiment.lock_read() is None
