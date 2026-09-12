"""The experiment runner end to end on tiny synthetic sets: an arm trained
with a stop, its retained checkpoint and endpoints, a relaunch that skips
finished steps, matches through a scripted `bot eval`, their bindings and
timings, the audit and the verdict."""

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
    (tmp_path / "bot.exe").write_bytes(b"scripted")
    (tmp_path / "net.nnue").write_bytes(b"weights")
    (tmp_path / "patch.exe").write_bytes(b"search build")
    (tmp_path / "base.exe").write_bytes(b"baseline build")
    (tmp_path / "runs" / "nnue_plan").mkdir(parents=True)
    return tmp_path


def scripted_train(calls: list):
    """A subprocess.run that trains in-process; any other command goes to the real function."""
    real = subprocess.run

    def run(command, *args, **kwargs):
        if isinstance(command, list) and command[1:3] == ["-m", "nnue.train"]:
            calls.append(command)
            train.train(*train.parse(command[3:]))
            return subprocess.CompletedProcess(command, 0)
        return real(command, *args, **kwargs)

    return run


def game(pair: int, number: int, result: str, sides: dict) -> dict:
    """One recorded game from the candidate's result (W, D or L); the candidate is first in game 0."""
    mine = 0 if number == 0 else 1
    first, second = (sides["candidate"], sides["reference"])[:: 1 if mine == 0 else -1]
    winner = None if result == "D" else mine if result == "W" else 1 - mine
    end = "CaptureClock" if result == "D" else "Goal"
    return {"first": first, "second": second, "winner": winner, "end": end, "plies": 40 + pair, "moves": [f"m{pair}"]}


def scripted_launch(calls: list, outcomes: dict):
    """A `launch` that answers `bot eval` with a canned report and records or a journal consistent with it;
    a sequential test also streams events: pairs of eight seconds, eight at a time, ten seconds apart."""
    results = {0: "LL", 1: "LD", 2: "DD", 3: "WD", 4: "WW"}

    def launch(command, err):
        calls.append(command)
        sides = {role: command[command.index(f"--{role}") + 1] for role in ("candidate", "reference")}
        tally: dict = {"W": 0, "D": 0, "L": 0}
        events = []
        if "--sprt" in command:
            journal = command[command.index("--sprt") + 1]
            tag = journal.replace("\\", "/").split("/")[-1].removesuffix(".jsonl")
            counts, stop = outcomes[tag]
            pairs = sum(counts)
            with open(journal, "w", encoding="utf-8") as f:
                f.write(json.dumps({"protocol": {"test": {"batch": 16}}}) + "\n")
                pair = 0
                for k, count in enumerate(counts):
                    for _ in range(count):
                        f.write(
                            json.dumps({"pair": pair, "games": [game(pair, g, results[k][g], sides) for g in (0, 1)]})
                        )
                        f.write("\n")
                        for r in results[k]:
                            tally[r] += 1
                        at = 10.0 * (pair // 8)
                        events += [
                            (at, {"event": "start", "pair": pair, "game": 0}),
                            (at + 8, {"event": "end", "pair": pair, "game": 1}),
                        ]
                        pair += 1
                f.write(f'{{"pair": {pair}, "games": [')  # a torn last line
            sequential = {
                "counts": counts,
                "llr": 3.0 if stop == "accept" else -3.0,
                "stop_reason": stop,
                "protocol_digest": "digest",
                "intervals_descriptive_only": True,
            }
        else:
            records = command[command.index("--records") + 1]
            tag = records.replace("\\", "/").split("/")[-1].removesuffix(".games.jsonl")
            pairs = int(command[command.index("--pairs") + 1])
            sequence = "".join(r * n for r, n in zip("WDL", outcomes[tag], strict=True))
            assert len(sequence) == 2 * pairs
            with open(records, "w", encoding="utf-8") as f:
                for n, r in enumerate(sequence):
                    f.write(json.dumps(game(n // 2, n % 2, r, sides)) + "\n")
                    tally[r] += 1
            sequential = None
        report = {
            **sides,
            "move_ms": int(command[command.index("--move-ms") + 1]),
            "wins": tally["W"],
            "draws": tally["D"],
            "losses": tally["L"],
            "pairs": pairs,
            "complete_pairs": pairs,
            "forfeits": 0,
            "bootstrap_interval": [-0.05, 0.25],
            **({"sequential": sequential} if sequential else {}),
        }
        return report, events

    return launch


def manifest(root) -> dict:
    return {
        "schema": 2,
        "name": "tiny",
        "bot": "bot.exe",
        "network": "net.nnue",
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
        "matches": [
            {"tag": "m50", "candidate": "arm:epoch4", "reference": "arm:epoch2", "move_ms": 50, "pairs": 3, "seed": 1},
            {"tag": "m100", "candidate": "arm:epoch4", "reference": "arm:best", "move_ms": 100, "pairs": 2, "seed": 2},
            {"tag": "q", "candidate": "patch.exe", "reference": "base.exe", "move_ms": 50, "sprt": True, "seed": 3},
            {"tag": "qf", "candidate": "patch.exe", "reference": "base.exe", "move_ms": 100, "pairs": 2, "seed": 4},
        ],
        "rules": [
            {"name": "gain", "type": "lower_bound_above", "threshold": 0.5, "matches": ["m50", "m100"]},
            {"name": "bar", "type": "score_at_least", "threshold": 0.6, "matches": ["m50"]},
            {"name": "patch", "type": "sprt_accept", "matches": ["q"]},
        ],
    }


def test_runner_trains_with_a_stop_plays_and_judges(workspace, monkeypatch):
    root = workspace
    directory = root / "runs" / "nnue_gauntlets" / "tiny"
    directory.mkdir(parents=True)
    spec = manifest(root)
    spec["identities"] = experiment.identities(spec)
    assert set(spec["identities"]) == {"bot.exe", "net.nnue", "patch.exe", "base.exe"} | {
        f"runs/nnue_data/{name}/provenance.json" for name in "ab"
    }
    file = directory / "experiment.json"
    file.write_text(json.dumps(spec), encoding="utf-8")
    calls: list = []
    outcomes = {"m50": (4, 1, 1), "m100": (1, 2, 1), "q": ([10, 30, 60, 40, 20], "accept"), "qf": (2, 1, 1)}
    monkeypatch.setattr(experiment.subprocess, "run", scripted_train(calls))
    monkeypatch.setattr(experiment, "launch", scripted_launch(calls, outcomes))
    monkeypatch.setattr(experiment, "nice", lambda mask: None)
    monkeypatch.setattr(sys, "argv", ["nnue.experiment"])
    assert experiment.main(["run", str(file)]) == 0

    run = paths.run_dir("arm")
    assert (run / "epoch2.pt").is_file() and (run / "epoch2.nnue").is_file() and (run / "epoch4.nnue").is_file()
    assert experiment.epochs_done("arm") == 4 and experiment.checkpoint_epochs(run / "epoch2.pt") == 2
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
        "consistent": True,
        "problems": [],
        "ends": {"Goal": 5, "CaptureClock": 1},
        "first_mover_score": pytest.approx(3.5 / 6),
        "distinct_openings": 3,
        "mean_plies": 41.0,
    }
    q = by_tag["q"]
    assert q["records"]["games"] == 320 and q["records"]["distinct_openings"] == 160 and q["records"]["consistent"]
    assert q["stop_reason"] == "accept" and q["pairs"] == 160 and q["valid"] and q["counts"] == [10, 30, 60, 40, 20]
    assert q["protocol_digest"] == "digest" and q["interval_descriptive_only"] and q["error"] is None
    rules = {r["name"]: r for r in verdict["rules"]}
    assert rules["gain"]["complete"] and rules["gain"]["met"] is False
    assert rules["bar"]["met"] is True
    assert rules["patch"]["complete"] and rules["patch"]["met"] is True

    # Search builds play the shared network through C1's rpsi seats; only the sequential test names the network.
    sprt = next(c for c in calls if "--sprt" in c)
    child = f"rpsi:{root / 'patch.exe'} rpsi --player nnue:{root / 'net.nnue'} --threads 1"
    assert sprt[sprt.index("--candidate") + 1] == child and sprt[-1] == "--stream"
    assert "--pairs" not in sprt and "--records" not in sprt and "--resume" not in sprt
    assert sprt[sprt.index("--candidate-network") + 1] == str(root / "net.nnue") and "--reference-network" in sprt
    fixed = next(c for c in calls if "--records" in c and "patch.exe" in c[c.index("--candidate") + 1])
    assert fixed[fixed.index("--candidate") + 1] == child and "--candidate-network" not in fixed
    timing = json.loads((directory / "q.timing.json").read_text(encoding="utf-8"))
    assert len(timing) == 1 and timing[0]["initial_pairs"] == 0 and timing[0]["final_pairs"] == 160
    assert timing[0]["observed_pairs"] == 160 and len(timing[0]["batches"]) == 10
    assert all(b["complete"] for b in timing[0]["batches"])
    assert timing[0]["idle_share"] == pytest.approx(1 - 128 / (8 * 18))
    bound = json.loads((directory / "m50.match.json").read_text(encoding="utf-8"))
    assert bound["match"] == spec["matches"][0] and set(bound["identities"]) == {
        "bot.exe",
        "runs/arm/epoch4.nnue",
        "runs/arm/epoch2.nnue",
    }
    with pytest.raises(ValueError):
        experiment.play(spec, directory, spec["matches"][2] | {"tag": "mixed", "reference": "arm:best"})
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
    assert len(json.loads((directory / "q.timing.json").read_text(encoding="utf-8"))) == 2

    # A match whose manifest entry changed is played again; the old report is not reused.
    calls.clear()
    spec["matches"][1]["seed"] = 9
    file.write_text(json.dumps(spec), encoding="utf-8")
    assert experiment.main(["run", str(file), "--only", "match"]) == 0
    assert len(calls) == 1 and "--records" in calls[0] and calls[0][calls[0].index("--seed") + 1] == "9"

    # Records that disagree with the report invalidate the match and its rule.
    records = directory / "m50.games.jsonl"
    records.write_text("".join(records.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]), encoding="utf-8")
    verdict = experiment.verdict(spec, directory)
    m50 = next(m for m in verdict["matches"] if m["tag"] == "m50")
    assert not m50["valid"] and m50["problems"] == ["5 recorded games for 3 complete pairs"]
    assert next(r for r in verdict["rules"] if r["name"] == "gain")["complete"] is False

    # A changed fixed input is refused.
    (root / "bot.exe").write_bytes(b"rebuilt")
    with pytest.raises(RuntimeError, match="differs"):
        experiment.main(["run", str(file), "--only", "match"])


def test_manifest_validation(workspace):
    spec = manifest(workspace)
    experiment.validate(spec)
    for bad in (
        {"rules": [{"name": "x", "type": "lower_bound_above", "threshold": 0.5, "matches": ["q"]}]},
        {"rules": [{"name": "x", "type": "sprt_accept", "matches": ["m50"]}]},
        {"rules": [{"name": "x", "type": "sprt_accept", "matches": ["nothing"]}]},
        {"rules": [{"name": "x", "type": "elo", "matches": ["m50"]}]},
        {"matches": [{**spec["matches"][2], "pairs": 10}]},
        {"matches": [spec["matches"][0], spec["matches"][0]]},
    ):
        with pytest.raises(ValueError):
            experiment.validate({**spec, **bad})


def test_journal_pairs_reject_malformed_lines(tmp_path):
    journal = tmp_path / "q.jsonl"
    pair = json.dumps({"pair": 0, "games": [game(0, 0, "W", {"candidate": "c", "reference": "r"})] * 2})
    journal.write_text('{"protocol": {}}\n' + pair + "\n" + pair.replace('"pair": 0', '"pair": 1') + "\n", "utf-8")
    assert len(experiment.journal_pairs(journal)) == 2
    journal.write_text('{"protocol": {}}\n' + pair + "\n{torn", "utf-8")
    assert len(experiment.journal_pairs(journal)) == 1
    for text in ('{"protocol": {}}\n{torn}\n' + pair + "\n", '{"other": 1}\n' + pair + "\n", pair + "\n"):
        journal.write_text(text, "utf-8")
        with pytest.raises(ValueError):
            experiment.journal_pairs(journal)


def test_lock_refuses_a_second_owner_and_a_wrong_release(workspace):
    experiment.lock_acquire("one", "phase", "0-1")
    with pytest.raises(RuntimeError, match="held by one"):
        experiment.lock_acquire("two", "other", "0-1")
    with pytest.raises(RuntimeError, match="not held by two"):
        experiment.lock_release("two")
    with pytest.raises(RuntimeError, match="pid 1"):
        experiment.lock_release("one", pid=1)
    assert experiment.lock_release("one")["phase"] == "phase"
    assert experiment.lock_read() is None
