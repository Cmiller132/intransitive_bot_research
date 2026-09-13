"""The experiment runner end to end on tiny synthetic sets: an arm trained
with a stop, its retained checkpoint and endpoints, a relaunch that skips
finished steps, matches through a scripted `bot eval`, their bindings and
timings, the audit and the verdict."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import pytest

from nnue import experiment, export, paths, train

from .test_train import synthetic


@pytest.fixture(autouse=True)
def fresh_cleanup_state():
    experiment.SURVIVORS.clear()
    experiment.CLEANUP_FAILURES.clear()
    yield
    experiment.SURVIVORS.clear()
    experiment.CLEANUP_FAILURES.clear()


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


def header_of(command: list, sides: dict) -> dict:
    """A journal header shaped like C1's: the protocol with its settings and the artifact provenance, and its digest."""

    def artifact(path: str) -> dict:
        return {"path": path, "sha256": paths.sha256(Path(path))}

    def identity(role: str) -> dict:
        kind, value = sides[role].split(":", 1)
        if kind == "rpsi":
            binary, network = value.split()[0], command[command.index(f"--{role}-network") + 1]
        else:
            binary, network = command[0], value
        return {"spec": sides[role], "binary": artifact(binary), "network": artifact(network)}

    def option(name: str) -> int:
        return int(command[command.index(name) + 1])

    protocol = {
        "schema": 1,
        "test": {"method": "pentanomial_expectation_mle", "batch": 16, "first_check": 128, "cap": 3008},
        "settings": {
            "capture_clock": 200,
            "pairs": 3008,
            "sims": 0,
            "move_ms": option("--move-ms"),
            "reference_move_ms": None,
            "reference_sims": None,
            "workers": option("--threads"),
            "seed": option("--seed"),
            "opening_plies": option("--opening-plies"),
            "stream": True,
        },
        "openings_sha256": "0" * 64,
        "provenance": {
            "coordinator": artifact(command[0]),
            "version": "test",
            "player_threads": option("--player-threads"),
            "logical_cpus": [0, 1],
            "candidate": identity("candidate"),
            "reference": identity("reference"),
        },
    }
    digest = hashlib.sha256(json.dumps(protocol, separators=(",", ":")).encode()).hexdigest()
    return {"protocol": protocol, "protocol_digest": digest}


def scripted_launch(calls: list, outcomes: dict):
    """A `launch` that answers `bot eval` with a canned report and records or a journal consistent with it;
    a sequential test also streams events: pairs of eight seconds, eight at a time, ten seconds apart. A
    resumed journal is replayed as C1 does: read back, nothing played, reported again."""
    results = {0: "LL", 1: "LD", 2: "DD", 3: "WD", 4: "WW"}

    def launch(command, err, events=None):
        calls.append(command)
        sides = {role: command[command.index(f"--{role}") + 1] for role in ("candidate", "reference")}
        tally: dict = {"W": 0, "D": 0, "L": 0}
        events = [] if events is None else events
        if "--resume" in command:
            journal = Path(command[command.index("--sprt") + 1])
            tag = journal.name.removesuffix(".jsonl")
            header, replayed = experiment.journal_pairs(journal)
            counts = [0] * 5
            for pair in replayed:
                counts[experiment.pair_points(pair)] += 1
                for played, mine in zip(pair, (0, 1), strict=True):
                    tally[experiment.game_result(played, mine)[0].upper()] += 1
            pairs = len(replayed)
            sequential = {
                **header,
                "counts": counts,
                "llr": 3.0 if outcomes[tag][1] == "accept" else -3.0,
                "stop_reason": outcomes[tag][1],
                "intervals_descriptive_only": True,
            }
        elif "--sprt" in command:
            journal = command[command.index("--sprt") + 1]
            tag = journal.replace("\\", "/").split("/")[-1].removesuffix(".jsonl")
            counts, stop = outcomes[tag]
            pairs = sum(counts)
            header = header_of(command, sides)
            with open(journal, "w", encoding="utf-8") as f:
                f.write(json.dumps(header) + "\n")
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
                **header,
                "counts": counts,
                "llr": 3.0 if stop == "accept" else -3.0,
                "stop_reason": stop,
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
        return report, events, 0

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
    assert len(q["protocol_digest"]) == 64 and q["interval_descriptive_only"] and q["error"] is None
    assert q["protocol"]["settings"]["seed"] == 3 and q["protocol"]["provenance"]["player_threads"] == 1
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
    (invocation,) = timing["invocations"]
    assert invocation["completed"] and invocation["initial_pairs"] == 0 and invocation["committed_pairs"] == 160
    assert invocation["reported_pairs"] == 160 and invocation["launched_utc"].endswith("Z")
    assert invocation["observed_pairs"] == 160 and len(invocation["batches"]) == 10
    assert all(b["complete"] for b in invocation["batches"])
    assert invocation["idle_share"] == pytest.approx(1 - 128 / (8 * 18))
    assert timing["committed_pairs"] == 160 and timing["active_seconds"] == invocation["play_seconds"]
    bound = json.loads((directory / "m50.match.json").read_text(encoding="utf-8"))
    assert bound["match"] == spec["matches"][0] and set(bound["identities"]) == {
        "bot.exe",
        "runs/arm/epoch4.nnue",
        "runs/arm/epoch2.nnue",
    }
    assert bound["affinity"] == "0-1" and set(bound["outputs"]) == {"report", "records"}
    assert bound["outputs"]["records"] == paths.sha256(directory / "m50.games.jsonl")
    with pytest.raises(ValueError):
        experiment.play(spec, directory, spec["matches"][2] | {"tag": "mixed", "reference": "arm:best"})
    assert verdict["arms"][0]["endpoints"]["best"]["sha256"]
    assert all(e["problem"] is None for e in verdict["arms"][0]["endpoints"].values())
    assert not experiment.lock_path().exists()

    # A relaunch trains nothing and plays nothing; the verdict's C1 replay of the finished journal is its only call.
    calls.clear()
    assert experiment.main(["run", str(file)]) == 0
    assert len(calls) == 1 and "--resume" in calls[0] and "--stream" in calls[0]

    # A sequential test that has not stopped is resumed from its journal.
    calls.clear()
    report = directory / "q.json"
    unfinished = json.loads(report.read_text(encoding="utf-8"))
    unfinished["sequential"]["stop_reason"] = "running"
    report.write_text(json.dumps(unfinished), encoding="utf-8")
    assert experiment.main(["run", str(file), "--only", "match"]) == 0
    assert len(calls) == 2 and all("--resume" in c for c in calls)  # the resumed test, then the verdict's replay
    timing = json.loads((directory / "q.timing.json").read_text(encoding="utf-8"))
    assert len(timing["invocations"]) == 2 and timing["invocations"][1]["play_seconds"] is None
    assert timing["committed_pairs"] == 160

    # A match whose manifest entry changed is played again; the old report is not reused.
    calls.clear()
    spec["matches"][1]["seed"] = 9
    file.write_text(json.dumps(spec), encoding="utf-8")
    assert experiment.main(["run", str(file), "--only", "match"]) == 0
    assert len(calls) == 2 and "--records" in calls[0] and calls[0][calls[0].index("--seed") + 1] == "9"

    # Records that disagree with the report invalidate the match and its rule.
    records = directory / "m50.games.jsonl"
    records.write_text("".join(records.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]), encoding="utf-8")
    verdict = experiment.verdict(spec, directory)
    m50 = next(m for m in verdict["matches"] if m["tag"] == "m50")
    assert not m50["valid"] and "5 recorded games for 3 complete pairs" in m50["problems"]
    assert "the report, journal or records changed after the match was bound" in m50["problems"]
    assert next(r for r in verdict["rules"] if r["name"] == "gain")["complete"] is False

    # An endpoint that cannot stand for its arm invalidates the matches that used it.
    sidecar = run / "epoch2.nnue.json"
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar.write_text(json.dumps({**info, "epoch": 3}), encoding="utf-8")
    m50 = next(m for m in experiment.verdict(spec, directory, replays=False)["matches"] if m["tag"] == "m50")
    assert "reference endpoint: epoch 3, the endpoint needs 1" in m50["problems"]
    sidecar.unlink()  # a missing sidecar is a broken endpoint too, not a silently accepted one
    judged = experiment.verdict(spec, directory, replays=False)
    m50 = next(m for m in judged["matches"] if m["tag"] == "m50")
    assert "reference endpoint: no sidecar" in m50["problems"] and judged["arms"][0]["endpoints"]["epoch2"]["exists"]
    sidecar.write_text(json.dumps(info), encoding="utf-8")

    # A cached report that C1's replay of the journal contradicts is invalid; so is one bound to another affinity.
    q_report, q_sidecar = directory / "q.json", directory / "q.match.json"
    kept, kept_sidecar = q_report.read_text(encoding="utf-8"), q_sidecar.read_text(encoding="utf-8")
    q_report.write_text(kept.replace('"llr": 3.0', '"llr": 2.0'), encoding="utf-8")
    stored = json.loads(kept_sidecar)  # bound as written, so only C1's replay can catch it
    stored["outputs"]["report"] = paths.sha256(q_report)
    q_sidecar.write_text(json.dumps(stored), encoding="utf-8")
    q = next(m for m in experiment.verdict(spec, directory)["matches"] if m["tag"] == "q")
    assert not q["valid"] and "C1's replay gives another llr" in q["problems"]
    q_report.write_text(kept, encoding="utf-8")
    q_sidecar.write_text(kept_sidecar, encoding="utf-8")
    assert experiment.bound({**spec, "affinity": "2-3"}, directory, spec["matches"][2]) == (
        "the report is not bound to this manifest entry, affinity and these files"
    )

    # A changed fixed input is refused.
    (root / "bot.exe").write_bytes(b"rebuilt")
    with pytest.raises(RuntimeError, match="differs"):
        experiment.main(["run", str(file), "--only", "match"])


def test_a_cuda_arm_is_given_the_gpu_and_a_cpu_run_is_not():
    cpu = {"arms": [{"run": "a", "train": {"config": {"device": "cpu"}}}, {"run": "b"}]}
    assert experiment.visible_devices(cpu) == "-1" and experiment.visible_devices({}) == "-1"
    assert experiment.visible_devices({"arms": [{"run": "a", "train": {"config": {"device": "cuda"}}}]}) == "0"


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
    header = '{"protocol": {}, "protocol_digest": "d"}\n'
    pair = json.dumps({"pair": 0, "games": [game(0, 0, "W", {"candidate": "c", "reference": "r"})] * 2})
    journal.write_text(header + pair + "\n" + pair.replace('"pair": 0', '"pair": 1') + "\n", "utf-8")
    assert experiment.journal_pairs(journal)[0]["protocol_digest"] == "d"
    assert len(experiment.journal_pairs(journal)[1]) == 2
    journal.write_text(header + pair + "\n{torn", "utf-8")
    assert len(experiment.journal_pairs(journal)[1]) == 1
    for text in (
        header + "{torn}\n" + pair + "\n",
        header + "null\n",
        header + "[1, 2]\n",
        '{"other": 1}\n' + pair + "\n",
        '{"protocol": {}}\n' + pair + "\n",
        pair + "\n",
    ):
        journal.write_text(text, "utf-8")
        with pytest.raises(ValueError):
            experiment.journal_pairs(journal)


def sequential_fixture(workspace, monkeypatch) -> tuple[dict, Path, dict, dict]:
    """The manifest, its directory, the sequential match `q` played through the scripted `bot eval` and its report."""
    spec = manifest(workspace)
    spec["identities"] = experiment.identities(spec)
    directory = workspace / "runs" / "nnue_gauntlets" / "tiny"
    directory.mkdir(parents=True)
    (directory / "experiment.json").write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(experiment, "launch", scripted_launch([], {"q": ([2, 3, 5, 4, 2], "accept"), "qf": (2, 1, 1)}))
    monkeypatch.setattr(experiment, "nice", lambda mask: None)
    match = spec["matches"][2]
    return spec, directory, match, experiment.play(spec, directory, match)


def test_audit_rejects_every_mutation_of_a_journal_or_its_report(workspace, monkeypatch):
    spec, directory, match, report = sequential_fixture(workspace, monkeypatch)
    journal = directory / "q.jsonl"
    original = journal.read_text(encoding="utf-8")
    assert experiment.audit(match, report, directory, spec) == {
        "games": 32,
        "consistent": True,
        "problems": [],
        "ends": {"Goal": 15, "CaptureClock": 17},
        "first_mover_score": pytest.approx(16.5 / 32),
        "distinct_openings": 16,
        "mean_plies": 47.5,
    }

    def swap_results(lines, report):
        report["wins"], report["losses"] = report["losses"], report["wins"]
        return "the recorded games give other results than the report"

    def other_counts(lines, report):
        report["sequential"]["counts"] = report["sequential"]["counts"][::-1]
        return "the recorded pairs give other pentanomial counts than the report"

    def other_digest(lines, report):
        lines[0]["protocol_digest"] = "x"
        return "the journal header's protocol differs from the report's"

    def other_seed(lines, report):
        lines[0]["protocol"]["settings"]["seed"] = 99
        report["sequential"]["protocol"]["settings"]["seed"] = 99
        return "protocol seed is 99, the match says 3"

    def other_binary(lines, report):
        for protocol in (lines[0]["protocol"], report["sequential"]["protocol"]):
            protocol["provenance"]["candidate"]["binary"]["sha256"] = "0" * 64
        return "the journal's candidate binary is not the pinned file"

    def other_coordinator(lines, report):
        for protocol in (lines[0]["protocol"], report["sequential"]["protocol"]):
            protocol["provenance"]["coordinator"]["sha256"] = "0" * 64
        return "the journal's coordinator is not the pinned bot"

    def swapped_seats(lines, report):
        g = lines[1]["games"][0]
        g["first"], g["second"] = g["second"], g["first"]
        return "pair 0: game 0 has other seats than the pair's order"

    def other_opening(lines, report):
        lines[1]["games"][1]["moves"] = ["elsewhere"]
        return "pair 0: the two games have different openings"

    def forfeited(lines, report):
        lines[1]["games"][0]["end"] = "Forfeit"
        return "pair 0: game 0 ended by Forfeit"

    def ply_capped(lines, report):
        lines[1]["games"][1]["end"] = "PlyCap"
        return "pair 0: game 1 ended by PlyCap"

    def reported_forfeits(lines, report):
        report["forfeits"] = 1
        return "1 forfeits"

    def accepted_with_error(lines, report):
        report["sequential"]["error"] = "a seat died"
        return "a completed test reports an error: a seat died"

    def malformed_moves(lines, report):
        lines[1]["games"][0]["moves"] = 5
        return "malformed records"

    def malformed_entry(lines, report):
        lines[1] = [1, 2]
        return "malformed records"

    def missing_pair(lines, report):
        del lines[-1]
        return "15 recorded pairs, the report says 16 of which 16 complete"

    for mutate in (
        swap_results,
        other_counts,
        other_digest,
        other_seed,
        other_binary,
        other_coordinator,
        swapped_seats,
        other_opening,
        forfeited,
        ply_capped,
        reported_forfeits,
        accepted_with_error,
        malformed_moves,
        malformed_entry,
        missing_pair,
    ):
        lines = [json.loads(line) for line in original.split("\n")[:-1] if line]  # the torn last line dropped
        mutated = json.loads(json.dumps(report))
        expected = mutate(lines, mutated)
        journal.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
        audit = experiment.audit(match, mutated, directory, spec)
        assert not audit["consistent"] and any(expected in p for p in audit["problems"]), (mutate.__name__, audit)
    journal.write_text(original, encoding="utf-8")
    assert experiment.audit(match, report, directory, spec)["consistent"]


def test_fixed_count_audit_reconciles_pairs_and_openings(workspace, monkeypatch):
    spec, directory, _, _ = sequential_fixture(workspace, monkeypatch)
    match = spec["matches"][3]
    report = experiment.play(spec, directory, match)
    assert experiment.audit(match, report, directory, spec)["consistent"]
    records = directory / "qf.games.jsonl"
    games = [json.loads(line) for line in records.read_text(encoding="utf-8").split("\n") if line]
    games[1]["first"], games[1]["second"] = games[1]["second"], games[1]["first"]
    games[3]["moves"] = ["elsewhere"]
    records.write_text("".join(json.dumps(g) + "\n" for g in games), encoding="utf-8")
    problems = experiment.audit(match, report, directory, spec)["problems"]
    assert "pair 0: game 1 has other seats than the pair's order" in problems
    assert "pair 1: the two games have different openings" in problems


def test_endpoint_reuse_is_validated_against_the_arm(workspace):
    arm = {
        "run": "arm2",
        "train": {"config": {"hidden": 32, "epochs": 4}, "init": "net.nnue", "data": [["a", 1.0]]},
        "endpoints": {"epoch2": "latest@2"},
    }
    file = paths.run_dir("arm2") / "epoch2.nnue"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"weights")
    sidecar = file.with_suffix(".nnue.json")
    good = {
        "sha256": paths.sha256(file),
        "epoch": 1,
        "config": {"hidden": 32, "epochs": 4, "batch": 64},
        "data": [["a", 1.0]],
        "datasets": {"a": paths.sha256(paths.data_dir("a") / "provenance.json")},
        "init_sha256": paths.sha256(workspace / "net.nnue"),
    }
    sidecar.write_text(json.dumps(good), encoding="utf-8")
    assert experiment.endpoint_problem(arm, "epoch2") is None
    experiment.run_arm({}, arm)  # nothing to do
    for change, expected in (
        ({"epoch": 2}, "epoch 2, the endpoint needs 1"),
        ({"config": {**good["config"], "hidden": 64}}, "config hidden is 64, the arm says 32"),
        ({"data": [["a", 0.5], ["b", 0.5]]}, "trained on other data shares"),
        ({"datasets": {"a": "0" * 64}}, "trained on other datasets"),
        ({"init_sha256": None}, "started from another init"),
    ):
        sidecar.write_text(json.dumps({**good, **change}), encoding="utf-8")
        assert experiment.endpoint_problem(arm, "epoch2") == expected
        with pytest.raises(RuntimeError, match="cannot stand for this arm"):
            experiment.run_arm({}, arm)
    sidecar.write_text(json.dumps(good), encoding="utf-8")
    file.write_bytes(b"other weights")
    assert experiment.endpoint_problem(arm, "epoch2") == "the file differs from its sidecar's hash"
    sidecar.write_text(json.dumps({k: v for k, v in good.items() if k != "data"}), encoding="utf-8")
    assert experiment.endpoint_problem(arm, "epoch2") == "the sidecar lacks data"
    # An arm that names an existing run binds to nothing but the file, its epoch and its config.
    file.write_bytes(b"weights")
    existing = {"run": "arm2", "endpoints": {"epoch2": "latest@2"}}
    assert experiment.endpoint_problem(existing, "epoch2") is None
    sidecar.write_text(json.dumps({k: v for k, v in good.items() if k != "epoch"}), encoding="utf-8")
    assert experiment.endpoint_problem(existing, "epoch2") == "the sidecar lacks epoch"


def test_the_verdict_never_replays_a_missing_or_truncated_journal(workspace, monkeypatch):
    spec, directory, match, report = sequential_fixture(workspace, monkeypatch)
    calls: list = []
    monkeypatch.setattr(experiment, "launch", lambda command, err, events=None: calls.append(command))
    journal = directory / "q.jsonl"
    kept = journal.read_text(encoding="utf-8")
    journal.write_text("".join(kept.splitlines(keepends=True)[:9]), encoding="utf-8")  # a running prefix
    q = next(m for m in experiment.verdict(spec, directory)["matches"] if m["tag"] == "q")
    assert not q["valid"] and calls == []
    assert any("8 recorded pairs, the report says 16" in p for p in q["problems"])
    journal.unlink()
    q = next(m for m in experiment.verdict(spec, directory)["matches"] if m["tag"] == "q")
    assert not q["valid"] and calls == []
    with pytest.raises(RuntimeError, match="no journal"):
        experiment.replay(spec, directory, match, report)
    assert calls == []


def test_an_interrupted_invocation_keeps_its_timing_and_the_resume_continues_it(workspace, monkeypatch):
    spec = manifest(workspace)
    spec["identities"] = experiment.identities(spec)
    directory = workspace / "runs" / "nnue_gauntlets" / "tiny"
    directory.mkdir(parents=True)
    monkeypatch.setattr(experiment, "nice", lambda mask: None)
    match = spec["matches"][2]
    scripted = scripted_launch([], {"q": ([2, 3, 5, 4, 2], "accept")})

    def interrupted(command, err, events=None):
        """Sixteen pairs committed and streamed, a seventeenth started, then the coordinator is interrupted."""
        scripted(command, err)  # writes the full journal; keep its first batch only
        journal = Path(command[command.index("--sprt") + 1])
        journal.write_text("".join(journal.read_text(encoding="utf-8").splitlines(keepends=True)[:17]), "utf-8")
        base = time.monotonic()
        for pair in range(16):
            events.append((base + 10.0 * (pair // 8), {"event": "start", "pair": pair, "game": 0}))
            events.append((base + 10.0 * (pair // 8) + 8, {"event": "end", "pair": pair, "game": 1}))
        events.append((base + 20.0, {"event": "start", "pair": 16, "game": 0}))
        raise RuntimeError("interrupted")

    monkeypatch.setattr(experiment, "launch", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        experiment.play(spec, directory, match)
    timing = json.loads((directory / "q.timing.json").read_text(encoding="utf-8"))
    (first,) = timing["invocations"]
    assert first["interrupted"] and not first["completed"] and first["returncode"] is None
    assert first["initial_pairs"] == 0 and first["committed_pairs"] == 16 and first["reported_pairs"] is None
    assert first["observed_pairs"] == 16 and len(first["batches"]) == 1 and first["batches"][0]["complete"]
    assert first["active_seconds"] is not None and first["finish_seconds"] is None
    assert timing["committed_pairs"] == 16 and timing["process_seconds"] > 0
    monkeypatch.setattr(experiment, "launch", scripted)
    resumed = experiment.play(spec, directory, match)
    assert resumed["pairs"] == 16 and resumed["sequential"]["stop_reason"] == "accept"
    timing = json.loads((directory / "q.timing.json").read_text(encoding="utf-8"))
    assert len(timing["invocations"]) == 2 and timing["invocations"][1]["initial_pairs"] == 16
    assert timing["committed_pairs"] == 16 and timing["unknown_invocations"] == 0

    # An invocation whose journal cannot be read keeps its timing with an unknown count; the aggregate says so.
    def unreadable(command, err, events=None):
        journal = Path(command[command.index("--sprt") + 1])
        journal.write_text("{torn", encoding="utf-8")  # interrupted while the header was being written
        raise RuntimeError("interrupted again")

    monkeypatch.setattr(experiment, "launch", unreadable)
    (directory / "q.json").unlink()
    with pytest.raises(RuntimeError, match="interrupted again"):
        experiment.play(spec, directory, match)
    timing = json.loads((directory / "q.timing.json").read_text(encoding="utf-8"))
    assert timing["invocations"][2]["committed_pairs"] is None and timing["invocations"][2]["initial_pairs"] == 16
    assert timing["committed_pairs"] is None and timing["known_committed_pairs"] == 16
    assert timing["unknown_invocations"] == 1 and timing["pairs_per_second"] is None


def test_launch_reaps_the_seats_of_a_coordinator_that_exits_abruptly(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    pid_file = tmp_path / "orphan"
    script = (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "sys.exit(3)\n"
    )
    with (tmp_path / "err").open("w") as err:
        report, events, code = experiment.launch([sys.executable, "-c", script], err)
    assert (report, events, code) == (None, [], 3)
    orphan = int(pid_file.read_text())
    assert not psutil.pid_exists(orphan) or psutil.Process(orphan).status() == psutil.STATUS_DEAD
    assert orphan not in experiment.SURVIVORS


def test_launch_raises_reader_errors_and_kills_a_coordinator_that_will_not_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    monkeypatch.setattr(experiment, "EXIT_GRACE", 1.0)
    pid_file = tmp_path / "pid"
    garbage = (
        "import os, sys, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "sys.stdout.buffer.write(b'\\xff\\xfe\\n'); sys.stdout.flush(); time.sleep(60)\n"
    )
    with (tmp_path / "err").open("w") as err, pytest.raises(RuntimeError, match="could not be read"):
        experiment.launch([sys.executable, "-c", garbage], err)
    assert not experiment.running(int(pid_file.read_text())) and not experiment.CLEANUP_FAILURES
    silent = (
        "import os, sys, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "print('{\"pairs\": 0}', flush=True); os.close(1); time.sleep(60)\n"
    )
    with (tmp_path / "err").open("w") as err, pytest.raises(RuntimeError, match="did not exit"):
        experiment.launch([sys.executable, "-c", silent], err)
    assert not experiment.running(int(pid_file.read_text())) and not experiment.CLEANUP_FAILURES


def test_a_cleanup_that_cannot_be_proved_stops_the_run_and_keeps_the_reservation(workspace, monkeypatch):
    spec = manifest(workspace)
    monkeypatch.setattr(experiment, "nice", lambda mask: None)
    command = [sys.executable, "-c", "print('{\"pairs\": 0}')"]
    real_job = experiment.Job

    def failing(self):
        raise OSError(5, "query failed")

    def reserved_launch(broken: str | None):
        original = getattr(experiment.Job, broken) if broken else None
        if broken:
            setattr(experiment.Job, broken, failing)
        try:
            with (workspace / "err").open("w") as err:
                return experiment.reserved(spec, "test", lambda: experiment.launch(command, err))
        finally:
            if broken:
                setattr(experiment.Job, broken, original)

    for step in ("members", "terminate", "close"):
        with pytest.raises(RuntimeError, match="cleanup could not be proved"):
            reserved_launch(step)
        assert experiment.CLEANUP_FAILURES == ["bot eval cleanup: [Errno 5] query failed"], step
        held = experiment.lock_read()
        assert held and held["owner"] == "experiment" and held["pid"] == os.getpid(), step
        # Nothing is launched again while the failure stands: no job is even created.
        monkeypatch.setattr(experiment, "Job", lambda: pytest.fail("launched after an unproved cleanup"))
        with (workspace / "err").open("w") as err, pytest.raises(RuntimeError, match="not launched"):
            experiment.launch(command, err)
        monkeypatch.setattr(experiment, "Job", real_job)
        experiment.lock_release("experiment", os.getpid())
        experiment.CLEANUP_FAILURES.clear()
    assert reserved_launch(None) == ({"pairs": 0}, [], 0)
    assert not experiment.CLEANUP_FAILURES and experiment.lock_read() is None

    # Through `run`: the sequential match's report is not cached, its timing is kept, the final verdict
    # (whose replay would be a second launch) is not reached, the lock is kept.
    spec["arms"], spec["rules"], spec["matches"] = [], [], [spec["matches"][2]]
    spec["identities"] = experiment.identities(spec)
    directory = workspace / "runs" / "nnue_gauntlets" / "tiny"
    directory.mkdir(parents=True)
    file = directory / "experiment.json"
    file.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(experiment, "command_of", lambda e, d, m: command)
    launches = []
    real_launch = experiment.launch
    monkeypatch.setattr(experiment, "launch", lambda *a, **k: (launches.append(a), real_launch(*a, **k))[1])
    monkeypatch.setattr(experiment.Job, "members", failing)
    with pytest.raises(RuntimeError, match="cleanup could not be proved"):
        experiment.main(["run", str(file), "--only", "match"])
    assert len(launches) == 1
    assert not (directory / "q.json").exists() and not (directory / "q.match.json").exists()
    timing = json.loads((directory / "q.timing.json").read_text(encoding="utf-8"))
    assert timing["invocations"][0]["interrupted"] and timing["invocations"][0]["reported_pairs"] is None
    held = experiment.lock_read()
    assert held and held["owner"] == "experiment" and experiment.CLEANUP_FAILURES
    experiment.lock_release("experiment", os.getpid())


def test_launch_takes_the_report_event_of_a_streamed_run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    script = (
        'print(\'{"event": "start", "pair": 0, "game": 0}\')\n'
        'print(\'{"event": "move", "pair": 0, "game": 0}\')\n'
        'print(\'{"event": "end", "pair": 0, "game": 0}\')\n'
        'print(\'{"event": "report", "pairs": 1, "wins": 1, "sequential": {"stop_reason": "running"}}\')\n'
    )
    with (tmp_path / "err").open("w") as err:
        report, events, code = experiment.launch([sys.executable, "-c", script], err)
    assert code == 0 and report == {"pairs": 1, "wins": 1, "sequential": {"stop_reason": "running"}}
    assert [e["event"] for _, e in events] == ["start", "end"]


def test_launch_takes_the_pretty_printed_report_of_a_fixed_count_run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    script = (
        "import json; "
        "print(json.dumps({'event': 'start', 'pair': 0, 'game': 0})); "
        "print(json.dumps({'event': 'end', 'pair': 0, 'game': 0})); "
        "print(json.dumps({'pairs': 1, 'wins': 1, 'sequential': None}, indent=2))"
    )
    with (tmp_path / "err").open("w") as err:
        report, events, code = experiment.launch([sys.executable, "-c", script], err)
    assert code == 0 and report == {"pairs": 1, "wins": 1, "sequential": None}
    assert [e["event"] for _, e in events] == ["start", "end"]
    torn = "print('{'); print('  \"pairs\": 1,')"
    with (tmp_path / "err").open("w") as err, pytest.raises(ValueError, match="ended inside"):
        experiment.launch([sys.executable, "-c", torn], err)


def test_launch_kills_the_process_tree_on_an_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    pid_file = tmp_path / "grandchild"
    script = (
        "import json, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "print(json.dumps({'event': 'start', 'pair': 0, 'game': 0}), flush=True)\n"
        "print('not json', flush=True)\n"
        "time.sleep(60)\n"
    )
    with (tmp_path / "err").open("w") as err, pytest.raises(ValueError, match="bot eval printed"):
        experiment.launch([sys.executable, "-c", script], err)
    grandchild = int(pid_file.read_text())
    assert not psutil.pid_exists(grandchild) or psutil.Process(grandchild).status() == psutil.STATUS_DEAD
    assert not any(p.pid == grandchild for p in psutil.Process().children(recursive=True))


def test_runner_keeps_the_lock_while_a_child_survives(workspace, monkeypatch):
    real_launch = experiment.launch
    spec = manifest(workspace)
    spec["arms"], spec["rules"], spec["matches"] = [], [], [spec["matches"][2]]  # the sequential test only
    spec["identities"] = experiment.identities(spec)
    directory = workspace / "runs" / "nnue_gauntlets" / "tiny"
    directory.mkdir(parents=True)
    file = directory / "experiment.json"
    file.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(experiment, "nice", lambda mask: None)
    orphans = []

    def dying_launch(command, err, events=None):
        orphans.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]))
        raise RuntimeError("the coordinator died")

    monkeypatch.setattr(experiment, "launch", dying_launch)
    with pytest.raises(RuntimeError, match="coordinator died"):
        experiment.main(["run", str(file), "--only", "match"])
    assert (directory / "q.timing.json").is_file()  # the failed invocation's timing is kept
    held = experiment.lock_read()
    assert held and held["owner"] == "experiment" and held["pid"] == os.getpid()
    with pytest.raises(RuntimeError, match="pid 1"):
        experiment.main(["lock", "release", "experiment", "1"])
    orphans[0].kill()
    orphans[0].wait()
    assert experiment.main(["lock", "release", "experiment", str(os.getpid())]) == 0
    assert experiment.lock_read() is None

    def clean_launch(command, err, events=None):
        raise RuntimeError("the coordinator died cleanly")

    monkeypatch.setattr(experiment, "launch", clean_launch)
    with pytest.raises(RuntimeError, match="died cleanly"):
        experiment.main(["run", str(file), "--only", "match"])
    assert experiment.lock_read() is None

    # A real coordinator that spawns a seat and exits abruptly: the seat dies with the job, the lock is released.
    pid_file = workspace / "orphan"
    script = (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "sys.exit(3)\n"
    )
    monkeypatch.setattr(experiment, "launch", real_launch)
    monkeypatch.setattr(experiment, "command_of", lambda e, d, m: [sys.executable, "-c", script])
    with pytest.raises(RuntimeError, match="exit code 3"):
        experiment.main(["run", str(file), "--only", "match"])
    orphan = int(pid_file.read_text())
    assert not psutil.pid_exists(orphan) or psutil.Process(orphan).status() == psutil.STATUS_DEAD
    assert experiment.lock_read() is None


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


def test_the_real_c1_fixture_parses_and_reconciles(tmp_path):
    """The journal and report that match::eval::eval wrote with scripted players (Astra, b5b6a47): the
    header, the pair records, the flattened sequential state and the audit's reconciliation."""
    fixtures = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    directory = tmp_path / "c1"
    directory.mkdir()
    shutil.copy(fixtures / "c1_journal.jsonl", directory / "c1.jsonl")
    report = json.loads((fixtures / "c1_report.json").read_text(encoding="utf-8"))
    header, pairs = experiment.journal_pairs(directory / "c1.jsonl")
    assert len(pairs) == 128 and header["protocol_digest"] == report["sequential"]["protocol_digest"]
    assert header["protocol"]["test"]["batch"] == 16 and header["protocol"]["settings"]["workers"] == 1
    match = {
        "tag": "c1",
        "candidate": "candidate.exe",
        "reference": "reference.exe",
        "move_ms": 50,
        "sprt": True,
        "seed": 2026091225,
        "opening_plies": 2,
        "concurrent": 1,
        "player_threads": 1,
    }
    assert experiment.audit(match, report, directory) == {
        "games": 256,
        "consistent": True,
        "problems": [],
        "ends": {"CaptureClock": 243, "Goal": 13},
        "first_mover_score": pytest.approx(0.474609375),
        "distinct_openings": 125,
        "mean_plies": pytest.approx(195.9765625),
    }
    assert experiment.finished(match, report) and experiment.pair_points(pairs[0]) == 2
    summary = experiment.summarise(match, report, directory)  # both seats carry the engine's id name
    assert summary["valid"] and summary["stop_reason"] == "accept" and summary["counts"] == [0, 0, 115, 13, 0]
    assert summary["games"] == 256 and summary["error"] is None and summary["protocol"]["schema"] == 1
    assert experiment.audit(match | {"seed": 1}, report, directory)["problems"] == [
        "protocol seed is 2026091225, the match says 1"
    ]
