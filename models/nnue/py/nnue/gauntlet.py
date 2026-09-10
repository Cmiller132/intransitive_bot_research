"""The gauntlet (DESIGN item 16): a candidate network against the incumbent
through `bot eval --move-ms`, in predeclared stages with predeclared seeds.

    python -m nnue.gauntlet --name <name> --candidate <a.nnue> --incumbent <b.nnue> [--stage screen|accept|confirm|all]
        [--candidate-bin <bot.exe>] [--incumbent-bin <bot.exe>]

A search change is a different binary: `--candidate-bin` (or `--incumbent-bin`)
seats that side through an external `bot rpsi` process of the given binary
(`rpsi:<bin> rpsi --player nnue:<file> --threads <T>`) instead of the host's
own player, so two builds can meet under the same clock.

Stages: screen (64 games at 50 ms, one thread each), accept (160 new games at
100 ms, one thread each), confirm (128 new games at 250 ms, four threads
each, one pair at a time). A stage passes at a score of at least 55 % with
the report's opening-pair bootstrap lower bound above 50 % (the screen only
needs more than 50 %), no forfeits and every pair complete. Results go to
`runs/nnue_gauntlets/<name>/<stage>.json` with the game records beside them.

`--stage all` runs screen and accept; confirm is asked for by name, for the
candidate that is about to be deployed or for an accept below 65 %, where
the deployment time control could still reverse a narrow verdict.

`--sims N` replaces every stage's move clock with a node budget
(`Clock::Sims(N)`, N x 2,500 nodes per move on one thread): for a change to
the evaluator alone, games under a fixed node budget are independent of the
machine's load and reproducible, and a stage takes a minute; a wall clock is
kept for search and speed changes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .paths import run_dir, workspace_root


@dataclass(frozen=True)
class Stage:
    name: str
    pairs: int
    move_ms: int
    player_threads: int
    concurrent: int
    seed: int
    min_score: float
    need_lower_bound: bool


STAGES = {
    "screen": Stage("screen", 32, 50, 1, 4, 20260909101, 0.5, False),
    "accept": Stage("accept", 80, 100, 1, 4, 20260909102, 0.55, True),
    "confirm": Stage("confirm", 64, 250, 4, 1, 20260909103, 0.55, True),
}


def bot_binary() -> Path:
    """The release `bot`, or the build named by NNUE_BOT (a frozen copy, so a
    running gauntlet never holds the file the next `cargo build` replaces)."""
    if os.environ.get("NNUE_BOT"):
        path = Path(os.environ["NNUE_BOT"])
        if not path.is_file():
            raise FileNotFoundError(f"NNUE_BOT={path} is not a file")
        return path.resolve()
    root = workspace_root()
    for name in ("bot.exe", "bot"):
        path = root / "target" / "release" / name
        if path.is_file():
            return path
    raise FileNotFoundError("build the release `bot` first (cargo build --release -p cli)")


def seat(network: Path, binary: Path | None, threads: int) -> str:
    """A player spec: the host's own player, or an external `bot rpsi` of `binary`."""
    if binary is None:
        return f"nnue:{network}"
    for path in (network, binary):
        if " " in str(path):
            raise ValueError(f"an external seat cannot carry a path with spaces: {path}")
    return f"rpsi:{binary} rpsi --player nnue:{network} --threads {threads}"


def play(
    stage: Stage,
    candidate: Path,
    incumbent: Path,
    out: Path,
    seed: int,
    candidate_bin: Path | None = None,
    incumbent_bin: Path | None = None,
    sims: int = 0,
) -> dict:
    """One stage's paired games; returns the eval report with the verdict added."""
    out.mkdir(parents=True, exist_ok=True)
    records = out / f"{stage.name}.games.jsonl"
    command = [
        str(bot_binary()),
        "eval",
        "--candidate",
        seat(candidate, candidate_bin, stage.player_threads),
        "--reference",
        seat(incumbent, incumbent_bin, stage.player_threads),
        *(["--sims", str(sims)] if sims else ["--move-ms", str(stage.move_ms)]),
        "--player-threads",
        str(1 if sims else stage.player_threads),
        "--threads",
        str(stage.concurrent),
        "--pairs",
        str(stage.pairs),
        "--seed",
        str(seed),
        "--records",
        str(records),
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=workspace_root(), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"bot eval failed: {result.stderr.strip()}")
    report = json.loads(result.stdout)
    games = report["wins"] + report["draws"] + report["losses"]
    score = (report["wins"] + 0.5 * report["draws"]) / max(1, games)
    lower = report.get("bootstrap_interval", report["interval"])[0]
    complete = report.get("complete_pairs", report["pairs"]) == stage.pairs
    forfeits = report.get("forfeits", 0)
    passed = (
        complete
        and forfeits == 0
        and score >= stage.min_score
        and (not stage.need_lower_bound or (lower + 1) / 2 > 0.5)
    )
    verdict = {
        "stage": stage.name,
        "candidate": str(candidate),
        "incumbent": str(incumbent),
        "candidate_bin": None if candidate_bin is None else str(candidate_bin),
        "incumbent_bin": None if incumbent_bin is None else str(incumbent_bin),
        "seed": seed,
        "sims": sims,
        "score": score,
        "games": games,
        "lower_bound_score": (lower + 1) / 2,
        "complete": complete,
        "forfeits": forfeits,
        "passed": passed,
        "bot": str(bot_binary()),
        "command": command,
        "report": report,
    }
    (out / f"{stage.name}.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="results under runs/nnue_gauntlets/<name>")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate-bin", type=Path, default=None, help="bot binary seating the candidate")
    parser.add_argument("--incumbent-bin", type=Path, default=None, help="bot binary seating the incumbent")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--sims", type=int, default=0, help="node budget per move (x 2,500) instead of the clock")
    parser.add_argument("--seed-offset", type=int, default=0, help="added to every stage's seed for a repeat")
    args = parser.parse_args(argv)
    out = run_dir("nnue_gauntlets") / args.name
    stages = [STAGES["screen"], STAGES["accept"]] if args.stage == "all" else [STAGES[args.stage]]
    for stage in stages:
        verdict = play(
            stage,
            args.candidate.resolve(),
            args.incumbent.resolve(),
            out,
            stage.seed + args.seed_offset,
            None if args.candidate_bin is None else args.candidate_bin.resolve(),
            None if args.incumbent_bin is None else args.incumbent_bin.resolve(),
            args.sims,
        )
        print(json.dumps({k: v for k, v in verdict.items() if k not in ("report", "command")}))
        if not verdict["passed"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
