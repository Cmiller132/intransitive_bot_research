"""Reads a quickstart `runs/` tree and turns it into decisions.

The files are the ones models/nnue/quickstart.sh writes today:

* runs/nnue_selfplay/<n>/manifest.json   the batch (settings, shards with counts, `complete`), pending/ and active/ while
                                          it plays;
* runs/nnue_data/selfplay_<n>[_quiet]/provenance.json   the imported rows;
* runs/<n>/config.json, log.csv, latest.pt, best.nnue(.json)   training;
* runs/<n>/eval*.json (the quickstart's eval.json, a sequential re-test's eval_sprt*.json) and sprt_*.jsonl journals;
* runs/nnue_quickstart/<n>/*.log and evaluate.jsonl   the script's own logs, read only for live progress.

The rules follow the research loop (models/nnue/DESIGN.md section 7) scaled to one machine:

* a network becomes the next start when the sequential test at 40k nodes (.50 against .52) accepts it, or when it stops
  inconclusive at its cap with the interval's lower bound above .50 (the loop's cap_lower rule); a fixed-count reading
  decides only at the decision budget (40k nodes, 400 pairs) with its lower bound above zero;
* anything else from the same start is a null: another batch from the same start, every earlier batch kept, the batch
  doubled after each null up to four times the plan (the loop's increment rule);
* batches grow with the lineage: a first reading of 3,200 games, then 6,400, then the loop's 12,800-game round
  (about 2.7 M rows), 25,600 (its 6 M-row rounds) and 51,200 (its 12 M rows after null rounds);
* labels at 50k nodes (the loop's budget since 2026-09-16: 25k-labelled gains faded at deeper search).

Pure standard library so it runs with the same python3 the quickstart uses.
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
            "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth"]

PLAN = {
    "nodes": 50_000,             # label budget of new batches (the loop since round 8)
    "nodes_floor": 25_000,       # below this the loop has no evidence at all
    "decision_sims": 16,         # 40k nodes: the loop's gain test
    "reading_sims": 8,           # 20k nodes: the quickstart's fast reading
    "decision_pairs": 400,       # a fixed-count decision
    "reading_pairs": 100,
    "sprt_target": 0.52,         # C1's upper hypothesis, about +14 Elo: the base the schedule ends at
    # the target by generation, the last repeating: early generations from a weak start gain +30 Elo and more, so a
    # higher target settles them in about half the pairs; later gains are smaller and need the base target to show
    "sprt_targets": [0.55, 0.54, 0.53, 0.53, 0.52],
    "sprt_cap": 3008,            # the most pairs any test plays
    "eval_share": 1 / 3,         # a test may spend at most this share of the search its batch cost
    "eval_threads_max": 16,      # the sequential coordinator's limit
    "ladder": [3_200, 6_400, 12_800, 25_600, 51_200],
    "retry_growth_max": 4,       # after nulls the batch doubles, up to 4x the plan for that generation
    "rows_per_game": 220,        # QUICKSTART.md; replaced by the tree's own ratio once batches exist
    "passes": [3.0, 5.0],        # QUICKSTART.md: the range that does not memorise a batch
    "arena_bar_elo": 30,         # the arena's confirmation bar in the loop (context only)
}
LADDER_WHY = [
    "a first reading from a new start",
    "twice the first reading",
    "the research loop's first round: 12,800 games, about 2.7 M rows",
    "the loop's 6 M-row rounds",
    "the loop's 12 M rows after null rounds",
]

SWEET_SPOTS = {
    "late_best_fraction": 0.5,      # best epoch in the second half of the run
    "early_best_epochs": 2,         # bottoming after one or two epochs is the memorising sign
    "objective_rise_warn": 0.005,   # final objective above the minimum, relative
    "objective_rise_bad": 0.02,
    "drift_warn": 0.01,             # an older set's validation mse rising from epoch 0, relative
    "drift_bad": 0.03,
    "bias_warn": 0.005,
    "bias_bad": 0.01,
    "stale_minutes": 30,            # no file written for this long: the step stopped
}

# What the loop's gain tests read later on the arena at 200k nodes (DESIGN.md section 7, 2026-09-15/16).
DEPTH_NOTE = ("Measured against the start at the evaluation budget only. In the research loop, gains of +19 to +39 Elo "
              "at 40k nodes read between -3 and +23 on the arena at 200k: deeper search compresses them, often to about half.")

METRIC_HELP = {
    "objective": ("Validation error that picks best.nnue: the share-weighted mean of each set's selection mse.",
                  "Lower. It should keep falling or flatten late; a minimum in epoch 0-2 followed by a rise means memorising."),
    "loss": ("Training loss (value loss + symmetry_weight x consistency), averaged over the epoch's steps.",
             "Falls steadily. Only meaningful next to objective: loss falling while objective rises is overfitting."),
    "value": ("The value part of the training loss (tanh mse plus the small logit term).", "Falls with loss."),
    "consistency": ("Squared disagreement between two random board symmetries of the same position.",
                    "Falls towards zero; the network should score a position the same from every symmetry."),
    "lr": ("Learning rate at the end of the epoch: linear warmup, cosine down to 15 % of the peak.",
           "The recipe's shape; a flat or odd curve means the schedule was overridden."),
    "seconds": ("Wall time of the epoch, including validation.", "Stable; spikes are contention on the machine."),
    "best": ("Lowest objective so far.", "A staircase that keeps stepping down."),
    "mse": ("Validation mean squared error against the search labels, all six symmetries.",
            "Lower. Compare epochs within one run; the level depends on the data."),
    "mae": ("Validation mean absolute error.", "Lower; less swayed by a few hard positions than mse."),
    "bias": ("Mean signed error: positive means the network is more optimistic than the search.",
             "Close to zero, within about +-0.005."),
    "symmetry_range_mean": ("Average spread of the six symmetric evaluations of one position.",
                            "Lower and falling: symmetric positions should evaluate alike."),
    "symmetry_range_p95": ("95th percentile of that spread: the worst symmetric disagreements.", "Lower and falling."),
    "rows": ("Validation rows in this slice.", "Small slices (a few hundred rows) are noisy; read their curves loosely."),
}
STRATA_HELP = {
    "selection": "rows that pick best.nnue (outcome-labelled rows excluded)",
    "sparse": "4 pieces or fewer",
    "middle": "5 to 12 pieces",
    "full": "more than 12 pieces",
    "long_clock": "long since the last capture",
}


# ---------------------------------------------------------------------------------------------------------------
# reading


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def parse_cell(value: str):
    if value in ("True", "False"):
        return value == "True"
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() and "." not in value and "e" not in value.lower() else number


def read_log(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [{k: parse_cell(v) for k, v in row.items() if k} for row in reader]
            columns = [c for c in (reader.fieldnames or []) if c]
    except (OSError, csv.Error):
        return None
    # a log written before the quickstart moved old runs aside can hold two runs; keep the last epoch 0 onwards
    starts = [i for i, r in enumerate(rows) if r.get("epoch") == 0]
    restarted = len(starts) > 1
    if restarted:
        rows = rows[starts[-1]:]
    return {"columns": columns, "rows": rows, "restarted": restarted, "mtime": path.stat().st_mtime}


def count_lines(path: Path, needle: bytes | None = None) -> int:
    try:
        with path.open("rb") as f:
            data = f.read()
    except OSError:
        return 0
    return data.count(b"\n") if needle is None else sum(1 for line in data.split(b"\n") if needle in line)


# ---------------------------------------------------------------------------------------------------------------
# strength


def score_to_elo(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def strength(raw: dict) -> dict:
    """Score, Elo and their uncertainty from one report, over opening pairs (the sampling unit).

    The pentanomial counts of a sequential report give the pair scores exactly; a fixed-count report gives the pair
    margins' 95 % normal interval, whose half-width is 1.96 standard errors. Elo is the logistic transform of the
    score and its interval of the score's interval. Normalised Elo divides the gain by the spread of the results, so
    it does not move with the draw rate (fishtest's nElo); LOS is the probability that the score is above .50."""
    counts = ((raw.get("sequential") or {}).get("counts")) or None
    n_pairs = raw.get("complete_pairs") or raw.get("pairs") or 0
    if counts and sum(counts) > 1:
        n = sum(counts)
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        mean = sum(c * v for c, v in zip(counts, values)) / n
        sd_pair = math.sqrt(sum(c * (v - mean) ** 2 for c, v in zip(counts, values)) / (n - 1))
        source = "pentanomial counts"
    else:
        n = n_pairs
        margin = raw.get("margin", 0.0)
        lo, hi = raw.get("interval") or (margin, margin)
        mean = 0.5 + margin / 2
        sd_pair = ((hi - lo) / (2 * 1.96)) / 2 * math.sqrt(n) if n else 0.0
        source = "pair margins"
    se = sd_pair / math.sqrt(n) if n and sd_pair > 0 else None
    gain = mean - 0.5
    lo_s, hi_s = (mean - 1.96 * se, mean + 1.96 * se) if se else (mean, mean)
    out = {
        "pairs": n,
        "score": mean,
        "score_interval": [lo_s, hi_s],
        "se": se,
        "sd_pair": sd_pair,
        "source": source,
        "elo": score_to_elo(mean),
        "elo_interval": [score_to_elo(lo_s), score_to_elo(hi_s)],
        "elo_se": (score_to_elo(hi_s) - score_to_elo(lo_s)) / (2 * 1.96) if se else None,
        "nelo": gain / (sd_pair * math.sqrt(2)) * 800 / math.log(10) if sd_pair > 0 else None,
        "los": normal_cdf(gain / se) if se else (1.0 if gain > 0 else 0.0 if gain < 0 else 0.5),
        # the smallest gain this sample would show with its lower bound above .50
        "resolution_elo": score_to_elo(0.5 + 1.96 * se) if se else None,
        # pairs for the observed gain to clear .50 if it were the true one
        "pairs_needed": math.ceil(n * (1.96 * se / gain) ** 2) if se and gain > 0 else None,
    }
    return out


def eval_summary(path: Path) -> dict | None:
    raw = read_json(path)
    if not raw or "wins" not in raw:
        return None
    games = raw["wins"] + raw["draws"] + raw["losses"]
    seq = raw.get("sequential")
    st = strength(raw)
    test = ((seq or {}).get("protocol") or {}).get("test") or {}
    sims = raw.get("sims") or 0
    pairs = raw.get("complete_pairs") or raw.get("pairs") or 0
    lo_s, hi_s = st["score_interval"]
    if seq:
        stop = seq.get("stop_reason")
        target = test.get("s1", PLAN["sprt_target"])
        cap = test.get("cap", PLAN["sprt_cap"])
        if stop == "accept":
            verdict = "better"
        elif stop in ("inconclusive", "stopped") and lo_s > 0.5:
            verdict = "better_small"
        elif stop == "reject" and target > PLAN["sprt_target"] + 1e-9 and lo_s > 0.5:
            verdict = "retest"  # not the raised gain, but better than its start: worth the base target's test
        elif stop in ("reject", "inconclusive", "stopped"):
            verdict = "worse" if hi_s < 0.5 else "null"
        else:
            verdict = "running" if stop == "running" else "invalid"
        grade = "decision" if stop in ("accept", "reject", "inconclusive", "stopped") else "partial"
    else:
        stop, target, cap = None, None, None
        grade = "decision" if sims >= PLAN["decision_sims"] and pairs >= PLAN["decision_pairs"] else "reading"
        verdict = "better" if lo_s > 0.5 else "worse" if hi_s < 0.5 else "null"
    boot = raw.get("bootstrap_interval")
    return {
        **raw,
        **st,
        "file": path.name,
        "mtime": path.stat().st_mtime,
        "games": games,
        "draw_rate": raw["draws"] / games if games else 0.0,
        "kind": "sequential" if seq else "fixed",
        "grade": grade,
        "verdict": verdict,
        "stop": stop,
        "llr": (seq or {}).get("llr"),
        "llr_bound": math.log(19),
        "sprt_target": target,
        "sprt_cap": cap,
        "descriptive_only": bool(seq),
        "bootstrap_score_interval": [0.5 + boot[0] / 2, 0.5 + boot[1] / 2] if boot else None,
        "reference_path": _player_path(raw.get("reference", "")),
        "candidate_path": _player_path(raw.get("candidate", "")),
    }


def _player_path(player: str) -> str:
    return player.split(":", 1)[-1].split("?", 1)[0] if player else ""


# ---------------------------------------------------------------------------------------------------------------
# batches and datasets


def generation_summary(runs_dir: Path, name: str) -> dict | None:
    folder = runs_dir / "nnue_selfplay" / name
    manifest = read_json(folder / "manifest.json")
    if not manifest:
        return None
    totals: dict[str, int] = {}
    for shard in manifest.get("shards", []):
        for k, v in (shard.get("counts") or {}).items():
            totals[k] = totals.get(k, 0) + int(v)
    settings = manifest.get("settings") or {}
    games = totals.get("games", 0)
    threads = settings.get("threads") or 1
    pending = len(list((folder / "pending").glob("*.json"))) if (folder / "pending").is_dir() else 0
    active = list((folder / "active").glob("*.jsonl")) if (folder / "active").is_dir() else []
    touched = max([p.stat().st_mtime for p in [folder / "manifest.json", *active] if p.exists()], default=0)
    search_hours = totals.get("search_ns", 0) / 3.6e12
    wall_rate = games / (search_hours / threads) if search_hours else None  # games per hour with every thread busy
    return {
        "complete": bool(manifest.get("complete")),
        "games": games,
        "games_done": games + pending,
        "in_flight": len(active),
        "games_target": settings.get("games"),
        "nodes": settings.get("nodes"),
        "threads": settings.get("threads"),
        "seed": settings.get("seed"),
        "multipv": settings.get("multipv", 1),
        "pv_labels": settings.get("pv_labels", 0),
        "player": settings.get("player"),
        "init": _player_path(settings.get("player", "")),
        "model_sha256": manifest.get("model_sha256"),
        "shards": len(manifest.get("shards", [])),
        "roots": totals.get("roots", 0),
        "eligible_roots": totals.get("eligible_roots", 0),
        "censored_share": totals.get("censored", 0) / games if games else 0.0,
        "roots_per_game": totals.get("roots", 0) / games if games else 0.0,
        "alternative_share": totals.get("alternative_roots", 0) / totals["roots"] if totals.get("roots") else 0.0,
        "games_per_hour": wall_rate,
        "games_per_core_hour": games / search_hours if search_hours else None,
        "touched": touched,
    }


def dataset_summary(runs_dir: Path, name: str) -> dict | None:
    folder = runs_dir / "nnue_data" / name
    prov = read_json(folder / "provenance.json")
    if not prov:
        return None
    counts = prov.get("counts") or {}
    line = max(0, int(counts.get("line_rows", 0)) - int(counts.get("line_duplicates", 0)))
    quiet = prov.get("quiet")
    return {
        "rows": prov.get("rows"),
        "root_rows": (prov.get("rows") or 0) - line,
        "line_rows": line,
        "games": counts.get("games"),
        "ends": prov.get("ends") or {},
        "min_ply": prov.get("min_ply"),
        "encoded": (folder / "ids8.json").is_file(),
        "quiet": {"kept": quiet.get("roots_kept"), "skipped": quiet.get("skipped") or {}} if quiet else None,
        "mtime": (folder / "provenance.json").stat().st_mtime,
    }


# ---------------------------------------------------------------------------------------------------------------
# one run


@dataclass
class Check:
    group: str
    label: str
    value: str
    target: str
    status: str  # ok | warn | bad | info
    note: str

    def as_dict(self) -> dict:
        return self.__dict__


def opponent_label(path: str) -> str:
    """runs/first/best.nnue -> first; runs/first.replaced_20260917_103210/best.nnue -> first, trained before 17 Sep 10:32."""
    m = re.search(r"runs/([^/]+)/best\.nnue$", path.replace("\\", "/"))
    if not m:
        return Path(path).name or path
    name = m.group(1)
    moved = re.fullmatch(r"(.+)\.replaced_(\d{8})_(\d{6})", name)
    if moved:
        try:
            when = time.strftime("%d %b %H:%M", time.strptime(moved.group(2) + moved.group(3), "%Y%m%d%H%M%S"))
        except ValueError:
            when = moved.group(2)
        return f"{moved.group(1)}, earlier training ({when})"
    return name


def earlier_trainings(runs_dir: Path, name: str, anchors: list[dict]) -> list[dict]:
    """The trainings of this run that a retraining moved aside (runs/<n>.replaced_<time>), newest first: their settings,
    best objective, own test against their start, and any direct match the current network played against them."""
    out = []
    for folder in sorted(Path(runs_dir).glob(f"{glob_escape(name)}.replaced_*"), reverse=True):
        if not re.fullmatch(rf"{re.escape(name)}\.replaced_\d{{8}}_\d{{6}}", folder.name):
            continue
        saved = read_json(folder / "config.json")
        if not saved:
            continue
        cfg = saved.get("config") or {}
        meta = read_json(folder / "best.nnue.json") or {}
        tests = [e for e in (eval_summary(p) for p in folder.glob("eval*.json")) if e and e["reference_path"] == cfg.get("init")]
        rank = {"decision": 2, "partial": 1, "reading": 0}
        test = max(tests, key=lambda e: (rank[e["grade"]], e["mtime"])) if tests else None
        path = f"runs/{folder.name}/best.nnue"
        direct = next((a for a in anchors if a["reference_path"] == path), None)
        out.append({
            "dir": folder.name, "label": opponent_label(path), "has_net": (folder / "best.nnue").is_file(),
            "batch": cfg.get("batch"), "epochs": cfg.get("epochs"), "steps_per_epoch": cfg.get("steps_per_epoch"),
            "lr": cfg.get("lr"), "ema": cfg.get("ema"), "objective": meta.get("objective"), "best_epoch": meta.get("epoch"),
            "test": {k: test[k] for k in ("score", "elo", "elo_interval", "verdict", "pairs", "file")} if test else None,
            "direct": {k: direct[k] for k in ("score", "score_interval", "elo", "elo_interval", "los", "pairs", "file")} if direct else None,
        })
    return out


def glob_escape(text: str) -> str:
    return re.sub(r"([*?\[])", r"[\1]", text)


def run_names(runs_dir: Path) -> list[str]:
    """Training runs (runs/<n>/) and batches that have not reached training yet (runs/nnue_selfplay/<n>/). Names
    starting with nnue_ are the research loop's directories (the quickstart refuses them) and <n>.replaced_<time>
    are runs the quickstart moved aside when their settings changed."""
    names = set()
    if runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and not d.name.startswith(("nnue_", ".")) and ".replaced_" not in d.name:
                names.add(d.name)
    selfplay = runs_dir / "nnue_selfplay"
    if selfplay.is_dir():
        names.update(d.name for d in selfplay.iterdir() if d.is_dir() and (d / "manifest.json").is_file())
    return sorted(names)


def load_run(runs_dir: Path, name: str) -> dict | None:
    directory = runs_dir / name
    config_file = read_json(directory / "config.json")
    meta = read_json(directory / "best.nnue.json")
    log = read_log(directory / "log.csv")
    found = [e for e in (eval_summary(p) for p in sorted(directory.glob("eval*.json"))) if e]
    start = ((config_file or meta or {}).get("config") or {}).get("init") or ""
    # an evaluation against another network than the start measures distance to that network; it never decides
    anchors = [e for e in found if e["reference_path"] and start and e["reference_path"] != start]
    evals = [e for e in found if e not in anchors]
    for e in found:
        e["against_start"] = e not in anchors
        e["opponent"] = "its start" if e["against_start"] else opponent_label(e["reference_path"])
    best_mtime = (directory / "best.nnue").stat().st_mtime if (directory / "best.nnue").is_file() else None
    for e in found:  # an evaluation from before the network last changed describes an older network
        e["stale"] = bool(best_mtime and e["mtime"] < best_mtime - 1)
    generation = generation_summary(runs_dir, name)
    if not (config_file or meta or log or evals or generation):
        return None
    source = config_file or meta or {}
    config = source.get("config") or {}
    mixture = [{"set": s, "share": float(w)} for s, w in source.get("data", [])]
    own = next((m["set"] for m in mixture if re.fullmatch(rf"selfplay_{re.escape(name)}(_quiet.*)?", m["set"])),
               f"selfplay_{name}")
    # the decision reads the strongest evidence: a finished sequential test, then a fixed count at the decision
    # budget, then a reading; the newest file within a grade
    rank = {"decision": 2, "partial": 1, "reading": 0}
    current = [e for e in evals if not e["stale"]]
    primary = max(current, key=lambda e: (rank[e["grade"]], e["mtime"])) if current else None
    files = [p for p in directory.iterdir() if p.is_file()] if directory.is_dir() else []
    init = config.get("init") or (generation or {}).get("init", "")
    return {
        "name": name,
        "path": str(directory),
        "config": config,
        "mixture": mixture,
        "own_set": own,
        "dataset_hashes": source.get("datasets") or {},
        "init": init,
        "init_sha256": source.get("init_sha256") or (generation or {}).get("model_sha256"),
        "best": {k: meta.get(k) for k in ("epoch", "objective", "sha256", "bytes", "hidden", "version", "features",
                                          "format")} if meta else None,
        "log": log,
        "evals": sorted(evals, key=lambda e: -e["mtime"]),
        "anchors": sorted(anchors, key=lambda e: -e["mtime"]),
        "all_evals": sorted(found, key=lambda e: -e["mtime"]),
        "earlier": earlier_trainings(runs_dir, name, sorted(anchors, key=lambda e: -e["mtime"])),
        "eval": primary,
        "generation": generation,
        "has_net": (directory / "best.nnue").is_file(),
        "has_checkpoint": (directory / "latest.pt").is_file(),
        "journals": sorted(p.name for p in directory.glob("sprt_*.jsonl")),
        "started": min((p.stat().st_mtime for p in (directory / "config.json", directory / "log.csv") if p.is_file()),
                       default=(generation or {}).get("touched", 0)),
        "mtime": max([p.stat().st_mtime for p in files] + [(generation or {}).get("touched", 0)], default=0),
    }


def training_stats(run: dict) -> dict:
    log = run["log"]
    if not log or not log["rows"]:
        return {}
    rows = log["rows"]
    valid = [(i, r.get("objective")) for i, r in enumerate(rows) if isinstance(r.get("objective"), (int, float))]
    if not valid:
        return {}
    best_i, best_v = min(valid, key=lambda t: t[1])
    first_v, last_v = valid[0][1], valid[-1][1]
    epochs_planned = run["config"].get("epochs") or len(rows)
    sets = [m["set"] for m in run["mixture"]] or sorted({c.split(".")[0] for c in log["columns"] if "." in c})

    def series(col):
        return [r.get(col) for r in rows]

    drift = {}
    for s in sets:
        vals = [v for v in series(f"{s}.mse") if isinstance(v, (int, float))]
        at_best = series(f"{s}.mse")[best_i]
        if len(vals) >= 2 and isinstance(at_best, (int, float)):
            drift[s] = {"first": vals[0], "at_best": at_best, "last": vals[-1], "change_to_best": (at_best - vals[0]) / vals[0]}
    tail = rows[best_i:]
    loss_tail = [r.get("loss") for r in tail if isinstance(r.get("loss"), (int, float))]
    diverging = (best_i < len(rows) - 2 and len(loss_tail) >= 2 and loss_tail[-1] < loss_tail[0]
                 and (last_v - best_v) / best_v > SWEET_SPOTS["objective_rise_warn"])
    q = max(2, len(valid) // 4)
    recent = [v for _, v in valid[-q:]]
    slope = (recent[-1] - recent[0]) / (len(recent) - 1) / recent[0] if len(recent) > 1 else 0.0
    seconds = [r.get("seconds") for r in rows if isinstance(r.get("seconds"), (int, float))]
    return {
        "epochs_done": len(rows),
        "epochs_planned": epochs_planned,
        "finished": len(rows) >= epochs_planned,
        "best_epoch": rows[best_i].get("epoch", best_i),
        "best_objective": best_v,
        "first_objective": first_v,
        "last_objective": last_v,
        "improvement": (first_v - best_v) / first_v if first_v else 0,
        "rise_after_best": (last_v - best_v) / best_v if best_v else 0,
        "recent_slope": slope,
        "diverging": diverging,
        "drift": drift,
        "bias_at_best": {s: rows[best_i].get(f"{s}.bias") for s in sets},
        "total_steps": rows[-1].get("step"),
        "seconds": sum(seconds),
        "seconds_per_epoch": sum(seconds) / len(seconds) if seconds else None,
        "restarted": log["restarted"],
    }


def run_status(run: dict, runs_dir: Path, now: float) -> dict:
    """A guess at what a run started outside the dashboard is doing, from when its files last changed (a run the
    dashboard started is described by its runner instead, which knows)."""
    stale = SWEET_SPOTS["stale_minutes"] * 60
    gen, tr = run["generation"], run["training"]
    logs = runs_dir / "nnue_quickstart" / run["name"]
    if gen and not gen["complete"]:
        target = gen["games_target"] or 0
        done = gen["games_done"]
        eta = (target - done) / gen["games_per_hour"] * 3600 if gen["games_per_hour"] and target else None
        live = now - gen["touched"] < stale
        return {"state": "generating" if live else "stopped", "step": "generate",
                "fraction": done / target if target else None, "eta_seconds": eta if live else None,
                "detail": f"{done:,} of {target:,} games" + (f", {gen['in_flight']} in flight" if live and gen["in_flight"] else "")}
    if not run["config"]:
        own = run["datasets"].get(run["own_set"]) if run.get("datasets") else None
        touched = max((p.stat().st_mtime for p in logs.glob("*.log")), default=gen["touched"] if gen else 0)
        live = now - touched < stale
        return {"state": "preparing" if live else "stopped", "step": "import",
                "fraction": None, "eta_seconds": None,
                "detail": "importing and encoding the batch" if live and not own else "the batch is ready for training"}
    if tr and not tr["finished"] and not run["eval"]:
        live = now - run["log"]["mtime"] < max(stale, 3 * (tr["seconds_per_epoch"] or 0))
        left = tr["epochs_planned"] - tr["epochs_done"]
        return {"state": "training" if live else "stopped", "step": "train",
                "fraction": tr["epochs_done"] / tr["epochs_planned"],
                "eta_seconds": left * tr["seconds_per_epoch"] if live and tr["seconds_per_epoch"] else None,
                "detail": f"epoch {tr['epochs_done']} of {tr['epochs_planned']}"}
    if not tr and run["config"] and not run["eval"]:
        return {"state": "training", "step": "train", "fraction": 0.0, "eta_seconds": None, "detail": "loading the data"}
    stream = logs / "evaluate.jsonl"
    if run["has_net"] and (not run["eval"] or (stream.is_file() and stream.stat().st_mtime > run["eval"]["mtime"])):
        if stream.is_file() and now - stream.stat().st_mtime < stale:
            games = count_lines(stream, b'"end"')
            return {"state": "evaluating", "step": "evaluate", "fraction": None, "eta_seconds": None,
                    "detail": f"{games:,} games ({games // 2:,} pairs) played"}
    if run["eval"] and run["eval"]["verdict"] == "running":
        return {"state": "stopped", "step": "evaluate", "fraction": None, "eta_seconds": None,
                "detail": f"sequential test stopped at {run['eval']['pairs']:,} pairs"}
    return {"state": "done" if run["eval"] else "idle", "step": None, "fraction": None, "eta_seconds": None, "detail": ""}


# ---------------------------------------------------------------------------------------------------------------
# the whole tree


def load_runs(runs_dir: Path, plan: dict | None = None) -> dict:
    plan = {**PLAN, **(plan or {})}
    runs_dir = Path(runs_dir)
    now = time.time()
    runs = [r for r in (load_run(runs_dir, n) for n in run_names(runs_dir)) if r]
    by_name = {r["name"]: r for r in runs}
    by_sha = {r["best"]["sha256"]: r["name"] for r in runs if r["best"] and r["best"].get("sha256")}

    for run in runs:
        parent = by_sha.get(run["init_sha256"])
        if not parent:
            m = re.search(r"runs/([^/]+)/best\.nnue$", run["init"].replace("\\", "/"))
            parent = m.group(1) if m and m.group(1) in by_name else None
        run["parent"] = parent if parent != run["name"] else None
    for run in runs:
        run["children"] = sorted((r["name"] for r in runs if r["parent"] == run["name"]), key=lambda n: by_name[n]["started"])
        key = run["init_sha256"] or run["init"]
        run["siblings"] = [r["name"] for r in runs if r is not run and key and (r["init_sha256"] or r["init"]) == key]
        run["training"] = training_stats(run)
        run["datasets"] = {m["set"]: dataset_summary(runs_dir, m["set"]) for m in run["mixture"]}
        if run["own_set"] not in run["datasets"]:
            run["datasets"][run["own_set"]] = dataset_summary(runs_dir, run["own_set"])
        depth, node = 0, run
        while node["parent"] and depth < 64:
            depth, node = depth + 1, by_name[node["parent"]]
        run["depth"] = depth
    for run in runs:
        run["status"] = run_status(run, runs_dir, now)

    on_disk = {}
    data_dir = runs_dir / "nnue_data"
    if data_dir.is_dir():
        for d in sorted(data_dir.iterdir()):
            if d.is_dir() and (d / "provenance.json").is_file():
                on_disk[d.name] = (d / "provenance.json").stat().st_mtime
    rows = games = 0
    for run in runs:
        own, gen = run["datasets"].get(run["own_set"]), run["generation"]
        if own and gen and gen["complete"] and own.get("root_rows"):
            rows, games = rows + own["root_rows"], games + gen["games"]
    plan["rows_per_game"] = rows / games if games >= 1000 else PLAN["rows_per_game"]

    for run in runs:
        run["start_label"] = start_label(run)
        run["outcome"] = outcome(run)
    for run in runs:
        run["chain"] = lineage(run, by_name)
        run["checks"] = [c.as_dict() for c in checks(run, by_name, plan, on_disk)]
    for run in runs:
        run["decision"] = decide(run, by_name, plan, on_disk)
        run["decision"]["followed_up"] = followed_up(run, by_name)

    return {
        "runs_dir": str(runs_dir.resolve()),
        "runs": runs,
        "roots": sorted((r["name"] for r in runs if not r["parent"]), key=lambda n: by_name[n]["started"]),
        "plan": plan,
        "ladder": [{"generation": i, "games": g, "rows": g * plan["rows_per_game"], "target": sprt_target(plan, i), "cap": sprt_cap(plan, g), "why": LADDER_WHY[min(i, len(LADDER_WHY) - 1)] if len(plan["ladder"]) == len(PLAN["ladder"]) else ""}
                   for i, g in enumerate(plan["ladder"])],
        "sweet_spots": SWEET_SPOTS,
        "metric_help": METRIC_HELP,
        "strata_help": STRATA_HELP,
        "depth_note": DEPTH_NOTE,
        "now": now,
    }


def outcome(run: dict) -> str:
    """adopt | null | worse | open: what this batch did for its start."""
    ev = run["eval"]
    if not ev:
        return "open"
    if ev["verdict"] in ("better", "better_small") and ev["grade"] == "decision":
        return "adopt"
    if ev["verdict"] == "worse":
        return "worse"
    if ev["verdict"] == "null" and ev["grade"] == "decision":
        return "null"
    return "open"


def lineage(run: dict, by_name: dict) -> list[dict]:
    """The networks from the root to this run with each one's measured step against its own start. Elo does not add
    exactly across steps (each is measured against a different opponent at a shallow budget), so the running total is
    a rough guide and its uncertainty grows with every step."""
    chain, node = [], run
    while node:
        chain.append(node)
        node = by_name.get(node["parent"]) if node["parent"] else None
    chain.reverse()
    steps, total, var = [], 0.0, 0.0
    root_start = chain[0]["init"] if chain else ""
    for k, node in enumerate(chain):
        ev = node["eval"]
        attempts = [node["name"], *node["siblings"]]
        tried = [by_name[n] for n in attempts]
        step = {"name": node["name"], "outcome": node["outcome"], "depth": node["depth"],
                "attempts": len(tried), "nulls": sum(1 for t in tried if t["outcome"] in ("null", "worse")),
                "elo": None, "elo_interval": None, "total": None, "total_interval": None, "sims": None, "kind": None,
                # every ancestor was used as a start; the run itself counts only once it is adopted
                "counted": k < len(chain) - 1 or node["outcome"] == "adopt"}
        direct = [a for a in node["anchors"] if not a["stale"] and a["reference_path"] == root_start]
        if direct:  # measured straight against the line's first network: no stacking error
            best = max(direct, key=lambda a: (a["pairs"], a["mtime"]))
            step["direct"] = {"elo": best["elo"], "elo_interval": best["elo_interval"], "pairs": best["pairs"],
                              "sims": best.get("sims"), "file": best["file"]}
        if ev:
            step.update({"elo": ev["elo"], "elo_interval": ev["elo_interval"], "sims": ev.get("sims"), "kind": ev["kind"]})
            if ev["elo_se"]:
                t, v = total + ev["elo"], var + ev["elo_se"] ** 2
                step.update({"total": t, "total_interval": [t - 1.96 * math.sqrt(v), t + 1.96 * math.sqrt(v)]})
                if step["counted"]:
                    total, var = t, v
        steps.append(step)
    return steps


def mixture_root_rows(run: dict) -> int | None:
    total = 0
    for m in run["mixture"]:
        info = run["datasets"].get(m["set"])
        if not info or not info.get("rows"):
            return None
        total += info["root_rows"]
    return total


def checks(run: dict, by_name: dict, plan: dict, on_disk: dict) -> list[Check]:
    out: list[Check] = []
    ev, tr, cfg, gen = run["eval"], run["training"], run["config"], run["generation"]
    s = SWEET_SPOTS

    # evaluation ------------------------------------------------------------------------------------------------
    if ev:
        lo_s, hi_s = ev["score_interval"]
        if ev["kind"] == "sequential":
            words = {"accept": f"accepted {ev['sprt_target']} over .50", "reject": f"rejected a gain of {ev['sprt_target']}",
                     "stopped": f"stopped by hand at {ev['pairs']:,} pairs",
                     "inconclusive": f"no decision within {ev['sprt_cap']:,} pairs", "running": "still running",
                     "invalid": "invalid"}
            status = {"better": "ok", "better_small": "ok", "worse": "bad", "null": "warn", "retest": "warn"}.get(ev["verdict"], "warn")
            out.append(Check("Evaluation", "Sequential test", words.get(ev["stop"], str(ev["stop"])),
                             "accept, or inconclusive with the lower bound above .50", status,
                             f"LLR {ev['llr']:+.2f} against +-{ev['llr_bound']:.2f} after {ev['pairs']:,} pairs. "
                             "The interval after a sequential stop is descriptive: an early accept tends to overstate the gain."
                             if isinstance(ev["llr"], (int, float)) else "The test's report carries no LLR."))
        else:
            out.append(Check("Evaluation", "Interval lower bound", f"{lo_s:.3f}", "above .500",
                             "ok" if lo_s > 0.5 else "bad" if hi_s < 0.5 else "warn",
                             "Above .50: better at this sample size. Straddling: not yet known."))
            pairs = ev["pairs"]
            out.append(Check("Evaluation", "Opening pairs", f"{pairs:,}", f"{plan['decision_pairs']} for a fixed-count decision",
                             "ok" if pairs >= plan["decision_pairs"] else "warn",
                             "100 pairs is a reading: a +14 Elo gain (score .52) is invisible in 200 games."))
        if ev.get("sims"):
            sims = ev["sims"]
            out.append(Check("Evaluation", "Evaluation budget", f"{sims * 2500:,} nodes",
                             f"{plan['decision_sims'] * 2500:,} nodes",
                             "ok" if sims >= plan["decision_sims"] else "warn",
                             "The loop's gain test runs at 40k nodes. Wins at 20k picked from several tries often vanished at 40k."))
        out.append(Check("Evaluation", "Resolution", f"±{ev['resolution_elo']:.0f} Elo" if ev["resolution_elo"] else "–",
                         "under +14 Elo (the test's .52)",
                         "ok" if ev["resolution_elo"] and ev["resolution_elo"] < 14 else "info",
                         "The smallest gain this sample could tell from zero."))
        b = ev.get("bootstrap_score_interval")
        if b and ev["kind"] == "fixed":
            agree = (b[0] > 0.5) == (lo_s > 0.5) and (b[1] < 0.5) == (hi_s < 0.5)
            out.append(Check("Evaluation", "Bootstrap interval", f"{b[0]:.3f} to {b[1]:.3f}", "agrees with the interval",
                             "ok" if agree else "warn", "Resampled pairs; it should tell the same story."))
        out.append(Check("Evaluation", "Forfeits", str(ev.get("forfeits", 0)), "0",
                         "ok" if not ev.get("forfeits") else "bad", "A forfeit is a crash or an illegal move, not a result."))
        if run["init"] and ev["reference_path"] and ev["reference_path"] != run["init"]:
            out.append(Check("Evaluation", "Reference", ev["reference_path"], run["init"], "warn",
                             "The evaluation was not against this run's starting network."))

    # training ----------------------------------------------------------------------------------------------------
    if tr:
        running = run["status"]["state"] == "training"
        late = tr["best_epoch"] >= (tr["epochs_planned"] - 1) * s["late_best_fraction"]
        early = tr["best_epoch"] <= s["early_best_epochs"] and tr["epochs_done"] > s["early_best_epochs"] + 2
        out.append(Check("Training", "Best epoch", f"{tr['best_epoch']} of {tr['epochs_planned'] - 1}",
                         "second half of the run",
                         "info" if running else "bad" if early and tr["rise_after_best"] > s["objective_rise_warn"]
                         else "ok" if late else "warn",
                         "An objective that bottoms in epoch 0-2 and then rises is memorising the batch."))
        rise = tr["rise_after_best"]
        out.append(Check("Training", "Objective rise after best", f"{rise:+.2%}", f"under {s['objective_rise_warn']:.1%}",
                         "ok" if rise <= s["objective_rise_warn"] else "warn" if rise <= s["objective_rise_bad"] else "bad",
                         "How far the last epoch sits above the best one."))
        out.append(Check("Training", "Loss against objective", "diverging" if tr["diverging"] else "moving together",
                         "moving together", "bad" if tr["diverging"] else "ok",
                         "Training loss falling while validation rises: more data is worth more than more epochs."))
        out.append(Check("Training", "Objective improvement", f"{tr['improvement']:.2%}", "any; context only", "info",
                         f"From epoch 0 to the best epoch, {'still falling' if tr['recent_slope'] < -0.0005 else 'flat'} "
                         f"at the end ({tr['recent_slope']:+.3%} per epoch over the last quarter)."))
        for name, d in tr["drift"].items():
            if name == run["own_set"]:
                continue
            ch = d["change_to_best"]
            out.append(Check("Training", f"Older batch {name.removeprefix('selfplay_')}", f"mse {ch:+.2%}",
                             f"not rising (under {s['drift_warn']:.0%})",
                             "ok" if ch <= s["drift_warn"] else "warn" if ch <= s["drift_bad"] else "bad",
                             "Validation error on an earlier batch; a rise means the network drifts from what it knew."))
        worst_bias = max((abs(v) for v in tr["bias_at_best"].values() if isinstance(v, (int, float))), default=None)
        if worst_bias is not None:
            out.append(Check("Training", "Worst bias at the best epoch", f"{worst_bias:.4f}", f"under {s['bias_warn']}",
                             "ok" if worst_bias < s["bias_warn"] else "warn" if worst_bias < s["bias_bad"] else "bad",
                             "Systematic optimism or pessimism against the search labels."))
        if tr["restarted"]:
            out.append(Check("Training", "log.csv", "holds more than one run", "one run", "warn",
                             "Only the last run is plotted. The current quickstart moves a changed run aside instead."))
    if cfg:
        ema = cfg.get("ema") or 0
        steps = cfg.get("steps_per_epoch") or 0
        window = 1 / (1 - ema) if 0 < ema < 1 else 0
        out.append(Check("Training", "Weight averaging", f"decay {ema:g} (about {window:,.0f} steps)" if ema else "off",
                         "about one epoch of steps",
                         "ok" if ema and steps and 0.5 * steps <= window <= 2 * steps else "info",
                         "The quickstart's EMA=auto; the loop promoted its averaged arm in round 3. Off is the older recipe."))
        rows = mixture_root_rows(run)
        if rows and steps and cfg.get("epochs") and cfg.get("batch"):
            passes = steps * cfg["epochs"] * cfg["batch"] / rows
            lo_p, hi_p = plan["passes"]
            out.append(Check("Training", "Passes over the rows", f"{passes:.1f}", f"{lo_p:.0f} to {hi_p:.0f}",
                             "ok" if lo_p <= passes <= hi_p else "bad" if passes > 2 * hi_p else "warn",
                             "epochs x steps x batch / searched-root rows. More passes memorise a batch."))
        if steps and cfg.get("epochs"):
            updates = steps * cfg["epochs"]
            out.append(Check("Training", "Optimiser updates", f"{updates:,} at batch {cfg.get('batch', '?')}", "context only", "info",
                             "Passes set how often each row is seen; the batch sets how many updates that is. On a small "
                             "continuation fewer updates keep more of the start: in a direct match here, a first batch "
                             "retrained at batch 2,048 (about 1,350 updates) played weaker than the same data at 8,192."))
        if cfg.get("version"):
            out.append(Check("Training", "Network format", f"format {cfg['version']}", "8 (9 is untested for strength)",
                             "ok" if cfg["version"] == 8 else "info",
                             "Format 9 adds goal-corner rows and eight heads; no run has measured it yet."))

    # data --------------------------------------------------------------------------------------------------------
    sets = [m["set"] for m in run["mixture"]]
    if sets:
        out.append(Check("Data", "Batches in the mixture", str(len(sets)), "every batch of the line",
                         "ok" if len(sets) >= 2 or run["depth"] == 0 and not run["siblings"] else "warn",
                         "A continuation trained on one small batch alone drifts away from what it knew."))
        rows = mixture_root_rows(run)
        if rows:
            out.append(Check("Data", "Root rows in the mixture", f"{rows / 1e6:.2f} M", "grows with every batch", "info",
                             "The loop's rounds trained on 2.7 to 12 M fresh rows beside 13 M older ones."))
        kept = set(sets)
        missing = sorted(n for n in line_sets(run, by_name, on_disk) if n not in kept and n != run["own_set"]
                         and on_disk.get(n, 1e18) < run["started"])
        out.append(Check("Data", "Earlier batches kept", "all" if not missing else "missing " +
                         ", ".join(n.removeprefix("selfplay_") for n in missing), "every batch from this line", 
                         "ok" if not missing else "warn", "Keep every batch; the next command passes them all in EXTRA_DATA."))
    for sib in run["siblings"]:
        other = by_name[sib]
        ours, theirs = (gen or {}).get("seed"), (other["generation"] or {}).get("seed")
        if ours is not None and ours == theirs:
            out.append(Check("Data", "Seed against siblings", str(ours), f"different from {sib}", "bad",
                             "Two batches from the same start with the same seed are the same games."))
    if gen:
        total = f"{gen['games_target']:,}" if isinstance(gen["games_target"], int) else "?"
        out.append(Check("Data", "Self-play batch", f"{gen['games_done']:,} of {total} games", "complete",
                         "ok" if gen["complete"] else "info", "An unfinished batch resumes when the same command runs again."))
        if isinstance(gen["games_target"], int):
            g, want = gen["games_target"], ladder_games(plan, run["depth"])
            out.append(Check("Data", "Batch size", f"{g:,} games", f"{want:,} or more at generation {run['depth']}",
                             "ok" if g >= want else "warn" if g < want / 2 else "info",
                             "The ladder grows with the lineage: a stronger network needs more games to show a gain."))
        if gen["nodes"]:
            n = gen["nodes"]
            out.append(Check("Data", "Label budget", f"{n:,} nodes", f"{plan['nodes']:,}",
                             "ok" if n == plan["nodes"] else "warn" if n < plan["nodes_floor"] else "info",
                             "The loop moved to 50k after 25k-labelled gains faded at depth; 100k costs twice the time. "
                             "Batches of different budgets stay separate sets."))
        if gen.get("multipv", 1) > 1 or gen.get("pv_labels"):
            out.append(Check("Data", "Generation extras", f"multipv {gen.get('multipv', 1)}, PV labels {gen.get('pv_labels', 0)}",
                             "optional", "info", "The loop's settings, adopted without their own A/B."))
        out.append(Check("Data", "Censored games", f"{gen['censored_share']:.1%}", "low", "info",
                         "Games without a real result; their rows keep search labels but no outcome."))
    own = run["datasets"].get(run["own_set"])
    if own and own.get("quiet"):
        q = own["quiet"]
        dropped = sum(q["skipped"].values())
        out.append(Check("Data", "Quiet filter", f"{dropped:,} roots dropped, {q['kept']:,} kept", "optional", "info",
                         "QUIET=1: captures, goal threats and mate scores left out. Its strength effect is not measured yet."))
    return out


# ---------------------------------------------------------------------------------------------------------------
# the decision


def base_name(name: str) -> str:
    return re.sub(r"_v\d+$", "", name)


def version_bump(name: str, taken: set[str]) -> str:
    base = base_name(name)
    m = re.search(r"_v(\d+)$", name)
    n = int(m.group(1)) + 1 if m else 2
    while f"{base}_v{n}" in taken:
        n += 1
    return f"{base}_v{n}"


def next_generation(name: str, taken: set[str], depth: int) -> str:
    """first -> second, second_v3 -> third; any other name keeps its stem and counts generations: line_g2."""
    base = re.sub(r"_g\d+$", "", base_name(name))
    candidate = ORDINALS[ORDINALS.index(base) + 1] if base in ORDINALS[:-1] else f"{base}_g{depth}"
    return candidate if candidate not in taken else version_bump(candidate, taken)


def ladder_games(plan: dict, depth: int) -> int:
    ladder = plan["ladder"]
    return ladder[min(depth, len(ladder) - 1)]


def line_sets(run: dict, by_name: dict, on_disk: dict) -> dict[str, float]:
    """Every batch this line has produced: the mixtures from the root to this run and every batch played from one of
    the line's starts (a dropped network's games stay). Only sets that exist on disk."""
    chain, node = [], run
    while node:
        chain.append(node)
        node = by_name.get(node["parent"]) if node["parent"] else None
    seen: dict[str, float] = {}
    for node in reversed(chain):
        for m in node["mixture"]:
            seen.setdefault(m["set"], m["share"])
        for other in [node, *(by_name[n] for n in node["siblings"])]:
            seen.setdefault(other["own_set"], 1.0)
    return {n: w for n, w in seen.items() if n in on_disk}


def settings_env(run: dict) -> list[str]:
    """Settings of this run that differ from the quickstart's defaults and should carry over."""
    cfg = run["config"]
    parts = []
    if cfg.get("device") and cfg["device"] != "cpu":
        parts.append(f"DEVICE={cfg['device']}")
    if cfg.get("batch") and cfg["batch"] != 8192:
        parts.append(f"BATCH={cfg['batch']}")
    if cfg.get("version") == 9:
        parts.append("VERSION=9")
    if (run["datasets"].get(run["own_set"]) or {}).get("quiet"):
        parts.append("QUIET=1")
    if (run["generation"] or {}).get("multipv", 1) > 1:
        parts.append("MULTIPV=2")
    return parts


def sprt_cap(plan: dict, games: int, nodes: int | None = None) -> int:
    """Pairs a sequential test may play: a third of the batch's search (a pair is two games at the decision budget),
    between 128 and 3,008 in steps of 16. Near its midpoint a test settles in a few hundred pairs anyway; the cap only
    trims the long tail, and a test stopped there still adopts when its lower bound clears .50."""
    budget = games * (nodes or plan["nodes"]) / (plan["decision_sims"] * 2500) / 2 * plan["eval_share"]
    return int(min(plan["sprt_cap"], max(128, round(budget / 16) * 16)))


def sprt_target(plan: dict, depth: int) -> float:
    targets = plan.get("sprt_targets") or [plan["sprt_target"]]
    return float(targets[min(max(depth, 0), len(targets) - 1)])


def next_pipeline(run: dict, start: str, name: str, games: int, extra: dict[str, float], plan: dict, depth: int) -> dict:
    """The same step as quickstart_command, as settings the dashboard's runner starts directly."""
    cfg = run["config"]
    return {"kind": "pipeline", "run": name, "settings": {
        "start": start, "name": name, "games": games, "nodes": plan["nodes"],
        "extra": [{"set": n, "share": w} for n, w in extra.items()],
        "epochs": "auto", "batch": cfg.get("batch") or 8192, "lr": cfg.get("lr") or 0.0001,
        "device": cfg.get("device") or "cpu", "version": "9" if cfg.get("version") == 9 else "auto",
        "quiet": bool((run["datasets"].get(run["own_set"]) or {}).get("quiet")),
        "multipv": (run["generation"] or {}).get("multipv", 1),
        "eval": {"mode": "sprt", "sims": plan["decision_sims"], "pairs": plan["decision_pairs"],
                 "target": sprt_target(plan, depth), "cap": sprt_cap(plan, games)}}}


def quickstart_command(run: dict, start: str, name: str, games: int, extra: dict[str, float], plan: dict, depth: int) -> str:
    target = sprt_target(plan, depth)
    parts = [f"GAMES={games}", f"NODES={plan['nodes']}", "SPRT=1", f"SIMS={plan['decision_sims']}",
             *([f"SPRT_TARGET={target:g}"] if abs(target - 0.52) > 1e-9 else []),
             *([f"SPRT_CAP={sprt_cap(plan, games)}"] if sprt_cap(plan, games) != 3008 else []), *settings_env(run)]
    if extra:
        parts.append('EXTRA_DATA="' + " ".join(f"{n}:{w:g}" for n, w in extra.items()) + '"')
    return " ".join(parts) + f" \\\n  models/nnue/quickstart.sh {start} {name}"


def resume_command(run: dict, by_name: dict) -> str | None:
    """The quickstart command that continues this run where it stopped, rebuilt from its manifest and config so that
    the batch and the training resume instead of starting over."""
    gen, cfg = run["generation"], run["config"]
    start = cfg.get("init") or (gen or {}).get("init")
    if not start:
        return None
    parts = []
    if gen:
        parts += [f"GAMES={gen['games_target']}", f"NODES={gen['nodes']}"]
    if cfg:
        extra = [m for m in run["mixture"] if m["set"] != run["own_set"]]
        parts += [f"SEED={cfg['seed']}", f"EPOCHS={cfg['epochs']}", f"STEPS={cfg['steps_per_epoch']}",
                  f"WARMUP={cfg['warmup_steps']}", f"EMA={cfg.get('ema') or 0:g}", f"THREADS={cfg['threads']}"]
        if cfg.get("lr") not in (None, 0.0001):
            parts.append(f"LR={cfg['lr']:g}")
        if extra:
            parts.append('EXTRA_DATA="' + " ".join(f"{m['set']}:{m['share']:g}" for m in extra) + '"')
    elif gen and gen.get("seed") is not None:
        parts.append(f"SEED={gen['seed']}")
    parts += ["SPRT=1", f"SIMS={PLAN['decision_sims']}", *settings_env(run)]
    return " ".join(parts) + f" \\\n  models/nnue/quickstart.sh {start} {run['name']}"


def sprt_eval_command(run: dict, plan: dict, target: float = 0.52, cap: int = 3008) -> str:
    ev = run["eval"] or {}
    cand = ev.get("candidate") or f"nnue:runs/{run['name']}/best.nnue?hash=64"
    ref = ev.get("reference") or f"nnue:{run['init'] or '<start.nnue>'}?hash=64"
    threads = min(run["config"].get("threads") or 8, plan["eval_threads_max"])
    seed = run["config"].get("seed") or 1
    tag = "" if abs(target - 0.52) < 1e-9 else f"_t{round(target * 1000):03d}"
    journal = f"runs/{run['name']}/sprt_s{plan['decision_sims']}{tag}.jsonl"
    resume = " --resume" if f"sprt_s{plan['decision_sims']}{tag}.jsonl" in run["journals"] else ""
    tag += "" if cap == 3008 else f"_c{cap}"
    flag = (f" --sprt-target {target:g}" if abs(target - 0.52) > 1e-9 else "") + (f" --sprt-cap {cap}" if cap != 3008 else "")
    return (f'target/release/bot eval --candidate "{cand}" --reference "{ref}" \\\n  --sims {plan["decision_sims"]} '
            f"--threads {threads} --seed {seed} --sprt {journal}{flag}{resume} \\\n  > runs/{run['name']}/eval_sprt_s{plan['decision_sims']}{tag}.json")


def followed_up(run: dict, by_name: dict) -> str | None:
    verdict = run["decision"]["verdict"]
    if run["children"]:
        return f"Already continued by {', '.join(run['children'])}."
    later = [n for n in run["siblings"] if by_name[n]["started"] > run["started"]]
    if verdict in ("more_data", "abort", "reeval") and later:
        return f"Already followed by {', '.join(sorted(later))} from the same start."
    return None


def start_label(run: dict) -> str:
    if run["parent"]:
        return run["parent"]
    m = re.search(r"runs/([^/]+)/best\.nnue$", run["init"].replace("\\", "/"))
    return m.group(1) if m else (Path(run["init"]).name or "the previous start")


def decide(run: dict, by_name: dict, plan: dict, on_disk: dict) -> dict:
    ev, status = run["eval"], run["status"]
    taken = set(by_name) | {n.removeprefix("selfplay_") for n in on_disk} | set(plan.get("reserved_names", []))
    start = run["init"] or "<start.nnue>"
    reasons: list[str] = []
    unhealthy = [c for c in run["checks"] if c["status"] == "bad" and c["group"] != "Evaluation"]

    gen = run["generation"] or {}
    this_batch = {"games": gen.get("games_target") or ladder_games(plan, run["depth"]), "generation": run["depth"],
                  "base": ladder_games(plan, run["depth"]), "nulls": 0, "current": True,
                  "rows": (gen.get("games_target") or ladder_games(plan, run["depth"])) * plan["rows_per_game"], "why": ""}

    def result(verdict, title, summary, command=None, label=None, extra=None, batch=None, next_step=None):
        return {"verdict": verdict, "title": title, "summary": summary, "reasons": reasons, "command": command,
                "command_label": label, "extra_commands": extra or [], "batch": batch or this_batch, "next": next_step}

    own_target = sprt_target(plan, run["depth"])
    own_cap = sprt_cap(plan, (run["generation"] or {}).get("games_target") or ladder_games(plan, run["depth"]),
                       (run["generation"] or {}).get("nodes"))
    sprt_next = {"kind": "eval", "run": run["name"], "settings": {"mode": "sprt", "sims": plan["decision_sims"],
                                                                "target": own_target, "cap": own_cap}}

    def retry_batch() -> dict:
        tried = [run, *(by_name[n] for n in run["siblings"])]
        nulls = sum(1 for t in tried if t["outcome"] in ("null", "worse"))
        base = ladder_games(plan, run["depth"])
        games = base * min(2 ** nulls, plan["retry_growth_max"])
        why = (f"Generation {run['depth']} plans {base:,} games; {nulls} batch{'es' if nulls != 1 else ''} from this start "
               f"showed no gain, so the next one is {games // base}x the plan (the loop doubles after each null, up to 4x).")
        return {"games": games, "rows": games * plan["rows_per_game"], "generation": run["depth"], "base": base,
                "nulls": nulls, "why": why}

    state = status["state"]
    if state in ("generating", "preparing", "training", "evaluating"):
        words = {"generating": "Generating games", "preparing": "Importing the batch", "training": "Training",
                 "evaluating": "Evaluating"}
        reasons.append("The decision waits for the evaluation at the end of the quickstart run.")
        return result("running", f"{words[state]}", status["detail"].capitalize() + ".")
    if state == "stopped":
        reasons.append("Nothing has been written for a while: the run was interrupted. The same command continues it; "
                       "this one is rebuilt from the batch's manifest and the run's config so nothing starts over.")
        resume = None
        if run["config"] or run["generation"]:
            cfg, gen = run["config"], run["generation"] or {}
            resume = {"kind": "pipeline", "run": run["name"], "settings": {
                "start": cfg.get("init") or gen.get("init"), "name": run["name"],
                "games": gen.get("games_target") or ladder_games(plan, run["depth"]), "nodes": gen.get("nodes") or plan["nodes"],
                "extra": [m for m in run["mixture"] if m["set"] != run["own_set"]],
                "seed": cfg.get("seed", gen.get("seed")), "epochs": cfg.get("epochs") or 20,
                "steps": cfg.get("steps_per_epoch"), "warmup": cfg.get("warmup_steps"),
                "ema": cfg["ema"] if cfg.get("ema") else ("off" if cfg else "auto"),
                "batch": cfg.get("batch") or 8192, "lr": cfg.get("lr") or 0.0001, "device": cfg.get("device") or "cpu",
                "version": "9" if cfg.get("version") == 9 else "auto", "multipv": gen.get("multipv", 1),
                "eval": {"mode": "sprt", "sims": plan["decision_sims"], "pairs": plan["decision_pairs"],
                         "target": sprt_target(plan, run["depth"]), "cap": plan["sprt_cap"]}}}
        return result("stopped", "Stopped before the end", status["detail"].capitalize() + ".",
                      resume_command(run, by_name), "Continue where it stopped", next_step=resume)

    if not ev:
        if run["has_net"]:
            reasons.append(f"Nothing is adopted or dropped without an evaluation: the sequential test at "
                           f"{plan['decision_sims'] * 2500:,} nodes decides.")
            return result("evaluate", "Evaluate before deciding", "best.nnue exists but no current evaluation of it.",
                          sprt_eval_command(run, plan, own_target, own_cap), "Run the sequential test", next_step=sprt_next)
        return result("incomplete", "Nothing to decide yet", "This run has no network and no evaluation.")

    lo_s, hi_s = ev["score_interval"]
    budget = f"{ev.get('sims', 0) * 2500:,} nodes"
    reasons.append(f"Score {ev['score']:.3f} over {ev['pairs']:,} pairs at {budget}: {ev['elo']:+.0f} Elo "
                   f"({ev['elo_interval'][0]:+.0f} to {ev['elo_interval'][1]:+.0f}).")

    if ev["verdict"] in ("running", "invalid"):
        reasons.append("The sequential test did not finish. Rerunning it with --resume continues the same journal."
                       if ev["verdict"] == "running" else "The sequential test is invalid (a forfeit, a ply cap or an "
                       "interruption inside a game); run it again with a new journal.")
        return result("reeval", "Finish the sequential test", "The evaluation stopped before its own rule decided.",
                      sprt_eval_command(run, plan, ev.get("sprt_target") or own_target, ev.get("sprt_cap") or own_cap), "Continue the sequential test",
                      next_step={**sprt_next, "settings": {**sprt_next["settings"], "target": ev.get("sprt_target") or own_target,
                                                           "cap": ev.get("sprt_cap") or plan["sprt_cap"]}})

    if ev["kind"] == "fixed" and ev["grade"] == "reading" and ev["verdict"] != "worse":
        why = ("Wins seen at a shallower budget were often gone at 40k nodes in the research loop"
               if (ev.get("sims") or 0) < plan["decision_sims"] else f"{ev['pairs']} pairs cannot tell a +14 Elo gain from noise")
        reasons.append(f"This is a reading ({ev['pairs']} pairs at {budget}), not a decision. {why}; the sequential test "
                       "settles it in 128 to 3,008 pairs, stopping early when the answer is clear.")
        return result("reeval", "Confirm with the sequential test", "Test the network at the decision budget before "
                      "spending hours on the next batch.", sprt_eval_command(run, plan, own_target, own_cap), "Run the sequential test",
                      next_step=sprt_next)

    if ev["verdict"] == "retest":
        base = plan["sprt_target"]
        reasons.append(f"The test at the raised target {ev['sprt_target']} (generation {run['depth']}) rejected a gain that large, "
                       f"but the interval's lower bound is {lo_s:.3f}, above .50: the network looks better than its start by less "
                       "than the raised target.")
        reasons.append(f"Raised targets settle early generations quickly; a rejection there that still shows a gain is tested again "
                       f"at the base target {base} before a batch is spent on it.")
        retest = {**sprt_next, "settings": {**sprt_next["settings"], "target": base}}
        return result("reeval", f"Test {run['name']} again at {base}", "A smaller gain than the raised target asked for, but likely real.",
                      sprt_eval_command(run, plan, base, own_cap), f"Run the sequential test at {base}", next_step=retest)

    if ev["verdict"] in ("better", "better_small"):
        if ev["verdict"] == "better":
            reasons.append(f"The sequential test accepted a gain of {ev['sprt_target']} over .50."
                           if ev["kind"] == "sequential" else "The interval's lower bound is above .50 at the decision budget.")
        else:
            reasons.append((f"You stopped the test at {ev['pairs']:,} pairs" if ev["stop"] == "stopped" else
                            f"The test stopped at its cap of {ev['sprt_cap']:,} pairs") + " without deciding, but the lower bound "
                           f"is {lo_s:.3f}: a real gain smaller than the test's +14 Elo. The loop's cap_lower rule "
                           "promotes such a network.")
        if ev["kind"] == "sequential" and ev["verdict"] == "better":
            reasons.append("The Elo after an early accept is optimistic; the next generation's test is the real check.")
        if unhealthy:
            reasons.append("Some training checks failed; the win stands, but look at them before the next batch.")
        depth = run["depth"] + 1
        games = ladder_games(plan, depth)
        extra = line_sets(run, by_name, on_disk)
        nxt = next_generation(run["name"], taken, depth)
        batch = {"games": games, "rows": games * plan["rows_per_game"], "generation": depth, "base": games, "nulls": 0,
                 "why": f"Generation {depth} of this line plans {games:,} games, decided at {sprt_target(plan, depth)}"
                        + (f", {LADDER_WHY[min(depth, len(LADDER_WHY) - 1)]}." if plan["ladder"] == PLAN["ladder"] else ".")}
        reasons.append(batch["why"])
        return result("adopt", f"Adopt {run['name']} as the next start",
                      f"Continue self-play from runs/{run['name']}/best.nnue with every batch kept.",
                      quickstart_command(run, f"runs/{run['name']}/best.nnue", nxt, games, extra, plan, depth),
                      "Next generation", batch=batch,
                      next_step=next_pipeline(run, f"runs/{run['name']}/best.nnue", nxt, games, extra, plan, depth))

    batch = retry_batch()
    extra = line_sets(run, by_name, on_disk)
    name = version_bump(run["name"], taken)

    if ev["verdict"] == "worse":
        reasons.append("The upper bound is below .50: this network is worse than its start.")
        if unhealthy:
            reasons.extend(f"{c['label']}: {c['value']}, looking for {c['target']}. {c['note']}" for c in unhealthy)
            reasons.append("Fix the training before blaming the data: PASSES at 4 and every earlier batch in the mixture.")
        else:
            reasons.append("Training looks healthy, so the batch did not help on its own; keep it and add another.")
        reasons.append(batch["why"])
        return result("abort", f"Drop {run['name']}, go back to {start_label(run)}",
                      "Keep its games in the mixture and play another batch from the previous network.",
                      quickstart_command(run, start, name, batch["games"], extra, plan, run["depth"]),
                      "Another batch from the previous start", batch=batch,
                      next_step=next_pipeline(run, start, name, batch["games"], extra, plan, run["depth"]))

    reasons.append(f"The test rejected a gain of {ev['sprt_target']}: no gain large enough to see."
                   if ev["kind"] == "sequential" and ev["stop"] == "reject" else
                   "No decision within the cap and the interval still straddles .50."
                   if ev["kind"] == "sequential" else "The interval straddles .50 at the decision budget.")
    if ev["pairs_needed"] and ev["pairs_needed"] > ev["pairs"]:
        reasons.append(f"If {ev['elo']:+.0f} Elo were real, about {ev['pairs_needed']:,} pairs would show it; "
                       "more data usually decides it sooner.")
    if unhealthy:
        reasons.extend(f"{c['label']}: {c['value']}, looking for {c['target']}. {c['note']}" for c in unhealthy)
    reasons.append(batch["why"])
    return result("more_data", f"Get more data from {start_label(run)}",
                  f"Another batch from the same start, trained together with every batch so far.",
                  quickstart_command(run, start, name, batch["games"], extra, plan, run["depth"]),
                  "Another batch from the same start", batch=batch,
                  next_step=next_pipeline(run, start, name, batch["games"], extra, plan, run["depth"]))
