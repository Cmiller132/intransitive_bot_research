"""The sequential test's state while it plays, read from its journal: the same pentanomial likelihood ratio as
match/src/sprt.rs, so the page shows the number the bot stops on.

A journal is a header line (the protocol, with the test's s0, s1 and cap) and one line per finished opening pair,
{"pair": n, "games": [first, second]}: the candidate plays first in the first game and second in the other. A pair
scores 0 to 4 points for the candidate (2 a win, 1 a draw); the counts of the five scores are the test's evidence.
The bot updates its decision only at checkpoints (every 16 pairs from 128); between them the value shown here is the
same formula on the pairs so far.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)
BOUND = 2.9444389791664403  # ln(19): alpha = beta = 0.05
BATCH, FIRST_CHECK = 16, 128
ZERO_CELL = 1e-3


def _multiplier(pdf: list[float], s: float) -> float:
    low, high = -1.0 / (1.0 - s), 1.0 / s
    for _ in range(80):
        mid = low + (high - low) * 0.5
        if mid == low or mid == high:
            break
        residual = sum(pdf[i] * (SCORES[i] - s) / (1.0 + mid * (SCORES[i] - s)) for i in range(5))
        if residual > 0:
            low = mid
        else:
            high = mid
    return low + (high - low) * 0.5


def llr(counts: list[int], s0: float = 0.5, s1: float = 0.52) -> float | None:
    if not any(counts):
        return None
    weights = [float(n) if n else ZERO_CELL for n in counts]
    total = sum(weights)
    pdf = [w / total for w in weights]
    lambda0, lambda1 = _multiplier(pdf, s0), _multiplier(pdf, s1)
    value = sum(weights[i] * (math.log1p(lambda0 * (SCORES[i] - s0)) - math.log1p(lambda1 * (SCORES[i] - s1))) for i in range(5))
    return value if math.isfinite(value) else None


class JournalReader:
    """Reads a growing journal from where it left off, so a 3,008-pair test costs its parse once."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 0
        self.counts = [0, 0, 0, 0, 0]
        self.pairs = 0
        self.invalid = None
        self.test = {"s0": 0.5, "s1": 0.52, "cap": 3008}
        self.first_seen: tuple[float, int] | None = None
        self.wdl = [0, 0, 0]          # the candidate's wins, draws, losses
        self.plies = 0

    def read(self) -> dict | None:
        try:
            with self.path.open("rb") as f:
                f.seek(self.offset)
                data = f.read()
        except OSError:
            return None
        end = data.rfind(b"\n")
        if end >= 0:
            for raw in data[: end + 1].split(b"\n")[:-1]:
                self.offset += len(raw) + 1
                try:
                    entry = json.loads(raw)
                except ValueError:
                    continue
                if "protocol" in entry:
                    test = (entry.get("protocol") or {}).get("test") or {}
                    self.test.update({k: test[k] for k in ("s0", "s1", "cap") if k in test})
                    continue
                games = entry.get("games") or []
                if len(games) != 2:
                    continue
                cell = 0
                for game, candidate in zip(games, (0, 1)):
                    if game.get("end") in ("Forfeit", "PlyCap", "Interrupted"):
                        self.invalid = f"pair {entry.get('pair')}: {game.get('end')}"
                    winner = game.get("winner")
                    cell += 2 if winner == candidate else 1 if winner is None else 0
                    self.wdl[0 if winner == candidate else 1 if winner is None else 2] += 1
                    self.plies += int(game.get("plies") or 0)
                self.counts[cell] += 1
                self.pairs += 1
        now = time.time()
        if self.first_seen is None:
            self.first_seen = (now, self.pairs)
        return self.state(now)

    def state(self, now: float) -> dict:
        s0, s1, cap = float(self.test["s0"]), float(self.test["s1"]), int(self.test["cap"])
        value = llr(self.counts, s0, s1) if self.pairs else None
        closeness = max(abs(value) / BOUND if value is not None else 0.0, self.pairs / cap if cap else 0.0)
        seen_at, seen_pairs = self.first_seen or (now, self.pairs)
        rate = (self.pairs - seen_pairs) / (now - seen_at) if now - seen_at > 20 and self.pairs > seen_pairs else None
        next_check = max(FIRST_CHECK, (self.pairs // BATCH + 1) * BATCH) if self.pairs < cap else cap
        points = sum(i * n for i, n in enumerate(self.counts))
        forecast = self.forecast(value, s0, s1, cap)
        return {
            "pairs": self.pairs, "cap": cap, "s0": s0, "s1": s1, "counts": list(self.counts), "llr": value, "bound": BOUND,
            "closeness": min(1.0, closeness), "lean": None if value is None else "stronger" if value > 0 else "weaker",
            "next_check": next_check, "invalid": self.invalid,
            "score": points / (4 * self.pairs) if self.pairs else None, "games": self.pairs * 2,
            "forecast": forecast, "pairs_per_second": rate,
            "eta_seconds": forecast["median_pairs"] / rate if forecast and rate else None,
        }

    def forecast(self, value, s0: float, s1: float, cap: int) -> dict | None:
        """Where the test is likely to stop from here. The ratio is a random walk: its drift is the gain, its spread the
        noise of the pairs, so it reaches a bound even with no drift at all. A few hundred walks continue it check by
        check to a bound or the cap. The true score is drawn around the observed one for each walk, so an early, noisy
        score gives a wide forecast rather than a confident wrong one."""
        n = self.pairs
        if value is None or n < 48:
            return None
        key = (n, cap, s0, s1)
        if getattr(self, "_forecast_key", None) == key:
            return self._forecast
        mean = sum(c * v for c, v in zip(self.counts, SCORES)) / n
        var = max(sum(c * (v - mean) ** 2 for c, v in zip(self.counts, SCORES)) / (n - 1), 1e-4)
        rng = random.Random(n)
        mid, delta = (s0 + s1) / 2, s1 - s0
        stops, accepted, capped = [], 0, 0
        for _ in range(400):
            true = rng.gauss(mean, math.sqrt(var / n))
            x, pairs = value, n
            while True:
                step = max(FIRST_CHECK, (pairs // BATCH + 1) * BATCH) - pairs if pairs < FIRST_CHECK else BATCH
                step = min(step, cap - pairs)
                if step <= 0:
                    capped += 1
                    break
                # the ratio moves by delta * (score - mid) / var per pair; its noise follows the pair scores' spread
                x += step * delta * (true - mid) / var + rng.gauss(0.0, math.sqrt(step) * delta / math.sqrt(var))
                pairs += step
                if x >= BOUND:
                    accepted += 1
                    break
                if x <= -BOUND:
                    break
            stops.append(pairs - n)
        stops.sort()
        result = {"accept_chance": accepted / 400, "cap_chance": capped / 400,
                  "median_pairs": stops[len(stops) // 2], "p90_pairs": stops[int(0.9 * len(stops))]}
        self._forecast_key, self._forecast = key, result
        return result
