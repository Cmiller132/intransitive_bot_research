"""Single source of truth for move-quality labels and accuracy."""

from __future__ import annotations

import math

BEST = 0.005
EXCELLENT = 0.02
GOOD = 0.05
INACCURACY = 0.10
MISTAKE = 0.20
ONLY_GAP = 0.15


def accuracy(loss: float) -> float:
    """Map expected-score loss onto the frozen chess-style 0-100 scale."""
    value = 103.17 * math.exp(-0.04 * 100 * max(0.0, min(1.0, loss))) - 3.17
    return round(max(0.0, min(100.0, value)), 2)


def label(
    loss: float,
    *,
    played_is_top: bool = False,
    best_score: float | None = None,
    played_score: float | None = None,
    second_loss: float | None = None,
    book: bool = False,
) -> str:
    """Classify a move, applying the contract's replacement/annotation rules."""
    loss = max(0.0, min(1.0, float(loss)))
    if book:
        return "book"
    if best_score is not None and played_score is not None and best_score >= 0.90 and played_score < 0.75:
        return "missed_win"
    if (played_is_top or loss < BEST) and second_loss is not None and second_loss >= ONLY_GAP:
        return "only"
    if played_is_top or loss < BEST:
        return "best"
    if loss < EXCELLENT:
        return "excellent"
    if loss < GOOD:
        return "good"
    if loss < INACCURACY:
        return "inaccuracy"
    if loss < MISTAKE:
        return "mistake"
    return "blunder"
