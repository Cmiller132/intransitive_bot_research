"""A test double for the bot binary: random games and a uniform analyser."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arena.core.game import (  # noqa: E402
    STAGNATION_PLIES,
    initial_board_list,
    legal_moves,
    terminal_after,
    terminal_at,
)
from arena.core.notation import move_token, parse_token  # noqa: E402

END_NAMES = {"corner": "Goal", "no_pieces": "Elimination", "no_moves": "Stalemate", "stagnation": "CaptureClock"}
MAX_PLIES = 4 * STAGNATION_PLIES


def apply_move(board: list[int], move: tuple[int, int]) -> bool:
    """Play one move on a board; returns whether it captured."""
    capture = board[move[1]] != 0
    board[move[1]], board[move[0]] = board[move[0]], 0
    return capture


def fake_info(rng: random.Random, moves: list[tuple[int, int]], chosen: tuple[int, int], sims: int) -> dict[str, Any]:
    """Invent plausible search statistics from the mover's point of view."""
    weights = [rng.random() + 0.1 for _ in moves]
    total = sum(weights)
    ranked = sorted(zip(moves, weights, strict=True), key=lambda item: -item[1])[:5]
    value = round(rng.uniform(-0.6, 0.6), 4)
    return {
        "sims": sims,
        "value": value,
        "plies_left": round(rng.uniform(10.0, 80.0), 1),
        "q": round(value + rng.uniform(-0.1, 0.1), 4),
        "pi": round(weights[moves.index(chosen)] / total, 4),
        "exact_win": False,
        "top": [
            {
                "move": move_token(move),
                "visits": max(1, int(sims * weight / total)),
                "q": round(value + rng.uniform(-0.2, 0.2), 4),
            }
            for move, weight in ranked
        ],
    }


def play_game(
    rng: random.Random, opening: list[tuple[int, int]], sims: int, emit: Any, pair: int, index: int
) -> tuple[int | None, str, int]:
    """Play one random game, streaming its events; returns the outcome."""
    board = initial_board_list()
    for move in opening:
        apply_move(board, move)
    psc = 0
    side = "blue" if len(opening) % 2 == 0 else "red"
    ply = len(opening)
    while ply < MAX_PLIES:
        moves = legal_moves(board, side)
        if not moves:
            return (1 if side == "blue" else 0), "Stalemate", ply
        chosen = rng.choice(moves)
        info = fake_info(rng, moves, chosen, sims)
        emit({"event": "move", "pair": pair, "game": index, "ply": ply, "move": move_token(chosen), "info": info})
        capture = apply_move(board, chosen)
        psc = 0 if capture else psc + 1
        result = terminal_after(board, side, psc)
        ply += 1
        if result is not None:
            winner = None if result.winner is None else (0 if result.winner == "blue" else 1)
            return winner, END_NAMES[result.reason], ply
        side = "red" if side == "blue" else "blue"
    return None, "CaptureClock", ply


def random_opening(rng: random.Random, plies: int) -> list[tuple[int, int]]:
    """Play a random legal opening from the initial position."""
    board = initial_board_list()
    side = "blue"
    opening = []
    for _ in range(plies):
        moves = legal_moves(board, side)
        if not moves:
            break
        move = rng.choice(moves)
        opening.append(move)
        apply_move(board, move)
        side = "red" if side == "blue" else "blue"
    return opening


def run_eval(options: argparse.Namespace) -> int:
    """Play the requested pairs and stream their events."""

    def emit(payload: dict[str, Any]) -> None:
        """Write one event line."""
        sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    wins = draws = losses = 0
    for pair in range(options.pairs):
        opening = random_opening(random.Random(options.seed * 1000 + pair), options.opening_plies)
        for index in range(2):
            rng = random.Random(options.seed * 10000 + pair * 10 + index)
            first = options.candidate if index == 0 else options.reference
            second = options.reference if index == 0 else options.candidate
            if options.stream:
                emit(
                    {
                        "event": "start",
                        "pair": pair,
                        "game": index,
                        "first": first,
                        "second": second,
                        "opening": [move_token(move) for move in opening],
                    }
                )
            winner, end, plies = play_game(
                rng, opening, options.sims, emit if options.stream else (lambda _payload: None), pair, index
            )
            if options.stream:
                emit({"event": "end", "pair": pair, "game": index, "winner": winner, "end": end, "plies": plies})
            if winner is None:
                draws += 1
            elif (winner == 0) == (index == 0):
                wins += 1
            else:
                losses += 1
    if options.stream:
        emit(
            {
                "event": "report",
                "pairs": options.pairs,
                "sims": options.sims,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "margin": wins - losses,
            }
        )
    return 0


def resolve(request: dict[str, Any]) -> tuple[list[int], str, int, int]:
    """Replay a request's moves, returning the board, side, psc and ply."""
    setup = request.get("setup")
    board = list(setup) if setup is not None else initial_board_list()
    if len(board) != 81 or any(not isinstance(cell, int) or not 0 <= cell <= 6 for cell in board):
        raise ValueError("setup must be 81 cells encoded 0..6")
    side = str(request.get("to_move") or "blue")
    if side not in ("blue", "red"):
        raise ValueError("to_move must be blue or red")
    psc = int(request.get("psc") or 0)
    ply = int(request.get("ply") or 0)
    for token in request.get("moves") or []:
        move = parse_token(str(token))
        if move not in legal_moves(board, side):
            raise ValueError(f"illegal move {token}")
        psc = 0 if apply_move(board, move) else psc + 1
        side = "red" if side == "blue" else "blue"
        ply += 1
    return board, side, psc, ply


def analyse_one(request: dict[str, Any]) -> dict[str, Any]:
    """Answer one analyse request with a uniform policy and zero values."""
    board, side, psc, ply = resolve(request)
    sims = int(request.get("sims") or 0)
    result = terminal_at(board, side, psc)
    response: dict[str, Any] = {
        "id": request.get("id"),
        "to_move": side,
        "ply": ply,
        "psc": psc,
        "legal": [],
        "terminal": None,
        "network": None,
        "search": None,
        "error": None,
    }
    if result is not None:
        response["terminal"] = {"winner": result.winner, "reason": result.reason}
        return response
    moves = legal_moves(board, side)
    tokens = [move_token(move) for move in moves]
    response["legal"] = tokens
    share = 1.0 / len(tokens)
    response["network"] = {
        "policy": dict.fromkeys(tokens, round(share, 6)),
        "q": dict.fromkeys(tokens, 0.0),
        "value": 0.0,
        "value_source": "policy_weighted_q",
        "plies_left": 40.0,
        "plies_to_end": {"centers": [8.0, 24.0, 56.0, 120.0], "p": [0.25, 0.25, 0.25, 0.25]},
        "draw": dict.fromkeys(tokens, 0.5),
    }
    if sims > 0:
        base, extra = divmod(sims, len(tokens))
        response["search"] = {
            "sims": sims,
            "nodes": sims + 1,
            "evaluations": sims,
            "ms": 1,
            "root_value": 0.0,
            "plies_left": 40.0,
            "lines": [
                {
                    "move": token,
                    "visits": base + (1 if index < extra else 0),
                    "q": 0.0,
                    "pi": round(share, 6),
                    "pv": [token],
                }
                for index, token in enumerate(tokens)
            ],
        }
    return response


def run_analyse() -> int:
    """Serve analyse requests from stdin until it closes."""
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            request = json.loads(text)
            response = analyse_one(request)
        except (ValueError, KeyError, TypeError) as error:
            request_id = None
            try:
                request_id = json.loads(text).get("id")
            except ValueError:
                pass
            response = {"id": request_id, "error": str(error)}
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the fake bot's command line."""
    parser = argparse.ArgumentParser(description="Fake Intransitive bot")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--candidate", required=True)
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--pairs", type=int, default=1)
    evaluate.add_argument("--sims", type=int, default=32)
    evaluate.add_argument("--threads", type=int, default=1)
    evaluate.add_argument("--seed", type=int, default=1)
    evaluate.add_argument("--opening-plies", type=int, default=8)
    evaluate.add_argument("--records")
    evaluate.add_argument("--stream", action="store_true")
    analyse = sub.add_parser("analyse")
    analyse.add_argument("--engine", required=True)
    analyse.add_argument("--threads", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Dispatch to the requested subcommand."""
    options = parse_args(argv)
    return run_eval(options) if options.command == "eval" else run_analyse()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
