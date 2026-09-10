"""The site's compact export: encoding a random engine game and decoding it
back, and the human importer on such a file."""

import base64
import json

import engine
import numpy as np

from nnue import data, features, games, paths
from nnue.importer import import_human


def encode_export(moves: list[str]) -> str:
    packed = 0
    for token in moves:
        start, end = token.split("-")
        origin = (int(start[1]) - 1) * 9 + games.FILES.index(start[0])
        delta = (games.FILES.index(end[0]) - games.FILES.index(start[0]), int(end[1]) - int(start[1]))
        packed = (packed << 10) | (origin << 3) | games.DIRS.index(delta)
    bits = 10 * len(moves)
    # The site packs from the top bit; pad the tail, not the head.
    shift = (8 - bits % 8) % 8
    raw = (packed << shift).to_bytes((bits + shift) // 8, "big") if moves else b""
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def random_game(rng, plies):
    board, since, ply, outcome = list(engine.initial_board()), 0, 0, 0
    moves = []
    for _ in range(plies):
        legal = np.flatnonzero(engine.legal_mask(board))
        action = int(rng.choice(legal))
        start = action % 81
        d_rank, d_file = games.DIRS[action // 81]
        end = (start // 9 + d_rank) * 9 + start % 9 + d_file
        if ply % 2 == 1:
            start, end = int(features.ANTI[start]), int(features.ANTI[end])
        moves.append(f"{games.FILES[start % 9]}{start // 9 + 1}-{games.FILES[end % 9]}{end // 9 + 1}")
        child, since, ply, outcome = engine.apply(board, since, ply, action, games.SITE_CLOCK)
        board = list(child)
        if outcome:
            break
    return moves, outcome


def test_export_round_trip():
    rng = np.random.default_rng(4)
    for _ in range(10):
        moves, _ = random_game(rng, int(rng.integers(1, 40)))
        assert games.decode_export(encode_export(moves)) == moves


def test_human_importer_labels_roots_with_the_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    rng = np.random.default_rng(9)
    lines, expected = [], 0
    for i, result in enumerate("brdub"):
        moves, _ = random_game(rng, 30)
        lines.append(f"{result} p{i} q{i} {encode_export(moves)}")
        expected += len(moves) if result != "u" else 0
    lines.append("u someone -")
    export = tmp_path / "games_export.txt"
    export.write_text("\n".join(lines) + "\n", encoding="ascii")
    target = import_human(export, "human", seed=1)
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["counts"] == {"games": 6, "unfinished": 1, "without_moves": 1, "finished": 4}
    assert provenance["positions"] == expected
    kind, source = np.load(target / "kind.npy"), np.load(target / "source.npy")
    assert (kind == data.KIND_RETURN).all() and (source == data.SOURCE_HUMAN).all()
    assert np.load(target / "outcome_ok.npy").all()
    board, target_value, ply = (np.load(target / f"{n}.npy") for n in ("board", "target", "ply"))
    initial = (board == np.frombuffer(engine.initial_board(), dtype=np.uint8)).all(1)
    # The four finished games share the initial position: b, r, d, b from Blue's view average to 0.25.
    assert initial.sum() == 1 and abs(float(target_value[initial][0]) - 0.25) < 1e-6
    assert (ply[initial] == 0).all()
