"""Board constants shared by the kernels, the network and the trainer, and a
reference encoder of the 25 observation planes built on the `engine` binding
(used by the export check and the kernel tests; the GPU path is `kernels.derive`).

Canonical frame: the mover plays from a1 (index 0) toward i9 (index 80);
cells are 0 empty, 1-3 own rock/paper/scissors, 4-6 enemy. Actions are
`direction * 81 + from` with the eight king directions in the order below."""

from __future__ import annotations

import numpy as np
import torch

N_SQUARES = 81
N_DIRS = 8
N_ACTIONS = N_DIRS * N_SQUARES
N_PLANES = 25
N_STATES = 7
CLOCK_SCALE = 200.0

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def mirror_anti(square: int) -> int:
    rank, file = divmod(square, 9)
    return (8 - file) * 9 + (8 - rank)


# Square permutation between consecutive plies (an involution).
ANTI_PERM = [mirror_anti(s) for s in range(N_SQUARES)]
# Reflection across the a1-i9 diagonal (rank <-> file): squares and directions.
DIAG_SQUARE_PERM = [(s % 9) * 9 + s // 9 for s in range(N_SQUARES)]
DIAG_DIR_PERM = [DIRS.index((df, dr)) for (dr, df) in DIRS]


def cycle_cell(cell: int, k: int) -> int:
    """Cell after `k` rock -> paper -> scissors -> rock relabellings (0 stays empty)."""
    if cell == 0:
        return 0
    side, kind = divmod(cell - 1, 3)
    return side * 3 + (kind + k) % 3 + 1


CYCLE_CELL = [[cycle_cell(v, k) for v in range(7)] for k in range(3)]


def cycle_plane_perm(k: int) -> list[int]:
    """Source plane of each plane after `k` type cycles: the own, enemy and
    threat triples rotate, every other plane stays."""
    perm = list(range(N_PLANES))
    for base in (0, 3, 6):
        for t in range(3):
            perm[base + (t + k) % 3] = base + t
    return perm


CYCLE_PLANE_PERM = [cycle_plane_perm(k) for k in range(3)]


def action_targets() -> torch.Tensor:
    """(648,) target square of each action, or -1 off the board."""
    out = torch.full((N_ACTIONS,), -1, dtype=torch.long)
    for d, (dr, df) in enumerate(DIRS):
        for s in range(N_SQUARES):
            r, f = divmod(s, 9)
            if 0 <= r + dr < 9 and 0 <= f + df < 9:
                out[d * N_SQUARES + s] = (r + dr) * 9 + f + df
    return out


def initial_board() -> list[int]:
    import engine

    return list(engine.initial_board())


def flip(board: list[int]) -> list[int]:
    """The same position seen by the other side."""
    swapped = [0 if v == 0 else v + 3 if v <= 3 else v - 3 for v in board]
    return [swapped[ANTI_PERM[s]] for s in range(N_SQUARES)]


def reference(board: list[int], since_capture: int, ply: int) -> np.ndarray:
    """(25, 81) planes of one position, every rule taken from `engine`."""
    import engine

    board = list(board)
    planes = np.zeros((N_PLANES, N_SQUARES), dtype=np.float32)
    own_legal = np.array(engine.legal_mask(board), dtype=bool).reshape(N_DIRS, N_SQUARES)
    enemy_legal = np.array(engine.legal_mask(flip(board)), dtype=bool).reshape(N_DIRS, N_SQUARES)
    targets = action_targets().numpy().reshape(N_DIRS, N_SQUARES)
    for s, v in enumerate(board):
        if v:
            planes[v - 1, s] = 1.0
    # Threat planes: the enemy's legal moves, mapped back into our frame.
    flipped = flip(board)
    for d in range(N_DIRS):
        for s in range(N_SQUARES):
            if enemy_legal[d, s]:
                kind = flipped[s]  # 1..3 in the enemy's frame
                planes[5 + kind, ANTI_PERM[targets[d, s]]] = 1.0
    planes[9, 0] = planes[9, 80] = 1.0
    planes[10] = since_capture / CLOCK_SCALE
    planes[11] = ply / CLOCK_SCALE
    planes[12] = 1.0
    rank = np.arange(N_SQUARES) // 9
    file = np.arange(N_SQUARES) % 9
    goal_distance = np.maximum(8 - rank, 8 - file)
    home_distance = np.maximum(rank, file)
    cells = np.array(board)
    own = (cells >= 1) & (cells <= 3)
    enemy = cells >= 4
    planes[13] = own * goal_distance / 8
    planes[14] = enemy * home_distance / 8
    planes[15] = (goal_distance[own].min() if own.any() else 9) / 8
    planes[16] = (home_distance[enemy].min() if enemy.any() else 9) / 8
    planes[17] = goal_distance / 8
    planes[18] = home_distance / 8
    own_mobility = own_legal.sum(0)
    enemy_mobility = np.zeros(N_SQUARES)
    for s in range(N_SQUARES):
        enemy_mobility[ANTI_PERM[s]] = enemy_legal[:, s].sum()
    planes[19] = own_mobility / 8
    planes[20] = enemy_mobility / 8
    planes[21] = own_mobility.sum() / 64
    planes[22] = enemy_mobility.sum() / 64
    planes[23] = own.sum() / 10
    planes[24] = enemy.sum() / 10
    return planes


def random_positions(count: int, seed: int, max_plies: int = 120) -> list[tuple[list[int], int, int]]:
    """Reachable `(board, since_capture, ply)` triples from random play through `engine`."""
    import engine

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        board, since_capture, ply = list(engine.initial_board()), 0, 0
        for _ in range(int(rng.integers(0, max_plies))):
            legal = np.flatnonzero(engine.legal_mask(board))
            if len(legal) == 0:
                break
            action = int(rng.choice(legal))
            board, since_capture, ply, outcome = engine.apply(board, since_capture, ply, action, 0)
            board = list(board)
            if outcome:
                break
        out.append((board, since_capture, ply))
    return out
