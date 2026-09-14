"""Board constants shared by the kernels, the network and the trainer, and a
reference encoder of the 46 observation planes (DESIGN.md item 1) built on the
`engine` binding, used by the export check and the kernel tests; the GPU path is
`kernels.derive`. `tactics_reference` is the brute-force oracle for the exact
tactics kernels.

Canonical frame: the mover plays from a1 (index 0) toward i9 (index 80);
cells are 0 empty, 1-3 own rock/paper/scissors, 4-6 enemy. Actions are
`direction * 81 + from` with the eight king directions in the order below."""

from __future__ import annotations

import numpy as np
import torch

N_SQUARES = 81
N_DIRS = 8
N_ACTIONS = N_DIRS * N_SQUARES
N_PLANES = 46
N_STATES = 7
CLOCK_SCALE = 200.0

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def mirror_anti(square: int) -> int:
    rank, file = divmod(square, 9)
    return (8 - file) * 9 + (8 - rank)


# Square permutation between consecutive plies (an involution).
ANTI_PERM = [mirror_anti(s) for s in range(N_SQUARES)]
# Direction permutation between consecutive plies: (dr, df) -> (-df, -dr).
ANTI_DIR_PERM = [DIRS.index((-df, -dr)) for (dr, df) in DIRS]


def flip_actions(actions: torch.Tensor) -> torch.Tensor:
    """The same moves seen from the other side's frame, for action indices
    of any shape (an involution)."""
    square_perm = torch.tensor(ANTI_PERM, device=actions.device)
    dir_perm = torch.tensor(ANTI_DIR_PERM, device=actions.device)
    a = actions.long()
    return (dir_perm[a // N_SQUARES] * N_SQUARES + square_perm[a % N_SQUARES]).to(actions.dtype)


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


# First plane of every (rock, paper, scissors) triple; the type cycle rotates each.
TYPE_TRIPLES = (0, 3, 6, 9, 18, 21, 24, 27, 32, 35, 40, 43)


def cycle_plane_perm(k: int) -> list[int]:
    """Source plane of each plane after `k` type cycles: every type triple
    rotates, every other plane stays."""
    perm = list(range(N_PLANES))
    for base in TYPE_TRIPLES:
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


def _neighbours() -> list[list[int]]:
    """The king-step neighbours of every square."""
    out = []
    for square in range(N_SQUARES):
        rank, file = divmod(square, 9)
        out.append([(rank + dr) * 9 + file + df for dr, df in DIRS if 0 <= rank + dr < 9 and 0 <= file + df < 9])
    return out


NEIGHBOURS = _neighbours()
# Path lengths are capped here and scaled by it; anything longer reads as 1.
DIST_CAP = 16
UNREACHABLE = 99


def beats(own_kind, enemy_kind):
    """Type 1-3 (rock, paper, scissors) beats type 1-3; works on arrays."""
    return (own_kind - enemy_kind) % 3 == 1


def _goal_distance(passable: np.ndarray, goal: int) -> np.ndarray:
    """King steps from each square to `goal` through passable squares; the
    square itself need not be passable, the goal must be."""
    dist = np.full(N_SQUARES, UNREACHABLE, dtype=np.int32)
    if not passable[goal]:
        return dist
    dist[goal] = 0
    frontier = [goal]
    while frontier:
        following = []
        for square in frontier:
            for neighbour in NEIGHBOURS[square]:
                if dist[neighbour] > dist[square] + 1:
                    dist[neighbour] = dist[square] + 1
                    if passable[neighbour]:
                        following.append(neighbour)
        frontier = following
    return dist


def _arrival_distance(passable: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """King steps for the nearest source piece to reach each square, entering
    passable squares only."""
    dist = np.full(N_SQUARES, UNREACHABLE, dtype=np.int32)
    dist[sources] = 0
    frontier = list(sources)
    while frontier:
        following = []
        for square in frontier:
            for neighbour in NEIGHBOURS[square]:
                if passable[neighbour] and dist[neighbour] > dist[square] + 1:
                    dist[neighbour] = dist[square] + 1
                    following.append(neighbour)
        frontier = following
    return dist


def reference(board: list[int], since_capture: int, ply: int, clock: int) -> np.ndarray:
    """(46, 81) planes of one position (DESIGN.md item 1); legality comes from
    `engine`, `clock` is the game's capture clock in plies."""
    import engine

    board = list(board)
    planes = np.zeros((N_PLANES, N_SQUARES), dtype=np.float32)
    own_legal = np.array(engine.legal_mask(board), dtype=bool).reshape(N_DIRS, N_SQUARES)
    flipped = flip(board)
    enemy_legal = np.array(engine.legal_mask(flipped), dtype=bool).reshape(N_DIRS, N_SQUARES)
    targets = action_targets().numpy().reshape(N_DIRS, N_SQUARES)
    for square, cell in enumerate(board):
        if cell:
            planes[cell - 1, square] = 1.0
    # Attack maps: the squares each side's legal moves land on, by type, in our frame.
    for d in range(N_DIRS):
        for square in range(N_SQUARES):
            if enemy_legal[d, square]:
                planes[5 + flipped[square], ANTI_PERM[targets[d, square]]] = 1.0
            if own_legal[d, square]:
                planes[8 + board[square], targets[d, square]] = 1.0
    planes[12, 0] = planes[12, 80] = 1.0
    planes[13] = 1.0
    planes[14] = since_capture / CLOCK_SCALE
    planes[15] = max(clock - since_capture, 0) / CLOCK_SCALE
    rank = np.arange(N_SQUARES) // 9
    file = np.arange(N_SQUARES) % 9
    planes[16] = np.maximum(8 - rank, 8 - file) / 8
    planes[17] = np.maximum(rank, file) / 8
    cells = np.array(board)
    for t in (1, 2, 3):
        own_pass = (cells == 0) | ((cells >= 4) & beats(t, cells - 3))
        enemy_pass = (cells == 0) | ((cells >= 1) & (cells <= 3) & beats(t, cells))
        own_goal = _goal_distance(own_pass, 80)
        enemy_goal = _goal_distance(enemy_pass, 0)
        own_here = cells == t
        enemy_here = cells == t + 3
        planes[17 + t] = np.minimum(own_goal, DIST_CAP) / DIST_CAP
        planes[20 + t] = np.minimum(enemy_goal, DIST_CAP) / DIST_CAP
        planes[23 + t] = np.minimum(_arrival_distance(own_pass, np.flatnonzero(own_here)), DIST_CAP) / DIST_CAP
        planes[26 + t] = np.minimum(_arrival_distance(enemy_pass, np.flatnonzero(enemy_here)), DIST_CAP) / DIST_CAP
        planes[31 + t] = own_here.sum() / 4
        planes[34 + t] = enemy_here.sum() / 4
        own_race = own_goal[own_here].min() if own_here.any() else UNREACHABLE
        enemy_race = enemy_goal[enemy_here].min() if enemy_here.any() else UNREACHABLE
        planes[39 + t] = min(own_race, DIST_CAP) / DIST_CAP
        planes[42 + t] = min(enemy_race, DIST_CAP) / DIST_CAP
    own_mobility = own_legal.sum(0)
    enemy_mobility = np.zeros(N_SQUARES)
    for square in range(N_SQUARES):
        enemy_mobility[ANTI_PERM[square]] = enemy_legal[:, square].sum()
    planes[30] = own_mobility / 8
    planes[31] = enemy_mobility / 8
    planes[38] = own_mobility.sum() / 64
    planes[39] = enemy_mobility.sum() / 64
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


def _wins_at_once(board: list[int]) -> bool:
    """The mover has a move that ends the game in its favour at once."""
    import engine

    return any(engine.apply(board, 0, 0, int(a), 0)[3] == 1 for a in np.flatnonzero(engine.legal_mask(board)))


def tactics_reference(
    board: list[int], since_capture: int = 0, clock: int = 0
) -> tuple[list[bool], list[bool], list[bool]]:
    """Brute-force oracle for the tactics kernels: per action, wins at once,
    loses in two (a reply wins at once) and wins in three (every reply leaves
    the mover a win at once), played through `engine` with `since_capture`
    plies since the last capture under a capture `clock` (0 disables it): a
    move on which the clock expires draws, so it is neither a loss in two nor
    a win in three, and a reply on which it expires refutes a win in three."""
    import engine

    board = list(board)
    legal = engine.legal_mask(board)
    win1 = [False] * N_ACTIONS
    loss2 = [False] * N_ACTIONS
    win3 = [False] * N_ACTIONS
    for action in range(N_ACTIONS):
        if not legal[action]:
            continue
        child, child_since, _, outcome = engine.apply(board, since_capture, 0, action, clock)
        if outcome == 1:
            win1[action] = True
            continue
        if outcome != 0:
            continue  # the clock draw ends the game
        child = list(child)
        replies = np.flatnonzero(engine.legal_mask(child))
        forced = len(replies) > 0
        for reply in replies:
            grandchild, _, _, reply_outcome = engine.apply(child, child_since, 0, int(reply), clock)
            if reply_outcome == 1:
                loss2[action] = True
                forced = False
                break
            if reply_outcome != 0:
                forced = False  # a reply that draws defeats the move
            elif forced and not _wins_at_once(list(grandchild)):
                forced = False
        win3[action] = forced
    return win1, loss2, win3
