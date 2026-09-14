"""Triton kernels for the batched GPU environment: the rules on many boards at
once. This is the only permitted second implementation of the rules; every
kernel is tested against the `engine` binding in `tests/test_kernels.py`.

Boards are int8 cells (0 empty, 1-3 own rock/paper/scissors, 4-6 enemy) in the
mover's frame, one program per board."""

from __future__ import annotations

import importlib.util
import os
import sys

if sys.platform == "win32" and "CC" not in os.environ:
    # Triton compiles its launcher stubs with a C compiler; use the one it bundles.
    triton_dir = os.path.dirname(importlib.util.find_spec("triton").origin)
    os.environ["CC"] = os.path.join(triton_dir, "runtime", "tcc", "tcc.exe")

import torch
import triton
import triton.language as tl

from .planes import CLOCK_SCALE, DIST_CAP, N_ACTIONS, N_PLANES, N_SQUARES

BLOCK = 128
# The twelve distance fields (goal-path then arrival, own then enemy types) are
# relaxed together in one vector of 12 * 81 lanes; FAR marks unreachable.
N_FIELDS = 12
FIELD_BLOCK = 1024
DERIVE_SCRATCH = 1024
BFS_ROUNDS = 16
FAR = 100
# One move takes at most 16 enemy moves away: the captured piece's own moves and
# the moves onto the square the mover comes to stand on. Two moves, the reply and
# the winning move, take at most twice that.
MOBILITY_BOUND = 16
TWO_MOVE_BOUND = 2 * MOBILITY_BOUND
# Per-board int32 scratch of the tactics kernels: the child board, the grandchild
# board (wins in three only) and `_any_win`'s per-square mobility.
LOSS_SCRATCH = 2 * N_SQUARES
WIN3_SCRATCH = 3 * N_SQUARES


@triton.jit
def _delta(d):
    # Direction index 0..7 -> (rank delta, file delta), skipping the (0, 0) entry.
    k = d + (d + 4) // 8
    return k // 3 - 1, k % 3 - 1


@triton.jit
def _beats(a, b):
    # Own type a (1..3) beats enemy type b (1..3).
    return ((a - b + 3) % 3) == 1


@triton.jit
def _store_plane(ob, plane: tl.constexpr, idx, m, value):
    # One observation plane of one board, as bf16.
    tl.store(ob + plane * 81 + idx, value.to(tl.float32).to(tl.bfloat16), mask=m)


@triton.jit
def apply_kernel(
    board_ptr, action_ptr, since_capture_ptr, ply_ptr, goal_ptr, eliminated_ptr, captured_ptr, BLOCK: tl.constexpr
):
    """Play one action per board, writing the child already flipped into the
    next mover's frame, and flag goal entry, elimination and capture."""
    pid = tl.program_id(0)
    base = pid * 81
    idx = tl.arange(0, BLOCK)
    m = idx < 81
    a = tl.load(action_ptr + pid).to(tl.int32)
    d = a // 81
    s = a % 81
    dr, dc = _delta(d)
    to = (s // 9 + dr) * 9 + (s % 9 + dc)
    mover = tl.load(board_ptr + base + s).to(tl.int32)
    target = tl.load(board_ptr + base + to).to(tl.int32)
    # child[idx] = swap(moved[mirror(idx)])
    r = idx // 9
    c = idx % 9
    mirrored = (8 - c) * 9 + (8 - r)
    cell = tl.load(board_ptr + base + mirrored, mask=m, other=0).to(tl.int32)
    cell = tl.where(mirrored == to, mover, cell)
    cell = tl.where(mirrored == s, 0, cell)
    enemies_left = tl.sum(((cell >= 4) & m).to(tl.int32), axis=0)
    swapped = tl.where(cell == 0, 0, tl.where(cell <= 3, cell + 3, cell - 3))
    tl.debug_barrier()
    tl.store(board_ptr + base + idx, swapped.to(tl.int8), mask=m)
    capture = target != 0
    since_capture = tl.load(since_capture_ptr + pid)
    tl.store(since_capture_ptr + pid, tl.where(capture, 0, since_capture + 1))
    tl.store(ply_ptr + pid, tl.load(ply_ptr + pid) + 1)
    tl.store(goal_ptr + pid, (to == 80).to(tl.int8))
    tl.store(eliminated_ptr + pid, (enemies_left == 0).to(tl.int8))
    tl.store(captured_ptr + pid, capture.to(tl.int8))


@triton.jit
def derive_kernel(
    board_ptr,
    since_capture_ptr,
    ply_ptr,
    clock_ptr,
    legal_ptr,
    any_legal_ptr,
    planes_ptr,
    scratch_ptr,
    CLOCK: tl.constexpr,
    CAP: tl.constexpr,
    NP: tl.constexpr,
    ROUNDS: tl.constexpr,
    BLOCK: tl.constexpr,
    FIELD: tl.constexpr,
    FIELDS: tl.constexpr,
    UNREACHED: tl.constexpr,
    SCRATCH: tl.constexpr,
):
    """Legal-move mask, the number of legal moves and the 46 observation planes
    (DESIGN.md item 1) of every board. `scratch_ptr` is per board `SCRATCH`
    int32 holding the `FIELDS` distance fields while they are relaxed;
    `UNREACHED` marks a square no search reached."""
    pid = tl.program_id(0)
    base = pid * 81
    idx = tl.arange(0, BLOCK)
    m = idx < 81
    b = tl.load(board_ptr + base + idx, mask=m, other=0).to(tl.int32)
    r = idx // 9
    c = idx % 9
    own = (b >= 1) & (b <= 3)
    enemy = b >= 4
    empty = b == 0
    kind = tl.where(enemy, b - 3, b)
    own_mobility = tl.zeros([BLOCK], dtype=tl.int32)
    enemy_mobility = tl.zeros([BLOCK], dtype=tl.int32)
    near_own_r = tl.zeros([BLOCK], dtype=tl.int32)
    near_own_p = tl.zeros([BLOCK], dtype=tl.int32)
    near_own_s = tl.zeros([BLOCK], dtype=tl.int32)
    near_enemy_r = tl.zeros([BLOCK], dtype=tl.int32)
    near_enemy_p = tl.zeros([BLOCK], dtype=tl.int32)
    near_enemy_s = tl.zeros([BLOCK], dtype=tl.int32)
    for k in tl.static_range(9):
        if k != 4:
            d = k - k // 5
            tr = r + (k // 3 - 1)
            tc = c + (k % 3 - 1)
            inb = (tr >= 0) & (tr < 9) & (tc >= 0) & (tc < 9) & m
            tv = tl.load(board_ptr + base + tr * 9 + tc, mask=inb, other=-1).to(tl.int32)
            legal = own & inb & ((tv == 0) | ((tv >= 4) & _beats(kind, tv - 3)))
            tl.store(legal_ptr + pid * 648 + d * 81 + idx, legal.to(tl.int8), mask=m)
            own_mobility += legal.to(tl.int32)
            enemy_mobility += (enemy & inb & ((tv == 0) | ((tv >= 1) & (tv <= 3) & _beats(kind, tv)))).to(tl.int32)
            near_own_r += (tv == 1).to(tl.int32)
            near_own_p += (tv == 2).to(tl.int32)
            near_own_s += (tv == 3).to(tl.int32)
            near_enemy_r += (tv == 4).to(tl.int32)
            near_enemy_p += (tv == 5).to(tl.int32)
            near_enemy_s += (tv == 6).to(tl.int32)
    own_total = tl.sum(own_mobility, axis=0)
    enemy_total = tl.sum(enemy_mobility, axis=0)
    tl.store(any_legal_ptr + pid, own_total)

    ob = planes_ptr + pid * (NP * 81)
    ones = (idx * 0 + 1).to(tl.float32)
    for v in tl.static_range(1, 7):
        _store_plane(ob, v - 1, idx, m, b == v)
    # Attack maps: a neighbour of that type stands ready to move onto the square.
    for t in tl.static_range(3):
        for_enemy = empty | (own & _beats(t + 1, kind))
        for_own = empty | (enemy & _beats(t + 1, kind))
        near_enemy = tl.where(t == 0, near_enemy_r, tl.where(t == 1, near_enemy_p, near_enemy_s))
        near_own = tl.where(t == 0, near_own_r, tl.where(t == 1, near_own_p, near_own_s))
        _store_plane(ob, 6 + t, idx, m, for_enemy & (near_enemy > 0))
        _store_plane(ob, 9 + t, idx, m, for_own & (near_own > 0))
    _store_plane(ob, 12, idx, m, (idx == 0) | (idx == 80))
    _store_plane(ob, 13, idx, m, ones)
    since_capture = tl.load(since_capture_ptr + pid).to(tl.int32)
    clock = tl.load(clock_ptr + pid).to(tl.int32)
    _store_plane(ob, 14, idx, m, ones * since_capture.to(tl.float32) / CLOCK)
    _store_plane(ob, 15, idx, m, ones * tl.maximum(clock - since_capture, 0).to(tl.float32) / CLOCK)
    _store_plane(ob, 16, idx, m, tl.maximum(8 - r, 8 - c).to(tl.float32) / 8)
    _store_plane(ob, 17, idx, m, tl.maximum(r, c).to(tl.float32) / 8)
    _store_plane(ob, 30, idx, m, own_mobility.to(tl.float32) / 8)
    _store_plane(ob, 31, idx, m, enemy_mobility.to(tl.float32) / 8)
    for t in tl.static_range(3):
        _store_plane(ob, 32 + t, idx, m, ones * tl.sum((b == t + 1).to(tl.int32), axis=0).to(tl.float32) / 4)
        _store_plane(ob, 35 + t, idx, m, ones * tl.sum((b == t + 4).to(tl.int32), axis=0).to(tl.float32) / 4)
    _store_plane(ob, 38, idx, m, ones * own_total.to(tl.float32) / 64)
    _store_plane(ob, 39, idx, m, ones * enemy_total.to(tl.float32) / 64)

    # Twelve distance fields at once: field f owns lanes f * 81 .. f * 81 + 80.
    # Fields 0-5 are goal-path (own types toward i9, enemy types toward a1),
    # 6-11 the arrival maps from the pieces of that type.
    fidx = tl.arange(0, FIELD)
    fm = fidx < FIELDS * 81
    f = fidx // 81
    sq = fidx % 81
    fr = sq // 9
    fc = sq % 9
    cell = tl.load(board_ptr + base + sq, mask=fm, other=0).to(tl.int32)
    t = f % 3 + 1
    own_field = (f < 3) | ((f >= 6) & (f < 9))
    goal_field = f < 6
    cell_kind = tl.where(cell >= 4, cell - 3, cell)
    prey = tl.where(own_field, cell >= 4, (cell >= 1) & (cell <= 3))
    passable = (cell == 0) | (prey & _beats(t, cell_kind))
    seeded = tl.where(
        goal_field,
        (sq == tl.where(own_field, 80, 0)) & passable,
        cell == tl.where(own_field, t, t + 3),
    )
    dist = tl.where(seeded & fm, 0, UNREACHED)
    fbase = pid * SCRATCH
    for _ in range(ROUNDS):
        # A goal-path square passes its distance on only when it is passable; an
        # arrival square takes one only when it is.
        tl.store(scratch_ptr + fbase + fidx, tl.where(goal_field & ~passable, UNREACHED, dist), mask=fm)
        tl.debug_barrier()
        best = tl.full([FIELD], UNREACHED, tl.int32)
        for k in tl.static_range(9):
            if k != 4:
                nr = fr + (k // 3 - 1)
                nc = fc + (k % 3 - 1)
                near = (nr >= 0) & (nr < 9) & (nc >= 0) & (nc < 9) & fm
                best = tl.minimum(best, tl.load(scratch_ptr + fbase + f * 81 + nr * 9 + nc, mask=near, other=UNREACHED))
        tl.debug_barrier()
        dist = tl.minimum(dist, tl.where(goal_field | passable, best + 1, UNREACHED))
    tl.store(scratch_ptr + fbase + fidx, dist, mask=fm)
    tl.debug_barrier()

    for k in tl.static_range(6):
        # Goal-path plane, and the race distance of the nearest piece of that type.
        d = tl.load(scratch_ptr + fbase + k * 81 + idx, mask=m, other=UNREACHED)
        race = tl.min(tl.where(b == k % 3 + 1 + 3 * (k // 3), d, UNREACHED), axis=0)
        _store_plane(ob, 18 + k, idx, m, tl.minimum(d, CAP).to(tl.float32) / CAP)
        _store_plane(ob, 40 + k, idx, m, ones * tl.minimum(race, CAP).to(tl.float32) / CAP)
    for k in tl.static_range(6):
        d = tl.load(scratch_ptr + fbase + (6 + k) * 81 + idx, mask=m, other=UNREACHED)
        _store_plane(ob, 24 + k, idx, m, tl.minimum(d, CAP).to(tl.float32) / CAP)


@triton.jit
def _remaining_enemy_moves(b_ptr, base, mob_ptr, total, r, c, kind, m, d, BLOCK: tl.constexpr):
    """Enemy moves left after the mover's piece on each square steps in direction
    `d`: the total before, minus the captured piece's own moves, plus what the
    vacated source and the occupied target change for their enemy neighbours.
    Returns that count with the target square, its cell and whether it is on the
    board. `mob_ptr` holds this board's 81 per-square enemy mobilities."""
    dr, dc = _delta(d)
    tr = r + dr
    tc = c + dc
    inb = (tr >= 0) & (tr < 9) & (tc >= 0) & (tc < 9) & m
    t = tr * 9 + tc
    captured = tl.load(b_ptr + base + t, mask=inb, other=0).to(tl.int32)
    removed = tl.where(captured >= 4, tl.load(mob_ptr + t, mask=inb, other=0), 0)
    gained_at_source = tl.zeros([BLOCK], dtype=tl.int32)
    changed_at_target = tl.zeros([BLOCK], dtype=tl.int32)
    for k in tl.static_range(9):
        if k != 4:
            nr = k // 3 - 1
            nc = k % 3 - 1
            # Enemy neighbours of the vacated source square: any of them may now
            # move there, only those beating the mover could before.
            sr = r + nr
            sc = c + nc
            s_inb = (sr >= 0) & (sr < 9) & (sc >= 0) & (sc < 9) & m
            sv = tl.load(b_ptr + base + sr * 9 + sc, mask=s_inb, other=0).to(tl.int32)
            s_enemy = (sv >= 4) & ((sr != tr) | (sc != tc))
            gained_at_source += (s_enemy & ~_beats(sv - 3, kind)).to(tl.int32)
            # Enemy neighbours of the target square: they could move there if it
            # was empty or held an own piece they beat; afterwards only if they
            # beat the mover.
            qr = tr + nr
            qc = tc + nc
            q_inb = (qr >= 0) & (qr < 9) & (qc >= 0) & (qc < 9) & inb
            qv = tl.load(b_ptr + base + qr * 9 + qc, mask=q_inb, other=0).to(tl.int32)
            q_enemy = qv >= 4
            before = (captured == 0) | ((captured >= 1) & (captured <= 3) & _beats(qv - 3, captured))
            after = _beats(qv - 3, kind)
            changed_at_target += tl.where(q_enemy, after.to(tl.int32) - before.to(tl.int32), 0)
    return total - removed + gained_at_source + changed_at_target, t, captured, inb


@triton.jit
def exact_kernel(board_ptr, legal_ptr, mobility_ptr, exact_ptr, BLOCK: tl.constexpr):
    """Flag every legal action that wins immediately: goal entry, capturing the
    last enemy piece, or leaving the enemy without a legal move.
    `mobility_ptr` is per-board scratch of 81 int32."""
    pid = tl.program_id(0)
    base = pid * 81
    idx = tl.arange(0, BLOCK)
    m = idx < 81
    b = tl.load(board_ptr + base + idx, mask=m, other=0).to(tl.int32)
    r = idx // 9
    c = idx % 9
    enemy = b >= 4
    kind = tl.where(enemy, b - 3, b)
    mobility = tl.zeros([BLOCK], dtype=tl.int32)
    for k in tl.static_range(9):
        if k != 4:
            tr = r + (k // 3 - 1)
            tc = c + (k % 3 - 1)
            inb = (tr >= 0) & (tr < 9) & (tc >= 0) & (tc < 9) & m
            tv = tl.load(board_ptr + base + tr * 9 + tc, mask=inb, other=-1).to(tl.int32)
            mobility += (enemy & inb & ((tv == 0) | ((tv >= 1) & (tv <= 3) & _beats(kind, tv)))).to(tl.int32)
    tl.store(mobility_ptr + base + idx, mobility, mask=m)
    tl.debug_barrier()
    total = tl.sum(mobility, axis=0)
    n_enemy = tl.sum(enemy.to(tl.int32), axis=0)
    for d in tl.static_range(8):
        remaining, t, captured, _ = _remaining_enemy_moves(
            board_ptr, base, mobility_ptr + base, total, r, c, kind, m, d, BLOCK
        )
        legal = tl.load(legal_ptr + pid * 648 + d * 81 + idx, mask=m, other=0) != 0
        eliminated = (n_enemy - (captured >= 4).to(tl.int32)) == 0
        wins = legal & ((t == 80) | eliminated | (remaining == 0))
        tl.store(exact_ptr + pid * 648 + d * 81 + idx, wins.to(tl.int8), mask=m)


@triton.jit
def _any_win(b_ptr, base, mob_ptr, BOUND: tl.constexpr, BLOCK: tl.constexpr):
    """1 when the mover of the board at `b_ptr + base` has a move that wins at
    once. Goal entry and capturing the last enemy piece are read off the board;
    the stalemate scan runs only when the enemy has at most `BOUND`
    moves, since one move cannot take away more. `mob_ptr` is 81 int32 of
    scratch for the per-square enemy mobility."""
    idx = tl.arange(0, BLOCK)
    m = idx < 81
    b = tl.load(b_ptr + base + idx, mask=m, other=0).to(tl.int32)
    r = idx // 9
    c = idx % 9
    own = (b >= 1) & (b <= 3)
    enemy = b >= 4
    kind = tl.where(enemy, b - 3, b)
    goal = tl.load(b_ptr + base + 80).to(tl.int32)
    win = tl.full((), 0, tl.int32)
    for k in tl.static_range(3):
        # The three neighbours of i9: an own piece there may step onto the goal.
        v = tl.load(b_ptr + base + (70, 71, 79)[k]).to(tl.int32)
        enters = (v >= 1) & (v <= 3) & ((goal == 0) | ((goal >= 4) & _beats(v, goal - 3)))
        win += enters.to(tl.int32)
    if win == 0:
        mobility = tl.zeros([BLOCK], dtype=tl.int32)
        capturable = tl.zeros([BLOCK], dtype=tl.int32)
        for k in tl.static_range(9):
            if k != 4:
                tr = r + (k // 3 - 1)
                tc = c + (k % 3 - 1)
                inb = (tr >= 0) & (tr < 9) & (tc >= 0) & (tc < 9) & m
                tv = tl.load(b_ptr + base + tr * 9 + tc, mask=inb, other=-1).to(tl.int32)
                mobility += (enemy & inb & ((tv == 0) | ((tv >= 1) & (tv <= 3) & _beats(kind, tv)))).to(tl.int32)
                capturable += (enemy & inb & (tv >= 1) & (tv <= 3) & _beats(tv, kind)).to(tl.int32)
        n_enemy = tl.sum(enemy.to(tl.int32), axis=0)
        total = tl.sum(mobility, axis=0)
        win = ((n_enemy == 1) & (tl.sum(capturable, axis=0) > 0)).to(tl.int32)
        if win == 0:
            if total <= BOUND:
                tl.store(mob_ptr + idx, mobility, mask=m)
                tl.debug_barrier()
                found = tl.zeros([BLOCK], dtype=tl.int32)
                for d in tl.static_range(8):
                    remaining, _, captured, inb = _remaining_enemy_moves(
                        b_ptr, base, mob_ptr, total, r, c, kind, m, d, BLOCK
                    )
                    legal = own & inb & ((captured == 0) | ((captured >= 4) & _beats(kind, captured - 3)))
                    found += (legal & (remaining == 0)).to(tl.int32)
                win = (tl.sum(found, axis=0) > 0).to(tl.int32)
    return win


@triton.jit
def loses_in_two_kernel(
    board_ptr,
    legal_ptr,
    since_ptr,
    clock_ptr,
    out_ptr,
    scratch_ptr,
    BOUND: tl.constexpr,
    SCRATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """1 for every legal move after which the opponent has a win at once. A move
    that ends the game stays 0: onto the goal, or a non-capture on which the
    capture clock expires (`since_ptr` and `clock_ptr` hold per board the plies
    since the last capture and the clock; a clock of 0 never expires, as in the
    engine's rules). `scratch_ptr` is per board 162 int32: the child board and
    `_any_win`'s mobility."""
    pid = tl.program_id(0)
    base = pid * 81
    child = scratch_ptr + pid * SCRATCH
    since = tl.load(since_ptr + pid).to(tl.int32)
    clock = tl.load(clock_ptr + pid).to(tl.int32)
    idx = tl.arange(0, BLOCK)
    m = idx < 81
    r = idx // 9
    c = idx % 9
    mirrored = (8 - c) * 9 + (8 - r)
    for s in range(81):
        mover = tl.load(board_ptr + base + s).to(tl.int32)
        if (mover >= 1) & (mover <= 3):
            for d in range(8):
                legal = tl.load(legal_ptr + pid * 648 + d * 81 + s)
                if legal != 0:
                    dr, dc = _delta(d)
                    to = (s // 9 + dr) * 9 + (s % 9 + dc)
                    target = tl.load(board_ptr + base + to).to(tl.int32)
                    # A non-capture on which the clock expires ends the game in a draw.
                    ongoing = (target != 0) | (clock <= 0) | (since + 1 < clock)
                    if (to != 80) & ongoing:
                        cell = tl.load(board_ptr + base + mirrored, mask=m, other=0).to(tl.int32)
                        cell = tl.where(mirrored == to, mover, cell)
                        cell = tl.where(mirrored == s, 0, cell)
                        swapped = tl.where(cell == 0, 0, tl.where(cell <= 3, cell + 3, cell - 3))
                        tl.store(child + idx, swapped, mask=m)
                        tl.debug_barrier()
                        loses = _any_win(child, 0, child + 81, BOUND, BLOCK)
                        tl.store(out_ptr + pid * 648 + d * 81 + s, loses.to(tl.int8))


@triton.jit
def wins_in_three_kernel(
    board_ptr,
    legal_ptr,
    since_ptr,
    clock_ptr,
    out_ptr,
    scratch_ptr,
    BOUND: tl.constexpr,
    TWO_BOUND: tl.constexpr,
    SCRATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """1 for every legal move that wins in three plies: no reply wins at once for
    the opponent, a reply exists, and every reply leaves the mover a win at once.
    The capture clock (`since_ptr`, `clock_ptr` per board; 0 never expires) is
    kept as the engine keeps it: a non-capture move on which it expires draws
    and is no win, and a non-capture reply on which it expires draws and
    refutes the move. `scratch_ptr` is per board 243 int32: the child board,
    the grandchild board and `_any_win`'s mobility."""
    pid = tl.program_id(0)
    base = pid * 81
    child = scratch_ptr + pid * SCRATCH
    grand = child + 81
    mob = child + 162
    since = tl.load(since_ptr + pid).to(tl.int32)
    clock = tl.load(clock_ptr + pid).to(tl.int32)
    idx = tl.arange(0, BLOCK)
    m = idx < 81
    r = idx // 9
    c = idx % 9
    mirrored = (8 - c) * 9 + (8 - r)
    for s in range(81):
        mover = tl.load(board_ptr + base + s).to(tl.int32)
        if (mover >= 1) & (mover <= 3):
            for d in range(8):
                legal = tl.load(legal_ptr + pid * 648 + d * 81 + s)
                if legal != 0:
                    dr, dc = _delta(d)
                    to = (s // 9 + dr) * 9 + (s % 9 + dc)
                    target = tl.load(board_ptr + base + to).to(tl.int32)
                    # A non-capture on which the clock expires ends the game in a draw;
                    # otherwise the replies face the clock reset by a capture or one ply on.
                    ongoing = (target != 0) | (clock <= 0) | (since + 1 < clock)
                    child_since = (since + 1) * (target == 0).to(tl.int32)
                    if (to != 80) & ongoing:
                        cell = tl.load(board_ptr + base + mirrored, mask=m, other=0).to(tl.int32)
                        cell = tl.where(mirrored == to, mover, cell)
                        cell = tl.where(mirrored == s, 0, cell)
                        swapped = tl.where(cell == 0, 0, tl.where(cell <= 3, cell + 3, cell - 3))
                        tl.store(child + idx, swapped, mask=m)
                        tl.debug_barrier()
                        possible = _any_win(child, 0, mob, BOUND, BLOCK) == 0
                        if possible:
                            # A win at once two plies on needs one of: a mover's
                            # piece already beside the goal (squares 1, 9 and 10
                            # of the child's frame), the opponent down to one
                            # piece, or few enough opponent moves for two moves
                            # to take them all. Nothing else can win in three,
                            # so the replies need not be searched.
                            cb = tl.load(child + idx, mask=m, other=0).to(tl.int32)
                            beside = (tl.load(child + 1) >= 4) | (tl.load(child + 9) >= 4) | (tl.load(child + 10) >= 4)
                            ckind = tl.where(cb >= 4, cb - 3, cb)
                            cown = (cb >= 1) & (cb <= 3)
                            cmob = tl.zeros([BLOCK], dtype=tl.int32)
                            for k in tl.static_range(9):
                                if k != 4:
                                    tr = r + (k // 3 - 1)
                                    tc = c + (k % 3 - 1)
                                    ib = (tr >= 0) & (tr < 9) & (tc >= 0) & (tc < 9) & m
                                    tv = tl.load(child + tr * 9 + tc, mask=ib, other=-1).to(tl.int32)
                                    cmob += (cown & ib & ((tv == 0) | ((tv >= 4) & _beats(ckind, tv - 3)))).to(tl.int32)
                            possible = (
                                beside | (tl.sum(cown.to(tl.int32), axis=0) == 1) | (tl.sum(cmob, axis=0) <= TWO_BOUND)
                            )
                        if possible:
                            forced = tl.full((), 1, tl.int32)
                            replies = tl.full((), 0, tl.int32)
                            for s2 in range(81):
                                if forced != 0:
                                    piece = tl.load(child + s2).to(tl.int32)
                                    if (piece >= 1) & (piece <= 3):
                                        for d2 in range(8):
                                            if forced != 0:
                                                dr2, dc2 = _delta(d2)
                                                r2 = s2 // 9 + dr2
                                                c2 = s2 % 9 + dc2
                                                if (r2 >= 0) & (r2 < 9) & (c2 >= 0) & (c2 < 9):
                                                    to2 = r2 * 9 + c2
                                                    tgt = tl.load(child + to2).to(tl.int32)
                                                    if (tgt == 0) | ((tgt >= 4) & _beats(piece, tgt - 3)):
                                                        replies += 1
                                                        if (tgt == 0) & (clock > 0) & (child_since + 1 >= clock):
                                                            # The reply draws on the clock: no win in three.
                                                            forced = tl.full((), 0, tl.int32)
                                                        else:
                                                            g = tl.load(child + mirrored, mask=m, other=0).to(tl.int32)
                                                            g = tl.where(mirrored == to2, piece, g)
                                                            g = tl.where(mirrored == s2, 0, g)
                                                            g = tl.where(g == 0, 0, tl.where(g <= 3, g + 3, g - 3))
                                                            tl.store(grand + idx, g, mask=m)
                                                            tl.debug_barrier()
                                                            forced = _any_win(grand, 0, mob, BOUND, BLOCK)
                            won = (forced != 0) & (replies > 0)
                            tl.store(out_ptr + pid * 648 + d * 81 + s, won.to(tl.int8))


def apply(board, action, since_capture, ply, goal, eliminated, captured) -> None:
    """In place: `board` becomes the child, counters advance, flags are written."""
    apply_kernel[(board.shape[0],)](board, action, since_capture, ply, goal, eliminated, captured, BLOCK=BLOCK)


def derive(board, since_capture, ply, clock, legal, any_legal, planes, scratch) -> None:
    """`clock` is the per-board capture clock in plies; `scratch` is (n, 1024) int32."""
    n = board.shape[0]
    assert legal.shape == (n, N_ACTIONS) and planes.shape == (n, N_PLANES, N_SQUARES)
    assert scratch.shape == (n, DERIVE_SCRATCH)
    derive_kernel[(n,)](
        board,
        since_capture,
        ply,
        clock,
        legal,
        any_legal,
        planes,
        scratch,
        CLOCK=CLOCK_SCALE,
        CAP=DIST_CAP,
        NP=N_PLANES,
        ROUNDS=BFS_ROUNDS,
        BLOCK=BLOCK,
        FIELD=FIELD_BLOCK,
        FIELDS=N_FIELDS,
        UNREACHED=FAR,
        SCRATCH=DERIVE_SCRATCH,
    )


def exact_wins(board, legal, exact, scratch=None) -> None:
    """`scratch` is per-board int32 space of 81 entries, allocated when not given."""
    n = board.shape[0]
    if scratch is None:
        scratch = torch.empty(n, N_SQUARES, dtype=torch.int32, device=board.device)
    exact_kernel[(n,)](board, legal, scratch, exact, BLOCK=BLOCK)


def loses_in_two(board, legal, since, clock, out, scratch=None) -> None:
    """`out` (n, 648) int8: legal moves after which the opponent wins at once.
    `since` and `clock` are per-board int32, the plies since the last capture
    and the capture clock (0 for none): a non-capture move on which the clock
    expires draws and is never a loss. `scratch` is per-board int32 space of
    `LOSS_SCRATCH` entries, allocated when not given."""
    n = board.shape[0]
    out.zero_()
    if scratch is None:
        scratch = torch.empty(n, LOSS_SCRATCH, dtype=torch.int32, device=board.device)
    loses_in_two_kernel[(n,)](
        board, legal, since, clock, out, scratch, BOUND=MOBILITY_BOUND, SCRATCH=LOSS_SCRATCH, BLOCK=BLOCK
    )


def wins_in_three(board, legal, since, clock, out, scratch=None) -> None:
    """`out` (n, 648) int8: legal moves after which every reply leaves a win at once.
    `since` and `clock` as in `loses_in_two`: a non-capture move or reply on
    which the clock expires draws, so it is neither a win in three nor part of
    one. `scratch` is per-board int32 space of `WIN3_SCRATCH` entries, allocated
    when not given."""
    n = board.shape[0]
    out.zero_()
    if scratch is None:
        scratch = torch.empty(n, WIN3_SCRATCH, dtype=torch.int32, device=board.device)
    wins_in_three_kernel[(n,)](
        board,
        legal,
        since,
        clock,
        out,
        scratch,
        BOUND=MOBILITY_BOUND,
        TWO_BOUND=TWO_MOVE_BOUND,
        SCRATCH=WIN3_SCRATCH,
        BLOCK=BLOCK,
    )


def derive_batch(board, since_capture, ply, clock):
    """`(planes bf16, legal bool, legal count)` for a batch of canonical boards."""
    n = board.shape[0]
    legal = torch.empty(n, N_ACTIONS, dtype=torch.int8, device=board.device)
    count = torch.empty(n, dtype=torch.int32, device=board.device)
    planes = torch.empty(n, N_PLANES, N_SQUARES, dtype=torch.bfloat16, device=board.device)
    scratch = torch.empty(n, DERIVE_SCRATCH, dtype=torch.int32, device=board.device)
    derive(
        board.contiguous(),
        since_capture.contiguous(),
        ply.contiguous(),
        clock.contiguous(),
        legal,
        count,
        planes,
        scratch,
    )
    return planes, legal.bool(), count
