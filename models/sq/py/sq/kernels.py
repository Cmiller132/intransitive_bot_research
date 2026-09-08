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

from .planes import CLOCK_SCALE, N_ACTIONS, N_PLANES, N_SQUARES

BLOCK = 128


@triton.jit
def _delta(d):
    # Direction index 0..7 -> (rank delta, file delta), skipping the (0, 0) entry.
    k = d + (d >= 4).to(tl.int32)
    return k // 3 - 1, k % 3 - 1


@triton.jit
def _beats(a, b):
    # Own type a (1..3) beats enemy type b (1..3).
    return ((a - b + 3) % 3) == 1


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
    legal_ptr,
    any_legal_ptr,
    planes_ptr,
    CLOCK: tl.constexpr,
    NP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Legal-move mask, the number of legal moves, and the 25 observation
    planes for every board."""
    pid = tl.program_id(0)
    base = pid * 81
    idx = tl.arange(0, BLOCK)
    m = idx < 81
    b = tl.load(board_ptr + base + idx, mask=m, other=0).to(tl.int32)
    r = idx // 9
    c = idx % 9
    own = (b >= 1) & (b <= 3)
    enemy = b >= 4
    kind = tl.where(enemy, b - 3, b)
    has_r = tl.zeros([BLOCK], dtype=tl.int32)
    has_p = tl.zeros([BLOCK], dtype=tl.int32)
    has_s = tl.zeros([BLOCK], dtype=tl.int32)
    own_mobility = tl.zeros([BLOCK], dtype=tl.int32)
    enemy_mobility = tl.zeros([BLOCK], dtype=tl.int32)
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
            has_r += (tv == 4).to(tl.int32)
            has_p += (tv == 5).to(tl.int32)
            has_s += (tv == 6).to(tl.int32)
    own_total = tl.sum(own_mobility, axis=0)
    tl.store(any_legal_ptr + pid, own_total)

    ob = planes_ptr + pid * (NP * 81)
    for v in tl.static_range(1, 7):
        tl.store(ob + (v - 1) * 81 + idx, (b == v).to(tl.float32).to(tl.bfloat16), mask=m)
    empty = b == 0
    # Threat planes: an adjacent enemy of that type could move here.
    thr_r = (empty | (own & _beats(1, kind))) & (has_r > 0)
    thr_p = (empty | (own & _beats(2, kind))) & (has_p > 0)
    thr_s = (empty | (own & _beats(3, kind))) & (has_s > 0)
    tl.store(ob + 6 * 81 + idx, thr_r.to(tl.float32).to(tl.bfloat16), mask=m)
    tl.store(ob + 7 * 81 + idx, thr_p.to(tl.float32).to(tl.bfloat16), mask=m)
    tl.store(ob + 8 * 81 + idx, thr_s.to(tl.float32).to(tl.bfloat16), mask=m)
    tl.store(ob + 9 * 81 + idx, ((idx == 0) | (idx == 80)).to(tl.float32).to(tl.bfloat16), mask=m)
    ones = (idx * 0 + 1).to(tl.float32)
    since_capture = tl.load(since_capture_ptr + pid).to(tl.float32)
    ply = tl.load(ply_ptr + pid).to(tl.float32)
    tl.store(ob + 10 * 81 + idx, (ones * since_capture / CLOCK).to(tl.bfloat16), mask=m)
    tl.store(ob + 11 * 81 + idx, (ones * ply / CLOCK).to(tl.bfloat16), mask=m)
    tl.store(ob + 12 * 81 + idx, ones.to(tl.bfloat16), mask=m)
    goal_distance = tl.maximum(8 - r, 8 - c)
    home_distance = tl.maximum(r, c)
    min_own = tl.min(tl.where(own, goal_distance, 99), axis=0)
    min_enemy = tl.min(tl.where(enemy, home_distance, 99), axis=0)
    tl.store(ob + 13 * 81 + idx, (tl.where(own, goal_distance, 0).to(tl.float32) / 8).to(tl.bfloat16), mask=m)
    tl.store(ob + 14 * 81 + idx, (tl.where(enemy, home_distance, 0).to(tl.float32) / 8).to(tl.bfloat16), mask=m)
    tl.store(ob + 15 * 81 + idx, (ones * tl.minimum(min_own, 9).to(tl.float32) / 8).to(tl.bfloat16), mask=m)
    tl.store(ob + 16 * 81 + idx, (ones * tl.minimum(min_enemy, 9).to(tl.float32) / 8).to(tl.bfloat16), mask=m)
    tl.store(ob + 17 * 81 + idx, (goal_distance.to(tl.float32) / 8).to(tl.bfloat16), mask=m)
    tl.store(ob + 18 * 81 + idx, (home_distance.to(tl.float32) / 8).to(tl.bfloat16), mask=m)
    enemy_total = tl.sum(enemy_mobility, axis=0)
    n_own = tl.sum(own.to(tl.int32), axis=0)
    n_enemy = tl.sum(enemy.to(tl.int32), axis=0)
    tl.store(ob + 19 * 81 + idx, (own_mobility.to(tl.float32) / 8).to(tl.bfloat16), mask=m)
    tl.store(ob + 20 * 81 + idx, (enemy_mobility.to(tl.float32) / 8).to(tl.bfloat16), mask=m)
    tl.store(ob + 21 * 81 + idx, (ones * own_total.to(tl.float32) / 64).to(tl.bfloat16), mask=m)
    tl.store(ob + 22 * 81 + idx, (ones * enemy_total.to(tl.float32) / 64).to(tl.bfloat16), mask=m)
    tl.store(ob + 23 * 81 + idx, (ones * n_own.to(tl.float32) / 10).to(tl.bfloat16), mask=m)
    tl.store(ob + 24 * 81 + idx, (ones * n_enemy.to(tl.float32) / 10).to(tl.bfloat16), mask=m)


@triton.jit
def exact_kernel(board_ptr, legal_ptr, mobility_ptr, exact_ptr, BLOCK: tl.constexpr):
    """Flag every legal action that wins immediately: goal entry, capturing the
    last enemy piece, or leaving the enemy without a legal move. The enemy's
    mobility after a move is the mobility before it, minus the captured piece's
    own moves, plus the enemy moves the vacated and the occupied square gain or
    lose. `mobility_ptr` is per-board scratch of 81 int32."""
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
        kk = d + (d >= 4)
        tr = r + (kk // 3 - 1)
        tc = c + (kk % 3 - 1)
        inb = (tr >= 0) & (tr < 9) & (tc >= 0) & (tc < 9) & m
        t = tr * 9 + tc
        legal = tl.load(legal_ptr + pid * 648 + d * 81 + idx, mask=m, other=0) != 0
        captured = tl.load(board_ptr + base + t, mask=inb, other=0).to(tl.int32)
        captures_enemy = captured >= 4
        removed = tl.where(captures_enemy, tl.load(mobility_ptr + base + t, mask=inb, other=0), 0)
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
                sv = tl.load(board_ptr + base + sr * 9 + sc, mask=s_inb, other=0).to(tl.int32)
                s_enemy = (sv >= 4) & ((sr != tr) | (sc != tc))
                gained_at_source += (s_enemy & ~_beats(sv - 3, kind)).to(tl.int32)
                # Enemy neighbours of the target square: they could move there if it
                # was empty or held an own piece they beat; afterwards only if they
                # beat the mover.
                qr = tr + nr
                qc = tc + nc
                q_inb = (qr >= 0) & (qr < 9) & (qc >= 0) & (qc < 9) & inb
                qv = tl.load(board_ptr + base + qr * 9 + qc, mask=q_inb, other=0).to(tl.int32)
                q_enemy = qv >= 4
                before = (captured == 0) | ((captured >= 1) & (captured <= 3) & _beats(qv - 3, captured))
                after = _beats(qv - 3, kind)
                changed_at_target += tl.where(q_enemy, after.to(tl.int32) - before.to(tl.int32), 0)
        remaining = total - removed + gained_at_source + changed_at_target
        eliminated = (n_enemy - captures_enemy.to(tl.int32)) == 0
        wins = legal & ((t == 80) | eliminated | (remaining == 0))
        tl.store(exact_ptr + pid * 648 + d * 81 + idx, wins.to(tl.int8), mask=m)


def apply(board, action, since_capture, ply, goal, eliminated, captured) -> None:
    """In place: `board` becomes the child, counters advance, flags are written."""
    apply_kernel[(board.shape[0],)](board, action, since_capture, ply, goal, eliminated, captured, BLOCK=BLOCK)


def derive(board, since_capture, ply, legal, any_legal, planes) -> None:
    n = board.shape[0]
    assert legal.shape == (n, N_ACTIONS) and planes.shape == (n, N_PLANES, N_SQUARES)
    derive_kernel[(n,)](
        board, since_capture, ply, legal, any_legal, planes, CLOCK=CLOCK_SCALE, NP=N_PLANES, BLOCK=BLOCK
    )


def exact_wins(board, legal, exact, scratch=None) -> None:
    """`scratch` is per-board int32 space of 81 entries, allocated when not given."""
    n = board.shape[0]
    if scratch is None:
        scratch = torch.empty(n, N_SQUARES, dtype=torch.int32, device=board.device)
    exact_kernel[(n,)](board, legal, scratch, exact, BLOCK=BLOCK)


def derive_batch(board, since_capture, ply):
    """`(planes bf16, legal bool, legal count)` for a batch of canonical boards."""
    n = board.shape[0]
    legal = torch.empty(n, N_ACTIONS, dtype=torch.int8, device=board.device)
    count = torch.empty(n, dtype=torch.int32, device=board.device)
    planes = torch.empty(n, N_PLANES, N_SQUARES, dtype=torch.bfloat16, device=board.device)
    derive(board.contiguous(), since_capture.contiguous(), ply.contiguous(), legal, count, planes)
    return planes, legal.bool(), count
