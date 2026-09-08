"""Search semantics with a tiny network stand-in: halving schedule, exact-win
admission and play, sign-alternating backups, clock and repetition terminals,
and tree reuse across a played move. Runs on the CPU through the Triton
interpreter when no GPU is available."""

import os

import engine
import numpy as np
import torch

from sq.config import Search as SearchConfig
from sq.env import Env
from sq.planes import N_ACTIONS, N_SQUARES, initial_board
from sq.search import GumbelSearch, RepetitionHistory, halving_plan

# TRITON_INTERPRET=1 in the environment runs everything on the CPU.
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


def flat_net(planes):
    """Uniform policy, zero Q: the rules alone drive the search."""
    b = planes.shape[0]
    return torch.zeros(b, N_ACTIONS, device=planes.device), torch.zeros(b, N_ACTIONS, device=planes.device)


def make(n, sims=8, cands=4, cheap_sims=4, cheap_cands=2, penalty=0.05, max_plies=1000, clock=0, reuse=8):
    cfg = SearchConfig(
        sims=sims,
        candidates=cands,
        cheap_sims=cheap_sims,
        cheap_candidates=cheap_cands,
        reuse_nodes=reuse,
        temperature_plies=0,
    )
    capture_clock = torch.tensor(clock, dtype=torch.int32, device=DEVICE)
    env = Env(n, DEVICE, max_plies, capture_clock)
    history = RepetitionHistory(n, 64, DEVICE)
    history.reset(env.board, env.ply)
    search = GumbelSearch(cfg, n, DEVICE, capture_clock, penalty, max_plies, seed=1, history=history)
    return env, search, history


def load(env, boards, since=None, ply=None):
    env.board.copy_(torch.tensor(boards, dtype=torch.int8, device=DEVICE))
    if since is not None:
        env.since_capture.copy_(torch.tensor(since, dtype=torch.int32, device=DEVICE))
    if ply is not None:
        env.ply.copy_(torch.tensor(ply, dtype=torch.int32, device=DEVICE))
    env.derive()


def test_halving_schedule_for_128_sims_16_candidates():
    plan = halving_plan(128, 16)
    assert len(plan) == 128
    phases = [m for m, _ in plan]
    assert phases.count(16) == 32 and phases.count(8) == 32 and phases.count(4) == 32 and phases.count(2) == 32
    assert halving_plan(16, 4) == [(4, s) for s in range(4)] * 2 + [(2, s) for s in range(2)] * 4


def test_exact_win_is_admitted_and_played():
    board = [0] * N_SQUARES
    board[71], board[0], board[40] = 1, 5, 4  # own rock i8 wins on i9; other pieces around
    env, search, _ = make(2)
    load(env, [board, initial_board()])
    target, value, action, candidates, candidate_q, visited = search(
        env.board, env.since_capture, env.ply, env.legal, env.planes, flat_net, True
    )
    winning = [a for a in np.flatnonzero(env.legal[0].cpu().numpy()) if engine.apply(board, 0, 0, int(a), 0)[3] == 1]
    assert action[0].item() in winning
    assert value[0].item() > 0.9
    assert target[0, winning].sum().item() > 0.9
    assert action[1].item() in np.flatnonzero(env.legal[1].cpu().numpy())


def test_backups_alternate_sign_and_losing_moves_are_avoided():
    # Enemy paper on b2 reaches our home next move unless our scissors on c3 captures it.
    # Sixteen candidates admit every legal move, so the capture is always considered.
    board = [0] * N_SQUARES
    board[10], board[20], board[60], board[79] = 5, 3, 1, 4
    env, search, _ = make(1, sims=32, cands=16)
    load(env, [board])
    target, value, action, *_ = search(env.board, env.since_capture, env.ply, env.legal, env.planes, flat_net, True)
    capture = [a for a in np.flatnonzero(env.legal[0].cpu().numpy()) if a % N_SQUARES == 20 and a // N_SQUARES == 0]
    assert action[0].item() == capture[0]
    visited = search.count[0, 0] > 0
    losing = visited & (torch.arange(N_ACTIONS, device=DEVICE) != capture[0])
    assert (search.total[0, 0][losing] / search.count[0, 0][losing]).max().item() < -0.9


def test_clock_terminal_pays_material_penalty():
    board = initial_board()
    board[43] = 0  # remove one enemy paper: the root mover is ahead
    env, search, _ = make(1, sims=4, cands=2, clock=5)
    load(env, [board], since=[4], ply=[10])
    search(env.board, env.since_capture, env.ply, env.legal, env.planes, flat_net, True)
    # Every child of the root reaches the clock at once. The child's mover is behind
    # and receives +penalty, so the root, which is ahead, sees the penalty on every edge.
    expanded = search.terminal[1 : search.node_count[0].item(), 0]
    assert expanded.all()
    edge_values = search.total[0, 0][search.count[0, 0] > 0] / search.count[0, 0][search.count[0, 0] > 0]
    assert torch.allclose(edge_values, torch.full_like(edge_values, -0.05), atol=1e-6)


def shuttle_net(planes):
    """A rock on b2 steps to c2 and a rock on c2 steps back to b2: with one rock
    a side, both sides shuttle and the root position recurs after four plies."""
    b = planes.shape[0]
    logits = torch.zeros(b, N_ACTIONS, device=planes.device)
    rock = planes[:, 0]  # own rock plane in the mover's frame
    logits[:, 4 * N_SQUARES + 10] = 12.0 * rock[:, 10]
    logits[:, 3 * N_SQUARES + 11] = 12.0 * rock[:, 11]
    return logits, torch.zeros(b, N_ACTIONS, device=planes.device)


def test_twofold_repetition_is_a_draw_in_the_tree():
    board = [0] * N_SQUARES
    board[10], board[70] = 1, 4  # own rock b2, enemy rock h8 (b2 in its own frame)
    env, search, history = make(1, sims=16, cands=4)
    load(env, [board])
    history.reset(env.board, env.ply)
    search(env.board, env.since_capture, env.ply, env.legal, env.planes, shuttle_net, True)
    assert search.metrics()["repetition_draw_frac"] > 0
    nodes = search.node_count[0].item()
    draws = search.terminal[1:nodes, 0] & (search.value[1:nodes, 0] == 0)
    assert draws.any()
    # The repeated position is the root itself, four plies down the shuttle line.
    root_hash = search.hash[0, 0]
    assert ((search.hash[1:nodes, 0] == root_hash) & search.terminal[1:nodes, 0]).any()
    assert (search.plies[1:nodes, 0][search.hash[1:nodes, 0] == root_hash] == 4).all()


def test_reuse_keeps_the_played_subtree():
    env, search, history = make(1, sims=16, cands=4, reuse=16)
    search(env.board, env.since_capture, env.ply, env.legal, env.planes, flat_net, True)
    assert search.metrics()["reuse_kept_frac"] == 0.0
    root_counts = search.count[0, 0].clone()
    action = torch.argmax(root_counts)[None]
    child = search.child[0, 0, action].item()
    assert child > 0
    child_counts = search.count[child, 0].clone()
    env.step(action.to(torch.int32))
    history.push(env.board, env.since_capture, env.ply, env.done)
    search.advance(action, env.done)
    search(env.board, env.since_capture, env.ply, env.legal, env.planes, flat_net, True)
    assert search.kept[0].item()
    assert (search.count[0, 0] >= child_counts).all()
    assert search.count[0, 0].sum().item() == child_counts.sum().item() + 16
    assert search.metrics()["reuse_kept_frac"] == 1.0
