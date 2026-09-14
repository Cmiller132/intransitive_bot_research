"""Search semantics with a tiny network stand-in: halving schedule, the exact
tactics of DESIGN item 14 (wins at once, losses in two, wins in three), the
policy target they shape, sign-alternating backups, clock and repetition
terminals, and tree reuse across a played move. Runs on the CPU through the
Triton interpreter when no GPU is available."""

import os

import engine
import numpy as np
import pytest
import torch

from conv import kernels
from conv.config import Search as SearchConfig
from conv.env import END_WIN, Env
from conv.planes import N_ACTIONS, N_SQUARES, initial_board, random_positions, tactics_reference
from conv.search import GumbelSearch, RepetitionHistory, halving_plan, spread, unpack_tactics

# TRITON_INTERPRET=1 in the environment runs everything on the CPU.
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


def flat(q_value=0.0, contempt_source="wdl", draw_value=0.0):
    """Uniform policy and a constant Q: the rules alone drive the search."""

    def evaluate(board, since, ply, clock):
        planes, legal, count = kernels.derive_batch(board, since, ply, clock)
        zeros = torch.zeros(board.shape[0], N_ACTIONS, device=board.device)
        q = torch.full_like(zeros, q_value)
        draw = (
            torch.full_like(zeros[:, 0], draw_value) if contempt_source == "wdl" else torch.full_like(zeros, draw_value)
        )
        return zeros, q, torch.full_like(zeros[:, 0], q_value), draw, legal, count

    return evaluate


def shuttle(board, since, ply, clock):
    """A rock on b2 steps to c2 and a rock on c2 steps back to b2: with one rock
    a side, both sides shuttle and the root position recurs after four plies."""
    planes, legal, count = kernels.derive_batch(board, since, ply, clock)
    logits = torch.zeros(board.shape[0], N_ACTIONS, device=board.device)
    rock = planes[:, 0].float()  # own rock plane in the mover's frame
    logits[:, 4 * N_SQUARES + 10] = 12.0 * rock[:, 10]
    logits[:, 3 * N_SQUARES + 11] = 12.0 * rock[:, 11]
    zeros = torch.zeros_like(logits)
    return logits, zeros, zeros[:, 0], zeros[:, 0], legal, count


def make(
    n,
    sims=8,
    cands=4,
    cheap_sims=4,
    cheap_cands=2,
    penalty=0.05,
    max_plies=1000,
    clock=200,
    reuse=8,
    contempt_source="wdl",
):
    cfg = SearchConfig(
        sims=sims,
        candidates=cands,
        cheap_sims=cheap_sims,
        cheap_candidates=cheap_cands,
        reuse_nodes=reuse,
        temperature_plies=0,
        contempt_source=contempt_source,
    )
    env = Env(n, DEVICE, max_plies, clock, clock, seed=1)
    history = RepetitionHistory(n, 64, DEVICE)
    history.reset(env.board, env.ply)
    search = GumbelSearch(cfg, n, DEVICE, env.clock, penalty, max_plies, seed=1, history=history)
    return env, search, history


def load(env, boards, since=None, ply=None):
    env.board.copy_(torch.tensor(boards, dtype=torch.int8, device=DEVICE))
    if since is not None:
        env.since_capture.copy_(torch.tensor(since, dtype=torch.int32, device=DEVICE))
    if ply is not None:
        env.ply.copy_(torch.tensor(ply, dtype=torch.int32, device=DEVICE))
    env.derive()


def run(env, search, evaluate, full=True):
    """`(target, value, action, candidates, candidate_q, visited, tactics)` with
    the target over the 648 actions and the labels unpacked to `(n, 648, 3)`."""
    root = search(env.board, env.since_capture, env.ply, env.legal, evaluate, full)
    target = spread(root.moves, root.target)
    tactics = unpack_tactics(spread(root.moves, root.tactics))
    return target, root.value, root.action, root.candidates, root.candidate_q, root.visited, tactics


def slots_of(search, row: int, actions) -> torch.Tensor:
    """The root slots of `actions` on `row`."""
    root = search.actions[0, row].long()
    return torch.tensor([int((root == a).nonzero()[0, 0]) for a in actions], device=DEVICE)


def place(cells: dict[int, int]) -> list[int]:
    board = [0] * N_SQUARES
    for square, cell in cells.items():
        board[square] = cell
    return board


def test_halving_schedule_for_128_sims_16_candidates():
    plan = halving_plan(128, 16)
    assert len(plan) == 128
    phases = [m for m, _ in plan]
    assert phases.count(16) == 32 and phases.count(8) == 32 and phases.count(4) == 32 and phases.count(2) == 32
    assert halving_plan(16, 4) == [(4, s) for s in range(4)] * 2 + [(2, s) for s in range(2)] * 4


def test_contempt_draw_uses_child_wdl_or_root_q_mass():
    env, search, _ = make(1, sims=2, cands=2, cheap_sims=1, cheap_cands=1)

    def wdl_draw(board, since, ply, clock):
        logits, q, value, _, legal, count = flat()(board, since, ply, clock)
        draw = torch.where(ply == 0, 0.2, 0.8)
        return logits, q, value, draw, legal, count

    run(env, search, wdl_draw)
    valid = search.lanes[0] < search.n_legal[0, 0]
    expanded = search.child[0, 0] >= 0
    assert expanded[valid].any() and (~expanded & valid).any()
    expanded_draw = search.root_draw[0][expanded & valid]
    assert torch.allclose(expanded_draw, torch.full_like(expanded_draw, 0.8))
    assert torch.allclose(
        search.root_draw[0][~expanded & valid], torch.full_like(search.root_draw[0][~expanded & valid], 0.2)
    )

    env, search, _ = make(1, sims=2, cands=2, cheap_sims=1, cheap_cands=1, contempt_source="q")

    def q_draw(board, since, ply, clock):
        logits, q, value, _, legal, count = flat(contempt_source="q")(board, since, ply, clock)
        draw = torch.arange(N_ACTIONS, device=board.device).float()[None].expand(board.shape[0], -1) / N_ACTIONS
        return logits, q, value, draw, legal, count

    run(env, search, q_draw)
    valid = search.lanes[0] < search.n_legal[0, 0]
    expected = search.actions[0, 0].float() / N_ACTIONS
    assert torch.allclose(search.root_draw[0][valid], expected[valid])


def test_exact_win_is_admitted_and_played():
    board = place({71: 1, 0: 5, 40: 4})  # own rock i8 wins on i9; other pieces around
    env, search, _ = make(2)
    load(env, [board, initial_board()])
    target, value, action, _, _, _, tactics = run(env, search, flat())
    at_once = [a for a in np.flatnonzero(env.legal[0].cpu().numpy()) if engine.apply(board, 0, 0, int(a), 0)[3] == 1]
    exact = np.flatnonzero((tactics[0, :, 0] | tactics[0, :, 2]).cpu().numpy())
    assert action[0].item() in at_once
    assert value[0].item() > 0.9
    # Every exactly winning move is worth +1, so together they take the target mass.
    assert target[0, exact].sum().item() > 0.9
    assert tactics[0, at_once, 0].all() and tactics[0, :, 0].sum().item() == len(at_once)
    assert action[1].item() in np.flatnonzero(env.legal[1].cpu().numpy())


@pytest.mark.parametrize("cands", [1, 2, 4, 16])
@pytest.mark.parametrize("full", [False, True])
def test_immediate_win_survives_candidate_limits_and_halving(cands, full, monkeypatch):
    board = place(
        {13: 1, 15: 2, 20: 1, 21: 3, 22: 2, 23: 3, 31: 6, 40: 5, 42: 4, 43: 5, 51: 6, 58: 4, 70: 3, 77: 4}
        if cands == 16
        else {71: 1, 0: 5, 40: 4}
    )
    cheap_cands = min(cands, 4)
    env, search, _ = make(
        1,
        sims=max(2, cands + 1),
        cands=cands,
        cheap_sims=max(2, cheap_cands + 1),
        cheap_cands=cheap_cands,
        clock=96,
    )
    load(env, [board], since=[5], ply=[56])
    at_once = [a for a in np.flatnonzero(env.legal[0].cpu().numpy()) if engine.apply(board, 5, 56, int(a), 96)[3] == 1]
    assert at_once
    immediate_actions = torch.tensor(at_once, dtype=torch.long, device=DEVICE)

    def hostile_prior(board, since, ply, clock):
        logits, q, value, draw, legal, count = flat()(board, since, ply, clock)
        logits.index_fill_(1, immediate_actions, -1000.0)
        return logits, q, value, draw, legal, count

    def assert_immediate_active():
        active = search.survivors[0, : int(search.previous_m[0])]
        assert search.root_win1[0, active].any()
        assert (active < search.n_legal[0, 0]).all()
        assert not search.root_excluded[0, active].any()

    start, simulate = search._start, search._simulate

    def checked_start(*args):
        start(*args)
        assert search.wins[0, 0].sum() > search.root_win1[0].sum()
        if cands == 16:
            assert search.wins[0, 0].sum() > cands
        assert_immediate_active()

    def checked_simulate():
        simulate()
        if DEVICE == "cpu":
            assert_immediate_active()

    monkeypatch.setattr(search, "_start", checked_start)
    monkeypatch.setattr(search, "_simulate", checked_simulate)
    root = search(env.board, env.since_capture, env.ply, env.legal, hostile_prior, full)
    active = search.survivors[0, : int(search.previous_m[0])]
    assert search.played_slot[0] in active
    assert root.action[0].item() in at_once
    assert root.played_q[0] == 1.0
    env.step(root.action)
    assert env.done[0] and env.end_reason[0] == END_WIN


@pytest.mark.parametrize("hostile", [False, True])
def test_win_in_three_is_flagged_and_played(hostile):
    # Own scissors g8; the enemy paper on i9 is two king steps away, so stepping to
    # h8 or h9 wins next move whatever the reply does.
    board = place({69: 3, 80: 5, 40: 5})
    env, search, _ = make(1, sims=8, cands=4, cheap_cands=2)
    load(env, [board])
    # Direction (0, 1) to h8 and (1, 1) to h9, both from g8.
    winning = [4 * N_SQUARES + 69, 7 * N_SQUARES + 69]
    winning_actions = torch.tensor(winning, dtype=torch.long, device=DEVICE)

    def evaluate(board, since, ply, clock):
        logits, q, value, draw, legal, count = flat()(board, since, ply, clock)
        logits.index_fill_(1, winning_actions, -1000.0 if hostile else 0.0)
        return logits, q, value, draw, legal, count

    target, value, action, _, _, _, tactics = run(env, search, evaluate)
    assert tactics[0, :, 0].sum().item() == 0
    assert list(np.flatnonzero(tactics[0, :, 2].cpu().numpy())) == winning
    slots = slots_of(search, 0, winning)
    assert search.wins[0, 0][slots].all() and search.wins[0, 0].sum().item() == len(winning)
    assert action[0].item() in winning
    assert search.played_slot[0] in search.survivors[0, : int(search.previous_m[0])]
    if not hostile:
        assert value[0].item() > 0.9
        assert target[0, winning].sum().item() > 0.9


def test_losing_moves_are_excluded_from_selection_and_target():
    # Own scissors b2 is the last own piece; the enemy scissors on b1 enters a1 next
    # move unless b2-a1 blocks it. Every other move loses in two.
    board = place({10: 3, 1: 6, 18: 5, 11: 5, 22: 4, 40: 5})
    env, search, _ = make(1, sims=8, cands=8, cheap_cands=2)
    load(env, [board])
    target, value, action, candidates, candidate_q, _, tactics = run(env, search, flat(-1.0))
    legal = np.flatnonzero(env.legal[0].cpu().numpy())
    safe = 0 * N_SQUARES + 10  # b2-a1: direction (-1, -1)
    losing = [int(a) for a in legal if a != safe]
    reference = tactics_reference(board)
    assert [bool(x) for x in tactics[0, :, 1]] == reference[1]
    assert action[0].item() == safe
    assert target[0, losing].sum().item() == 0.0
    assert target[0, safe].item() == 1.0
    lost, kept = slots_of(search, 0, losing), slots_of(search, 0, [safe])
    assert search.root_excluded[0, lost].all() and not search.root_excluded[0, kept].any()
    # A losing edge ends the descent at -1 without expanding a child.
    assert (search.child[0, 0, lost] == -1).all()
    assert (candidate_q[0][torch.isin(candidates[0], torch.tensor(losing, device=DEVICE))] == -1.0).all()
    assert value[0].item() > -1.0


def test_root_tactics_match_the_reference():
    positions = random_positions(4, seed=11)
    env, search, _ = make(len(positions), sims=4, cands=2, cheap_cands=2)
    load(env, [b for b, _, _ in positions], [s for _, s, _ in positions], [p for _, _, p in positions])
    *_, tactics = run(env, search, flat())
    for row, (board, since, _) in enumerate(positions):
        expected = tactics_reference(board, since, int(env.clock[row]))
        for channel in range(3):
            assert [bool(x) for x in tactics[row, :, channel]] == expected[channel], (row, channel)


def test_clock_terminal_pays_material_penalty():
    board = initial_board()
    board[43] = 0  # remove one enemy paper: the root mover is ahead
    env, search, _ = make(1, sims=4, cands=2, clock=5)
    load(env, [board], since=[4], ply=[10])
    run(env, search, flat())
    # Every child of the root reaches the clock at once. The child's mover is behind
    # and receives +penalty, so the root, which is ahead, sees the penalty on every edge.
    expanded = search.terminal[1 : search.node_count[0].item(), 0]
    assert expanded.all()
    edge_values = search.total[0, 0][search.count[0, 0] > 0] / search.count[0, 0][search.count[0, 0] > 0]
    assert torch.allclose(edge_values, torch.full_like(edge_values, -0.05), atol=1e-6)


def test_twofold_repetition_is_a_draw_in_the_tree():
    board = place({10: 1, 70: 4})  # own rock b2, enemy rock h8 (b2 in its own frame)
    env, search, history = make(1, sims=16, cands=4)
    load(env, [board])
    history.reset(env.board, env.ply)
    run(env, search, shuttle)
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
    evaluate = flat()  # one actor across both searches, as in training: a new one recaptures the graph
    run(env, search, evaluate)
    assert search.metrics()["reuse_kept_frac"] == 0.0
    root_counts = search.count[0, 0].clone()
    slot = torch.argmax(root_counts)[None]
    child = search.child[0, 0, slot].item()
    assert child > 0
    child_counts = search.count[child, 0].clone()
    # The search re-roots on the move it returned, so that is the move to play.
    search.played_slot.copy_(slot)
    action = search.actions[0, 0, slot].int()
    env.step(action)
    history.push(env.board, env.since_capture, env.ply, env.done)
    search.advance(env.done)
    run(env, search, evaluate)
    assert search.kept[0].item()
    assert (search.count[0, 0] >= child_counts).all()
    assert search.count[0, 0].sum().item() == child_counts.sum().item() + 16
    assert search.metrics()["reuse_kept_frac"] == 1.0
