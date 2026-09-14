"""The GPU kernels must agree with the engine on random positions: legal masks,
applied children with their counters and end flags, the observation planes
against the reference encoder, and the three exact tactics against the
brute-force oracle. Constructed positions cover each tactical motif, and the
environment is checked on its per-board capture clock. Runs on the CPU through
the Triton interpreter when TRITON_INTERPRET=1."""

import os

import engine
import numpy as np
import pytest
import torch

from conv import kernels
from conv.env import END_MAX_PLIES, Env
from conv.planes import CLOCK_SCALE, N_SQUARES, initial_board, random_positions, reference, tactics_reference

# TRITON_INTERPRET=1 in the environment runs everything on the CPU.
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


@pytest.fixture(scope="module")
def positions():
    """`(board, since_capture, ply, clock)` with the counters drawn independently
    of the play that produced the board."""
    rng = np.random.default_rng(5)
    return [
        (board, int(rng.integers(0, 201)), ply, int(rng.integers(50, 201)))
        for board, _, ply in random_positions(48, seed=11)
    ]


def tensors(positions):
    board = torch.tensor([p[0] for p in positions], dtype=torch.int8, device=DEVICE)
    since = torch.tensor([p[1] for p in positions], dtype=torch.int32, device=DEVICE)
    ply = torch.tensor([p[2] for p in positions], dtype=torch.int32, device=DEVICE)
    clock = torch.tensor([p[3] for p in positions], dtype=torch.int32, device=DEVICE)
    return board, since, ply, clock


def motifs() -> dict[str, list[int]]:
    """One board per tactical motif, all in the mover's frame."""
    goal_threat = [0] * N_SQUARES
    # Two rocks beside an empty i9; the enemy is too far to stop both.
    goal_threat[70] = goal_threat[71] = goal_threat[40] = 1
    goal_threat[30] = goal_threat[31] = 5
    reply_wins = [0] * N_SQUARES
    # The enemy paper on b2 steps onto a1 whatever the lone rock does.
    reply_wins[40], reply_wins[10] = 1, 5
    elimination = [0] * N_SQUARES
    # The last enemy piece, a scissors on i1, can only run onto a rock's square.
    elimination[16] = elimination[17] = elimination[40] = 1
    elimination[8] = 6
    smothered = [0] * N_SQUARES
    # Both rocks have one square between them; a paper stepping onto h1 stalemates.
    smothered[17] = smothered[72] = 1
    for square in (7, 16, 26, 63, 64, 73):
        smothered[square] = 5
    smother_win = [0] * N_SQUARES
    # The enemy rock on i1 already has no move: nearly every move wins at once.
    smother_win[8] = 4
    smother_win[7] = smother_win[16] = smother_win[17] = 2
    smother_win[40] = 1
    return {
        "goal_threat": goal_threat,
        "reply_wins": reply_wins,
        "elimination": elimination,
        "smothered": smothered,
        "smother_win": smother_win,
    }


def tactics(board: torch.Tensor, legal: torch.Tensor, since: torch.Tensor, clock: torch.Tensor):
    """The three tactics kernels over a batch of boards under their clocks."""
    win1 = torch.zeros(board.shape[0], 648, dtype=torch.int8, device=DEVICE)
    loss2 = torch.zeros_like(win1)
    win3 = torch.zeros_like(win1)
    kernels.exact_wins(board, legal, win1)
    kernels.loses_in_two(board, legal, since, clock, loss2)
    kernels.wins_in_three(board, legal, since, clock, win3)
    return win1, loss2, win3


def test_legal_mask_and_planes_match_engine(positions):
    board, since, ply, clock = tensors(positions)
    planes, legal, count = kernels.derive_batch(board, since, ply, clock)
    for i, (b, s, p, c) in enumerate(positions):
        expected = np.array(engine.legal_mask(b), dtype=bool)
        assert np.array_equal(legal[i].cpu().numpy(), expected), f"legal mask differs at position {i}"
        assert count[i].item() == expected.sum()
        ref = torch.tensor(reference(b, s, p, c))
        # One bf16 ulp of tolerance: the interpreter truncates where the GPU rounds.
        assert torch.allclose(planes[i].float().cpu(), ref, atol=4e-3), f"planes differ at position {i}"


def test_apply_matches_engine_including_end_flags(positions):
    board, since, ply, _ = tensors(positions)
    rng = np.random.default_rng(0)
    actions, expected = [], []
    for b, s, p, _ in positions:
        legal = np.flatnonzero(engine.legal_mask(b))
        action = int(rng.choice(legal)) if len(legal) else 0
        actions.append(action)
        expected.append(engine.apply(b, s, p, action, 0) if len(legal) else None)
    action = torch.tensor(actions, dtype=torch.int32, device=DEVICE)
    flags = torch.zeros(3, len(positions), dtype=torch.int8, device=DEVICE)
    kernels.apply(board, action, since, ply, flags[0], flags[1], flags[2])
    for i in range(len(positions)):
        if expected[i] is None:
            continue
        child, child_since, child_ply, outcome = expected[i]
        assert board[i].cpu().tolist() == list(child), f"child differs at position {i}"
        assert since[i].item() == child_since and ply[i].item() == child_ply
        won = bool(flags[0, i] | flags[1, i]) or not any(engine.legal_mask(list(child)))
        assert won == (outcome == 1), f"end flag differs at position {i}"
        assert bool(flags[2, i]) == (child_since == 0)


def test_tactics_match_the_oracle(positions):
    board, since, ply, clock = tensors(positions)
    _, legal, _ = kernels.derive_batch(board, since, ply, clock)
    win1, loss2, win3 = tactics(board, legal, since, clock)
    for i, (b, s, _, c) in enumerate(positions):
        expected = tactics_reference(b, s, c)
        for name, got, want in zip(("wins", "loses", "wins in three"), (win1, loss2, win3), expected, strict=True):
            assert np.array_equal(got[i].cpu().numpy().astype(bool), np.array(want)), f"{name} differs at {i}"


def test_tactics_cover_each_motif():
    boards = motifs()
    board = torch.tensor(list(boards.values()), dtype=torch.int8, device=DEVICE)
    zero = torch.zeros(len(boards), dtype=torch.int32, device=DEVICE)
    planes, legal, _ = kernels.derive_batch(board, zero, zero, zero)
    win1, loss2, win3 = tactics(board, legal, zero, zero)
    found = {}
    for i, (name, b) in enumerate(boards.items()):
        expected = tactics_reference(b)
        for label, got, want in zip(("wins", "loses", "wins in three"), (win1, loss2, win3), expected, strict=True):
            assert np.array_equal(got[i].cpu().numpy().astype(bool), np.array(want)), f"{label} differs on {name}"
        found[name] = tuple(sum(flags) for flags in expected)
    assert found["goal_threat"][2] and found["elimination"][2], "a win in three should exist"
    assert found["reply_wins"][1] and found["smothered"][1], "a loss in two should exist"
    assert found["smother_win"][0], "a win at once should exist"
    # The smothered board leaves the mover two moves, so `_any_win` runs its full
    # stalemate scan rather than stopping at the mobility bound.
    mobility = planes[list(boards).index("smothered"), 38, 0].float().item() * 64
    assert mobility <= kernels.MOBILITY_BOUND


def test_tactics_respect_the_clock():
    """A non-capture move on which the capture clock expires draws, so it is
    neither a loss in two nor a win in three, and a non-capture reply that
    draws refutes a win in three; captures reset the clock and keep their
    labels. The random boards and the motifs are checked with the clock one,
    two and three plies from expiring against the oracle playing the engine
    under that clock, and the edge must change some label somewhere."""
    boards = [b for b, _, _ in random_positions(48, seed=11)] + list(motifs().values())
    board = torch.tensor(boards, dtype=torch.int8, device=DEVICE)
    zero = torch.zeros(len(boards), dtype=torch.int32, device=DEVICE)
    _, legal, _ = kernels.derive_batch(board, zero, zero, zero)  # legality does not depend on the clock
    clock = torch.full((len(boards),), 100, dtype=torch.int32, device=DEVICE)
    changed = 0
    for left in (1, 2, 3):
        win1, loss2, win3 = tactics(board, legal, clock - left, clock)
        for i, b in enumerate(boards):
            expected = tactics_reference(b, 100 - left, 100)
            blind = tactics_reference(b)
            changed += sum(x != y for e, bl in zip(expected, blind, strict=True) for x, y in zip(e, bl, strict=True))
            for name, got, want in zip(("wins", "loses", "wins in three"), (win1, loss2, win3), expected, strict=True):
                assert np.array_equal(got[i].cpu().numpy().astype(bool), np.array(want)), (
                    f"{name} differs at {i} with {left} plies left"
                )
    assert changed > 0, "the clock edge should change some label"


def plane_15_matches(planes: torch.Tensor, clock: torch.Tensor) -> bool:
    """Plane 15 counts a fresh clock down from a position with no capture yet."""
    want = (clock.float() / CLOCK_SCALE)[:, None].expand(clock.shape[0], N_SQUARES)
    return bool(torch.allclose(planes[:, 15].float(), want, atol=4e-3))


def test_env_clock_and_reset():
    n = 6
    env = Env(n, DEVICE, max_plies=1, clock_min=50, clock_max=200, seed=7)
    first = env.clock.clone()
    assert int(first.min()) >= 50 and int(first.max()) <= 200
    assert plane_15_matches(env.planes, first)
    env.reset_all()
    assert not torch.equal(first, env.clock), "a reset draws fresh clocks"
    assert int(env.clock.min()) >= 50 and int(env.clock.max()) <= 200
    before = env.clock.clone()
    action = env.legal.int().argmax(dim=1).to(torch.int32)
    env.step(action)
    start = torch.tensor(initial_board(), dtype=torch.int8, device=DEVICE)
    assert bool(env.done.all()) and bool((env.end_reason == END_MAX_PLIES).all())
    assert torch.equal(env.board, start[None].expand(n, -1)), "a finished board restarts"
    assert bool((env.ply == 0).all()) and bool((env.since_capture == 0).all())
    assert not torch.equal(before, env.clock), "a finished board draws a fresh clock"
    assert int(env.clock.min()) >= 50 and int(env.clock.max()) <= 200
    assert plane_15_matches(env.planes, env.clock)
