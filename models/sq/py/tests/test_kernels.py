"""The GPU kernels must agree with the engine on random positions: legal masks,
applied children with their counters and end flags, immediate-win flags, and
the observation planes against the reference encoder. Runs on the CPU through
the Triton interpreter when no GPU is available."""

import os

import engine
import numpy as np
import pytest
import torch

from sq import kernels
from sq.planes import N_SQUARES, random_positions, reference

# TRITON_INTERPRET=1 in the environment runs everything on the CPU.
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


@pytest.fixture(scope="module")
def positions():
    return random_positions(48, seed=11)


def tensors(positions):
    board = torch.tensor([b for b, _, _ in positions], dtype=torch.int8, device=DEVICE)
    since = torch.tensor([s for _, s, _ in positions], dtype=torch.int32, device=DEVICE)
    ply = torch.tensor([p for _, _, p in positions], dtype=torch.int32, device=DEVICE)
    return board, since, ply


def test_legal_mask_and_planes_match_engine(positions):
    board, since, ply = tensors(positions)
    planes, legal, count = kernels.derive_batch(board, since, ply)
    for i, (b, s, p) in enumerate(positions):
        expected = np.array(engine.legal_mask(b), dtype=bool)
        assert np.array_equal(legal[i].cpu().numpy(), expected), f"legal mask differs at position {i}"
        assert count[i].item() == expected.sum()
        ref = torch.tensor(reference(b, s, p))
        # One bf16 ulp of tolerance: the interpreter truncates where the GPU rounds.
        assert torch.allclose(planes[i].float().cpu(), ref, atol=4e-3), f"planes differ at position {i}"


def test_apply_matches_engine_including_end_flags(positions):
    board, since, ply = tensors(positions)
    rng = np.random.default_rng(0)
    actions, expected = [], []
    for b, s, p in positions:
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


def test_exact_wins_match_engine(positions):
    board, since, ply = tensors(positions)
    _, legal, _ = kernels.derive_batch(board, since, ply)
    exact = torch.zeros_like(legal)
    kernels.exact_wins(board, legal, exact)
    for i, (b, s, p) in enumerate(positions):
        for action in np.flatnonzero(engine.legal_mask(b)):
            outcome = engine.apply(b, s, p, int(action), 0)[3]
            assert bool(exact[i, action]) == (outcome == 1), f"exact flag differs at position {i} action {action}"
        assert not (exact[i] & ~legal[i]).any()


def test_exact_wins_cover_each_end_condition():
    goal = [0] * N_SQUARES
    goal[71], goal[0] = 1, 5  # own rock i8, enemy paper a1
    last = [0] * N_SQUARES
    last[40], last[41] = 2, 4  # own paper e5 captures the last enemy rock f5
    smother = [0] * N_SQUARES
    smother[8], smother[7], smother[16], smother[17], smother[40] = 4, 2, 2, 2, 1
    board = torch.tensor([goal, last, smother], dtype=torch.int8, device=DEVICE)
    zero = torch.zeros(3, dtype=torch.int32, device=DEVICE)
    _, legal, _ = kernels.derive_batch(board, zero, zero)
    exact = torch.zeros_like(legal)
    kernels.exact_wins(board, legal, exact)
    for i in range(3):
        expected = [
            a
            for a in np.flatnonzero(legal[i].cpu().numpy())
            if engine.apply(board[i].cpu().tolist(), 0, 0, int(a), 0)[3] == 1
        ]
        assert sorted(np.flatnonzero(exact[i].cpu().numpy()).tolist()) == sorted(expected)
        assert expected, f"position {i} should have a winning move"
