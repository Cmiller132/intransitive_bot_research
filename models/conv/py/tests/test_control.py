"""The search control (DESIGN.md item 33): the buffer's admission, eviction
and priority rules, the capped draw probabilities, restart draws through the
environment, and the walk from a collected window to buffer entries. Runs on
the CPU (Triton interpreter)."""

import os
from types import SimpleNamespace

import numpy as np
import torch

from conv.config import Control
from conv.control import REGRET_MAX, Buffer, SearchControl, State, restart_probabilities
from conv.env import END_CLOCK, Env
from conv.planes import N_SQUARES, initial_board

# TRITON_INTERPRET=1 in the environment runs everything on the CPU.
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


def state(seed: int, ply: int = 10) -> State:
    """A distinct restart state per seed: the seed is its plies since a capture."""
    return State(np.array(initial_board(), dtype=np.int8), since_capture=seed, ply=ply, clock=100)


def test_buffer_admits_by_regret_and_evicts_by_priority():
    buffer = Buffer(capacity=3, ema=0.5)
    assert buffer.insert(state(1), 0.2, (0.0, 1.0)) and buffer.insert(state(2), 0.5) and buffer.insert(state(3), 0.1)
    # A played row enters with its game as the first observation, a search node with none.
    assert buffer.entries[state(1).key()].count == 1 and not buffer.entries[state(1).key()].tree
    assert buffer.entries[state(2).key()].count == 0 and buffer.entries[state(2).key()].tree
    # Full: a candidate below the lowest priority is refused, one above replaces it.
    assert not buffer.insert(state(4), 0.05) and len(buffer) == 3
    assert buffer.insert(state(5), 0.3) and state(3).key() not in buffer
    # A candidate already present is one more observation, not a duplicate: its
    # regret takes an EMA step and, from two observations on, its priority is
    # the variance-corrected squared mean residual.
    assert buffer.insert(state(1), 0.8, (0.2, -1.0)) and len(buffer) == 3
    entry = buffer.entries[state(1).key()]
    assert abs(entry.regret - 0.5) < 1e-6 and entry.count == 2
    assert abs(entry.mean_q - 0.1) < 1e-6 and abs(entry.mean_z - 0.0) < 1e-6 and entry.priority() == 0.0
    # Now the lowest priority: the next candidate evicts it, not the 0.3 entry.
    assert buffer.insert(state(6), 0.2) and state(1).key() not in buffer and state(5).key() in buffer
    # Non-finite regrets never enter; out-of-domain ones are clamped.
    assert not buffer.insert(state(7), float("nan"))
    assert buffer.insert(state(8), 9.0) and buffer.entries[state(8).key()].regret == REGRET_MAX
    assert buffer.insert(state(8), -1.0) and buffer.entries[state(8).key()].regret == REGRET_MAX / 2
    # Dropping reports whether the opening was there.
    assert buffer.drop(state(8).key()) and not buffer.drop(state(8).key()) and len(buffer) == 2
    # The state round-trips through the checkpoint form, and an old form loads unobserved.
    again = Buffer(capacity=3, ema=0.5)
    again.load(buffer.state())
    assert again.entries.keys() == buffer.entries.keys()
    fields = ("regret", "mean_q", "mean_z", "count", "r_mean", "r_m2", "tree", "predicted")
    assert all(getattr(again.entries[k], f) == getattr(buffer.entries[k], f) for k in buffer.entries for f in fields)
    old = {k: v for k, v in buffer.state().items() if k in ("board", "since_capture", "ply", "clock", "regret")}
    again.load(old)
    assert all(e.count == 0 and e.priority() == e.regret for e in again.entries.values())


def test_replays_fold_as_one_ema_step_per_game_and_never_resurrect():
    buffer = Buffer(capacity=2, ema=0.5)
    buffer.insert(state(1), 0.8)
    # Three replays: (1-a)^3 of the old regret, the rest toward their mean 0.4.
    assert buffer.replay(state(1).key(), [0.2, 0.4, 0.6], [(0.5, 1.0), (0.5, -1.0), (0.5, 1.0)])
    entry = buffer.entries[state(1).key()]
    assert abs(entry.regret - (0.125 * 0.8 + 0.875 * 0.4)) < 1e-6 and entry.count == 3
    # Outcome variation exceeds the squared mean residual and clamps the estimate to zero.
    assert entry.priority() == 0.0
    # An evicted opening is not brought back by its replays.
    buffer.drop(state(1).key())
    assert not buffer.replay(state(1).key(), [0.9], [(0.0, 1.0)]) and state(1).key() not in buffer


def test_residual_priority_switches_at_two_observations():
    buffer = Buffer(capacity=4, ema=0.5)
    # Two agreeing outcomes at Q=0 have a constant residual and priority one.
    buffer.insert(state(1), 0.7, (0.0, 1.0))
    buffer.replay(state(1).key(), [0.3], [(0.0, 1.0)])
    assert buffer.entries[state(1).key()].priority() == 1.0
    # Two disagreeing outcomes have no persistent residual after correction.
    buffer.insert(state(2), 0.7, (0.0, 1.0))
    buffer.replay(state(2).key(), [0.3], [(0.0, -1.0)])
    assert buffer.entries[state(2).key()].priority() == 0.0
    # Four consistent observations retain priority one.
    buffer.insert(state(3), 0.7, (0.0, -1.0))
    buffer.replay(state(3).key(), [0.3] * 3, [(0.0, -1.0)] * 3)
    entry = buffer.entries[state(3).key()]
    assert entry.count == 4 and abs(entry.priority() - 1.0) < 1e-12


def test_restart_probabilities_cap_the_share_of_any_entry():
    regret = np.array([2.0, 1.0, 0.5, 0.5, 0.25, 0.0])
    # Without the cap the top entry takes almost everything at this temperature.
    raw = restart_probabilities(regret, 0.1, share=1.0)
    assert raw[0] > 0.999 and abs(raw.sum() - 1) < 1e-9 and raw[-1] < 1e-30
    capped = restart_probabilities(regret, 0.1, share=0.25)
    assert abs(capped.sum() - 1) < 1e-9 and capped.max() <= 0.25 + 1e-9
    # The two that exceed the cap sit at it; the rest share the remainder in proportion.
    assert np.allclose(capped[:2], 0.25) and capped[2] == capped[3] < 0.25 and capped[4] < capped[3]
    # Too few entries for the cap to hold: the raw probabilities stay.
    assert np.allclose(restart_probabilities(regret[:2], 0.1, share=0.25), raw[:2] / raw[:2].sum())
    # The run's temperature spreads the draws: 0.5 gives the ratio of squared priorities.
    spread = restart_probabilities(regret[:2], 0.5, share=1.0)
    assert abs(spread[0] / spread[1] - 4.0) < 1e-9


def test_restarts_come_from_the_buffer_with_their_clocks():
    cfg = Control(restart=1.0, temperature=0.1, capacity=8, ema=0.5, warmup=0)
    n = 4
    control = SearchControl(cfg, n, steps=2, device=DEVICE, seed=0, warmup=0)
    env = Env(n, DEVICE, max_plies=1000, clock_min=50, clock_max=200, seed=7)
    # Nothing to draw from: every board restarts from the initial position.
    control.upload()
    control.draw(env)
    assert (env.restart_board == env.start_board[None]).all() and (env.restart_origin == -1).all()
    target = state(11, ply=40)
    control.buffer.insert(target, 0.9)
    control.buffer.insert(state(12, ply=20), 0.0)  # priority 0: never drawn at this temperature
    control.upload()
    control.draw(env)
    assert (env.restart_origin == 0).all() and (env.restart_ply == 40).all()
    # A board that ends its game continues from the stored state with its clock and origin.
    env.since_capture.fill_(env.clock[0].item())  # every clock has run out
    env.clock.fill_(env.clock[0].item())
    env.step(torch.full((n,), 28, dtype=torch.int32, device=DEVICE))  # a quiet opening move
    assert env.done.all() and (env.end_reason == END_CLOCK).all()
    assert (env.board == torch.from_numpy(target.board).to(DEVICE)).all()
    assert (env.ply == 40).all() and (env.since_capture == 11).all() and (env.clock == 100).all()
    assert (env.origin == 0).all() and env.legal.any(-1).all()


def make_rollout(done, origin, tree_score=None, tree_regret=None, tree_board=None, tree_since=None, tree_ply=None):
    T, N = done.shape
    return SimpleNamespace(
        done=done,
        origin=origin,
        tree_score=torch.full((T, N), -torch.inf) if tree_score is None else tree_score,
        tree_regret=torch.zeros(T, N) if tree_regret is None else tree_regret,
        tree_board=torch.zeros(T, N, N_SQUARES, dtype=torch.int8) if tree_board is None else tree_board,
        tree_since=torch.zeros(T, N, dtype=torch.int16) if tree_since is None else tree_since,
        tree_ply=torch.zeros(T, N, dtype=torch.int16) if tree_ply is None else tree_ply,
        clock=torch.full((T, N), 100, dtype=torch.int16),
    )


def make_window(T, N, boards):
    rows = T * N
    return SimpleNamespace(
        iteration=0,
        board=boards,
        since_capture=torch.zeros(rows, dtype=torch.int16),
        ply=torch.arange(rows, dtype=torch.int16),
        clock=torch.full((rows,), 100, dtype=torch.int16),
        regret=torch.zeros(rows),
        rank=torch.zeros(rows),
        played_q=torch.zeros(rows),
        ret=torch.zeros(rows),
    )


def put_state(window, T, N, t, n, s: State) -> None:
    """Row `(t, n)` of `window` holds the state `s`."""
    window.board.view(T, N, N_SQUARES)[t, n] = torch.from_numpy(s.board)
    window.since_capture.view(T, N)[t, n] = s.since_capture
    window.ply.view(T, N)[t, n] = s.ply
    window.clock.view(T, N)[t, n] = s.clock


def test_fresh_candidates_are_max_measured_regret_include_ply_zero_and_wait_for_upload():
    """Fresh games queue one measured-regret candidate in finish order."""
    T, N = 3, 2
    cfg = Control(restart=0.5, temperature=0.1, capacity=8, ema=0.5, warmup=1)
    control = SearchControl(cfg, N, T, DEVICE, seed=0, warmup=1)
    done = torch.zeros(T, N, dtype=torch.bool)
    done[2] = True
    origin = torch.full((T, N), -1, dtype=torch.int32)
    boards = torch.zeros(T * N, N_SQUARES, dtype=torch.int8)
    boards[:, 0] = torch.arange(T * N, dtype=torch.int8) + 1  # distinct rows
    window = make_window(T, N, boards)
    first = state(51, ply=0)
    middle = state(61, ply=7)
    put_state(window, T, N, 0, 0, first)
    put_state(window, T, N, 1, 1, middle)
    window.regret.view(T, N)[:, 0] = torch.tensor([0.9, 0.3, 0.6])
    window.regret.view(T, N)[:, 1] = torch.tensor([0.4, 0.8, 0.1])
    window.played_q.view(T, N)[:, 0] = torch.tensor([0.5, -0.4, 0.3])
    window.played_q.view(T, N)[:, 1] = torch.tensor([0.6, 0.2, 0.0])
    window.ret.view(T, N)[2] = torch.tensor([1.0, -1.0])
    rows = [(window, 0, 2)]

    control.observe(make_rollout(done, origin), torch.full((N,), -1))
    control.finish(1, 2, 1, True, rows)
    control.finish(0, 2, 1, True, rows)
    assert len(control.buffer) == 0 and control.candidates == {"row": 0, "tree": 0}
    assert [candidate[1].key() for candidate in control.pending] == [middle.key(), first.key()]
    control.upload()
    assert list(control.buffer.entries) == [middle.key(), first.key()]
    assert control.buffer.entries[first.key()].state.ply == 0
    assert abs(control.buffer.entries[first.key()].regret - 0.9) < 1e-6
    assert abs(control.buffer.entries[middle.key()].regret - 0.8) < 1e-6
    assert all(entry.count == 1 and not entry.tree for entry in control.buffer.entries.values())
    assert not control.pending
    metrics = control.end_iteration()
    assert metrics["control_warmup"] == 1 and control.warmup_left == 0
    assert metrics["buffer_candidates_row"] == 2 and metrics["buffer_inserted"] == 2
    assert metrics["buffer_candidates_tree"] == metrics["buffer_inserted_tree"] == 0.0
    assert metrics["tree_regret_error"] == 0.0


def test_replays_only_observe_their_opening_and_ply_cap_drops_it():
    T = N = 1
    cfg = Control(restart=1.0, temperature=0.1, capacity=8, ema=0.5, warmup=0)
    control = SearchControl(cfg, N, T, DEVICE, seed=0, warmup=0)
    opening = state(21, ply=7)
    control.buffer.insert(opening, 0.8)
    control.upload()
    control.restarted[0] = True
    done = torch.ones(T, N, dtype=torch.bool)
    origin = torch.full((T, N), -1, dtype=torch.int32)
    window = make_window(T, N, torch.zeros(T * N, N_SQUARES, dtype=torch.int8))
    put_state(window, T, N, 0, 0, opening)
    window.regret[0] = 0.4
    window.played_q[0] = 0.6
    window.ret[0] = -0.15

    control.observe(make_rollout(done, origin), torch.tensor([-1]))
    control.finish(0, 0, 0, True, [(window, 0, 0)])
    assert not control.pending and control.buffer.entries[opening.key()].regret == 0.8
    control.upload()
    entry = control.buffer.entries[opening.key()]
    assert abs(entry.regret - 0.6) < 1e-6 and entry.count == 1
    assert abs(entry.mean_q - 0.6) < 1e-6 and abs(entry.mean_z + 0.15) < 1e-6

    control.restarted[0] = True
    control.observe(make_rollout(done, origin), torch.tensor([-1]))
    control.finish(0, 0, None, True, [(window, 0, 0)])
    assert opening.key() not in control.buffer
    assert control.end_iteration()["buffer_dropped"] == 1


def test_upload_folds_replay_before_same_window_candidate_can_evict_it():
    T, N = 1, 2
    cfg = Control(restart=1.0, temperature=0.1, capacity=1, ema=0.5, warmup=0)
    control = SearchControl(cfg, N, T, DEVICE, seed=0, warmup=0)
    opening = state(71)
    candidate = state(72, ply=0)
    control.buffer.insert(opening, 0.2, (1.0, -1.0))
    control.upload()
    control.restarted[1] = True
    done = torch.ones(T, N, dtype=torch.bool)
    origin = torch.full((T, N), -1, dtype=torch.int32)
    window = make_window(T, N, torch.zeros(T * N, N_SQUARES, dtype=torch.int8))
    put_state(window, T, N, 0, 0, candidate)
    put_state(window, T, N, 0, 1, opening)
    window.regret[:] = torch.tensor([0.5, 0.2])
    window.played_q[:] = torch.tensor([0.0, 1.0])
    window.ret[:] = torch.tensor([1.0, -1.0])

    control.observe(make_rollout(done, origin), torch.full((N,), -1))
    control.finish(0, 0, 1, True, [(window, 0, 0)])
    control.finish(1, 0, 0, True, [(window, 0, 0)])
    assert control.buffer.entries[opening.key()].count == 1 and candidate.key() not in control.buffer
    control.upload()
    assert opening.key() in control.buffer and candidate.key() not in control.buffer
    assert control.buffer.entries[opening.key()].count == 2
    metrics = control.end_iteration()
    assert metrics["buffer_candidates_row"] == 1 and metrics["buffer_inserted"] == 0
    assert metrics["buffer_lost"] == 0


def test_replays_of_one_opening_fold_as_one_step_per_game():
    T, N = 2, 2
    cfg = Control(restart=1.0, temperature=0.1, capacity=8, ema=0.5, warmup=0)
    control = SearchControl(cfg, N, T, DEVICE, seed=0, warmup=0)
    opening = state(41)
    control.buffer.insert(opening, 0.8)  # a search node: predicted 0.8
    control.upload()
    control.restarted[:] = True
    done = torch.zeros(T, N, dtype=torch.bool)
    done[1] = True
    control.observe(make_rollout(done, torch.full((T, N), -1, dtype=torch.int32)), torch.tensor([-1, -1]))
    window = make_window(T, N, torch.zeros(T * N, N_SQUARES, dtype=torch.int8))
    for n in range(N):
        put_state(window, T, N, 0, n, opening)
    window.regret.view(T, N)[0] = torch.tensor([0.2, 0.6])
    window.played_q.view(T, N)[0] = torch.tensor([0.1, 0.1])
    window.ret.view(T, N)[1] = torch.tensor([1.0, 1.0])  # both games won by the second mover
    control.finish(0, 1, 0, True, [(window, 0, 1)])
    control.finish(1, 1, 1, True, [(window, 0, 1)])
    control.upload()
    entry = control.buffer.entries[opening.key()]
    # Two steps of `ema` from 0.8 toward the replays' mean 0.4: 0.25 * 0.8 + 0.75 * 0.4.
    assert abs(entry.regret - 0.5) < 1e-6
    # Both games were lost from the opening's view (-1 one ply before the end) while
    # it was played at Q 0.1: two observations, so the priority is the squared bias.
    assert entry.count == 2 and abs(entry.mean_z + 1.0) < 1e-6 and abs(entry.priority() - 1.1**2) < 1e-6
    metrics = control.end_iteration()
    assert metrics["restart_distinct"] == 1 and metrics["restart_max"] == 2 and metrics["restart_frac"] == 1.0
    assert abs(metrics["tree_regret_error"] - 0.4) < 1e-6
    assert abs(metrics["replay_priority_mean"] - 1.21) < 1e-6
    assert metrics["buffer_scored_frac"] == 1.0 and metrics["replay_q_drift"] == 0.0


def test_replays_of_an_evicted_opening_are_lost_not_resurrected():
    """A restart drawn under one copy of the buffer starts a game that ends
    after its opening was evicted: the replay's observation reaches nothing
    (the entry at the opening's old index is untouched) and is counted lost."""
    T, N = 2, 1
    cfg = Control(restart=1.0, temperature=0.1, capacity=2, ema=0.5, warmup=0)
    control = SearchControl(cfg, N, T, DEVICE, seed=0, warmup=0)
    first, second, third = state(31), state(32), state(33)
    control.buffer.insert(first, 0.9)
    control.buffer.insert(second, 0.3)  # a thousandth of the mass at this temperature
    control.upload()
    env = Env(N, DEVICE, max_plies=1000, clock_min=50, clock_max=200, seed=1)
    control.draw(env)
    drawn = int(env.restart_origin.item())
    assert drawn == 0  # `first`
    # Before that game starts, `first` is measured low and evicted; `second` takes its index.
    control.buffer.replay(first.key(), [0.0, 0.0], [])  # 0.225 now, below `second`
    assert control.buffer.insert(third, 0.95) and first.key() not in control.buffer
    control.upload()
    assert list(control.buffer.entries)[drawn] == second.key()
    # The board's game ends at step 0; the replay from `first` starts at step 1.
    done = torch.tensor([[True], [False]])
    control.observe(make_rollout(done, torch.tensor([[-1], [drawn]], dtype=torch.int32)), torch.tensor([drawn]))
    assert control.restarted[0]
    # The replay ends at step 0 of the next window with regret 0.5 at its first row.
    window = make_window(T, N, torch.zeros(T * N, N_SQUARES, dtype=torch.int8))
    put_state(window, T, N, 0, 0, first)
    window.regret[0] = 0.5
    control.observe(make_rollout(done, torch.full((T, N), -1, dtype=torch.int32)), torch.tensor([-1]))
    control.finish(0, 0, 0, True, [(window, 0, 0)])
    control.upload()
    assert first.key() not in control.buffer and second.key() in control.buffer
    assert abs(control.buffer.entries[third.key()].regret - 0.95) < 1e-6
    assert control.end_iteration()["buffer_lost"] == 1


def test_active_gumbel_choice_admits_tree_without_observation_and_played_with_one():
    T, N = 3, 2
    cfg = Control(restart=0.5, temperature=0.5, capacity=8, ema=0.5, warmup=0)
    control = SearchControl(cfg, N, T, DEVICE, seed=7, warmup=0)
    control.heads_active = True
    done = torch.zeros(T, N, dtype=torch.bool)
    done[-1] = True
    origin = torch.full((T, N), -1, dtype=torch.int32)
    boards = torch.zeros(T * N, N_SQUARES, dtype=torch.int8)
    boards[:, 0] = torch.arange(T * N, dtype=torch.int8) + 1
    window = make_window(T, N, boards)
    window.regret.view(T, N)[:] = torch.tensor([[0.1, 0.2], [0.3, 0.8], [0.4, 0.1]])
    window.rank.view(T, N)[:] = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    window.ret.view(T, N)[-1] = torch.tensor([1.0, 1.0])
    tree_board = torch.zeros(T, N, N_SQUARES, dtype=torch.int8)
    tree_board[1, 0, 0] = 40
    tree_score = torch.full((T, N), -torch.inf)
    tree_score[1, 0] = 100.0
    tree_regret = torch.zeros(T, N)
    tree_regret[1, 0] = 0.6
    tree_ply = torch.zeros(T, N, dtype=torch.int16)
    tree_ply[1, 0] = 12
    rollout = make_rollout(done, origin, tree_score, tree_regret, tree_board, tree_ply=tree_ply)

    # Reproduce the played-row Gumbels: game 0 consumes the first three, game 1 the next three.
    expected_rng = torch.Generator(device=DEVICE).manual_seed(7)
    torch.rand(T, generator=expected_rng, device=DEVICE)
    u = torch.rand(T, generator=expected_rng, device=DEVICE).clamp_(1e-12, 1.0)
    expected_row = int((-(-u.log()).log()).argmax())

    control.observe(rollout, torch.full((N,), -1))
    control.finish(0, T - 1, 1, True, [(window, 0, T - 1)])
    control.finish(1, T - 1, 1, True, [(window, 0, T - 1)])
    assert [candidate[0] for candidate in control.pending] == ["tree", "row"]
    assert control.pending[1][1].key() == state_at_row(window, T, N, expected_row, 1).key()
    control.upload()
    entries = list(control.buffer.entries.values())
    tree = next(entry for entry in entries if entry.tree)
    played = next(entry for entry in entries if not entry.tree)
    assert tree.count == 0 and abs(tree.regret - 0.6) < 1e-6 and tree.state.ply == 12
    assert played.count == 1


def state_at_row(window, T, N, t, n):
    return State(
        window.board.view(T, N, N_SQUARES)[t, n].numpy(),
        int(window.since_capture.view(T, N)[t, n]),
        int(window.ply.view(T, N)[t, n]),
        int(window.clock.view(T, N)[t, n]),
    )


def test_readiness_requires_two_positive_windows_and_state_round_trips():
    cfg = Control(capacity=8, warmup=0)
    control = SearchControl(cfg, 1, 1, DEVICE, seed=0, warmup=0)
    control.buffer.insert(state(81), 0.5, (0.0, 1.0))

    def positive_window():
        control.lift_ranks.extend([0.0, 2.0])
        control.lift_labels.extend([0.0, 1.0])
        return control.end_iteration()

    first = positive_window()
    assert first["rank_lift"] > 0 and first["heads_active"] == 0 and not control.heads_active
    second = positive_window()
    assert second["rank_lift"] > 0 and second["heads_active"] == 1 and control.heads_active
    control.lift_ranks.extend([2.0, 0.0])
    control.lift_labels.extend([0.0, 1.0])
    third = control.end_iteration()
    assert third["rank_lift"] < 0 and third["heads_active"] == 1
    assert third["buffer_board_distinct"] == 1
    assert third["buffer_priority_unscored_mean"] == 0.5

    saved = control.state()
    again = SearchControl(cfg, 1, 1, DEVICE, seed=1, warmup=3)
    again.load(saved)
    assert again.heads_active and again.rank_lift == control.rank_lift and again.warmup_left == 0
    entry = next(iter(again.buffer.entries.values()))
    original = next(iter(control.buffer.entries.values()))
    assert (entry.r_mean, entry.r_m2, entry.count) == (original.r_mean, original.r_m2, original.count)
