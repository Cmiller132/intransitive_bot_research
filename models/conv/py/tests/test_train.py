"""One tiny collection and learner iteration end to end in both self-play
phases: labels have the right shapes and masks, the tactics labels survive the
augmentation, losses are finite and the checkpoint round-trips. Runs on the CPU
(Triton interpreter) without a GPU."""

import math
import os

import torch

from conv.config import Config, Learn, Net, Search
from conv.distil import Distiller
from conv.model import ConvNet, build, load_checkpoint, save_checkpoint
from conv.paths import workspace_root
from conv.planes import N_ACTIONS, N_SQUARES, initial_board, tactics_reference
from conv.replay import Replay, Rollout, build_window, lambda_returns, plies_to_end
from conv.search import SLOTS, spread, unpack_tactics
from conv.train import Actor, Augment, Learner, conversion_metrics, lr_multiplier, student_evaluator

# TRITON_INTERPRET=1 in the environment runs everything on the CPU.
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


def tiny_config(envs=4, steps=6) -> Config:
    cfg = Config(run="test", seed=3, checkpoint_every=1)
    cfg.net = Net(
        width=32,
        pairs=2,
        heads=2,
        ffn=64,
        smolgen_every=2,
        smolgen_first=1,
        smolgen_dim=4,
        smolgen_hidden=16,
        smolgen_latent=8,
        atoms=11,
        head_dim=8,
        value_hidden=16,
    )
    cfg.search = Search(
        sims=6, candidates=3, cheap_sims=3, cheap_candidates=2, full_fraction=0.5, reuse_nodes=6, temperature_plies=2
    )
    cfg.learn = Learn(
        envs=envs,
        steps=steps,
        window_min=2,
        batches=envs * steps // 8,
        batch=8,
        compile=False,
        graphs=False,
        occupancy_horizon=2,
    )
    cfg.rules.max_plies = 40
    cfg.rules.clock_min, cfg.rules.clock_max = 6, 12
    return cfg


def test_lambda_returns_truncation_and_plies_to_end():
    reward = torch.tensor([[0.0], [0.0], [1.0]])
    done = torch.tensor([[False], [False], [True]])
    none = torch.zeros_like(done)
    value = torch.zeros(4, 1)
    ret = lambda_returns(reward, done, none, value, lam=1.0)
    assert ret.flatten().tolist() == [1.0, -1.0, 1.0]
    # A step stopped by the ply cap keeps its own search value instead of its reward.
    value = torch.tensor([[0.0], [0.0], [0.25], [0.0]])
    ret = lambda_returns(torch.zeros(3, 1), done, done, value, lam=1.0)
    assert ret.flatten().tolist() == [0.25, -0.25, 0.25]
    terminal = torch.tensor([[False], [False], [True]])
    assert plies_to_end(done, terminal).flatten().tolist() == [3, 2, 1]
    assert plies_to_end(done, torch.zeros_like(done)).flatten().tolist() == [-1, -1, -1]


def test_final_material_labels_wins_and_clock_draws_not_ply_caps():
    from conv.env import END_CLOCK, END_MAX_PLIES, END_WIN

    cfg = tiny_config(envs=3, steps=3)
    rollout = Rollout.allocate(cfg.learn.steps, cfg.learn.envs, cfg.search.candidates, "cpu")
    rollout.board.copy_(torch.tensor(initial_board(), dtype=torch.int8)[None, None])
    rollout.action.fill_(28)
    rollout.done[-1].fill_(True)
    rollout.end_reason[-1] = torch.tensor([END_WIN, END_CLOCK, END_MAX_PLIES], dtype=torch.uint8)
    rollout.ply[:, 2] = torch.arange(cfg.rules.max_plies - 3, cfg.rules.max_plies, dtype=torch.int16)
    window = build_window(rollout, cfg, iteration=0)
    material_ok = window.material_ok.view(cfg.learn.steps, cfg.learn.envs)
    tte_ok = window.plies_to_end.view(cfg.learn.steps, cfg.learn.envs) > 0
    assert material_ok[:, :2].all() and not material_ok[:, 2].any()
    assert tte_ok[:, 0].all() and not tte_ok[:, 1:].any()
    assert (window.material[window.material_ok] > 0).all()


def test_student_evaluator_draw_shape_follows_contempt_source():
    cfg = tiny_config(envs=2, steps=1)
    actor = Actor(cfg, DEVICE)
    net = ConvNet(cfg.net).to(DEVICE).eval()
    args = (actor.env.board, actor.env.since_capture, actor.env.ply, actor.env.clock)
    wdl_draw = student_evaluator(net, contempt_source="wdl")(*args)[3]
    q_draw = student_evaluator(net, contempt_source="q")(*args)[3]
    assert wdl_draw.shape == (cfg.learn.envs,)
    assert q_draw.shape == (cfg.learn.envs, N_ACTIONS)


def test_reply_labels_are_the_next_move_in_the_movers_frame():
    """`flip_actions` carries a move across the frame change between plies
    (DESIGN.md item 32): from and target squares map by the anti-diagonal
    mirror, and the map is an involution. In a window, a row's reply is the
    next row's move of the same game, flipped, and nothing after a reset."""
    from conv.planes import ANTI_PERM, action_targets, flip_actions

    actions = torch.arange(N_ACTIONS)
    targets = action_targets()
    on_board = targets >= 0
    flipped = flip_actions(actions)
    assert torch.equal(flip_actions(flipped), actions)
    anti = torch.tensor(ANTI_PERM)
    assert torch.equal(flipped[on_board] % N_SQUARES, anti[actions[on_board] % N_SQUARES])
    assert torch.equal(targets[flipped[on_board]], anti[targets[on_board]])

    cfg = tiny_config(envs=3, steps=12)
    actor = Actor(cfg, DEVICE)
    net = ConvNet(cfg.net).to(DEVICE).eval()
    with torch.no_grad():
        rollout = actor.collect(student_evaluator(net))
    window = build_window(rollout, cfg, iteration=0)
    T, N = cfg.learn.steps, cfg.learn.envs
    reply, ok = window.reply.view(T, N), window.reply_ok.view(T, N)
    done, action = rollout.done.cpu(), rollout.action.cpu()
    assert not ok[-1].any() and not ok[:-1][done[:-1]].any()
    assert ok[:-1][~done[:-1]].all()
    assert torch.equal(flip_actions(reply[:-1][ok[:-1]]), action[1:][ok[:-1]])


def test_augmented_tactics_match_the_reference():
    """The diagonal flip and the type cycle permute the labels as they permute
    the board, so the reference oracle on the augmented board agrees."""
    cfg = tiny_config(envs=2, steps=2)
    actor = Actor(cfg, DEVICE)
    net = ConvNet(cfg.net).to(DEVICE).eval()
    with torch.no_grad():
        rollout = actor.collect(student_evaluator(net))
    window = build_window(rollout, cfg, iteration=0)
    augment = Augment(DEVICE)
    rows = torch.arange(len(window))
    flip = (rows % 2 == 1).to(DEVICE)
    cycle = (rows % 3).to(DEVICE)
    boards = augment.board(window.board.to(DEVICE), flip, cycle)
    moves = augment.actions(window.moves.to(DEVICE).long(), flip)
    tactics = unpack_tactics(spread(moves, window.tactics.to(DEVICE)))
    for row in range(len(window)):
        expected = tactics_reference(boards[row].tolist(), int(window.since_capture[row]), int(window.clock[row]))
        for channel in range(3):
            assert [bool(x) for x in tactics[row, :, channel]] == expected[channel], (row, channel)


def test_collection_labels_learner_step_and_checkpoint(tmp_path):
    cfg = tiny_config()
    net = ConvNet(cfg.net).to(DEVICE)
    ema = ConvNet(cfg.net).to(DEVICE)
    ema.load_state_dict(net.state_dict())
    ema.eval()
    actor = Actor(cfg, DEVICE)
    with torch.no_grad():
        rollout = actor.collect(student_evaluator(ema))
    window = build_window(rollout, cfg, iteration=0)
    rows = cfg.learn.envs * cfg.learn.steps
    assert window.board.shape == (rows, N_SQUARES) and window.target.shape == (rows, SLOTS)
    assert window.tactics.shape == (rows, SLOTS) and window.tactics.dtype == torch.uint8
    assert window.moves.shape == (rows, SLOTS) and window.moves.dtype == torch.int16
    assert window.target_ok.dtype == torch.bool and window.plies_to_end.min() >= -1
    assert (window.clock >= cfg.rules.clock_min).all() and (window.clock <= cfg.rules.clock_max).all()
    assert window.material_ok.dtype == torch.bool and (window.material[window.material_ok] >= 0).all()
    assert torch.isfinite(window.ret).all() and window.ret.abs().max() <= 1.0
    assert window.candidate_visited[window.target_ok].any()
    assert rollout.tree_score.shape == (cfg.learn.steps, cfg.learn.envs)
    assert rollout.tree_board.shape == (cfg.learn.steps, cfg.learn.envs, N_SQUARES)
    assert torch.isfinite(rollout.tree_score).any()
    metrics = conversion_metrics(rollout)
    assert 0.0 <= metrics["win_end_frac"] <= 1.0

    learner = Learner(cfg, net, ema, DEVICE)
    replay = Replay(cfg.learn, str(tmp_path / "run"), seed=0)
    replay.add(window)
    stats = learner.train(replay)
    assert stats["batches"] == cfg.learn.batches
    assert all(abs(v) < 1e6 for k, v in stats.items() if k != "batches")
    assert stats["policy"] > 0 and stats["q"] > 0 and stats["value"] > 0 and stats["tactics"] > 0
    assert replay.path(0).is_file() and replay.restore(0) == 1

    path = str(tmp_path / "ckpt.pt")
    from conv.train import load_optimizer_state, optimizer_state

    save_checkpoint(path, net, ema, optimizer_state(learner.opt, net), 0, cfg)
    ck = load_checkpoint(path)
    again = build(cfg, ck, DEVICE, "model")
    for a, b in zip(again.state_dict().values(), net.state_dict().values(), strict=True):
        assert torch.equal(a.cpu(), b.cpu())
    fresh = Learner(cfg, again, build(cfg, ck, DEVICE, "ema"), DEVICE)
    load_optimizer_state(fresh.opt, again, ck["optimizer"], DEVICE)
    assert len(fresh.opt.state) == len(learner.opt.state)


def test_distillation_trains_toward_the_sq_reference():
    """The reference plays, the student matches its policy and Q on every position."""
    cfg = tiny_config()
    cfg.distil.envs, cfg.distil.steps = 4, 3
    cfg.distil.teacher = "sq"
    cfg.distil.ckpt = str(workspace_root() / "weights/sq_g128.pt")
    net = ConvNet(cfg.net).to(DEVICE)
    ema = ConvNet(cfg.net).to(DEVICE)
    ema.load_state_dict(net.state_dict())
    distiller = Distiller(cfg, net, ema, DEVICE)
    before = [p.clone() for p in net.parameters()]
    for _ in range(cfg.distil.steps):
        distiller.step()
    metrics = distiller.metrics()
    assert all(abs(v) < 1e6 for v in metrics.values())
    assert metrics["kl"] > 0 and metrics["teacher_entropy"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters(), strict=True))
    assert distiller.env.ply.max().item() >= cfg.distil.steps or distiller.env.done.any()


def test_conv_distillation_uses_full_q_v_and_wdl_distributions(tmp_path):
    cfg = tiny_config()
    cfg.distil.envs, cfg.distil.steps = 2, 1
    teacher = ConvNet(cfg.net)
    teacher_ema = ConvNet(cfg.net)
    teacher_ema.load_state_dict(teacher.state_dict())
    path = str(tmp_path / "teacher.pt")
    save_checkpoint(path, teacher, teacher_ema, None, 0, cfg)
    cfg.distil.teacher, cfg.distil.ckpt = "conv", path
    net, ema = ConvNet(cfg.net).to(DEVICE), ConvNet(cfg.net).to(DEVICE)
    ema.load_state_dict(net.state_dict())
    distiller = Distiller(cfg, net, ema, DEVICE)
    distiller.step()
    metrics = distiller.metrics()
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["q"] > 0 and metrics["value"] > 0 and metrics["wdl"] > 0


def test_learner_guards_warmup_and_rollback(tmp_path):
    """The fan-in groups, warm-up, iteration schedule, persistent rollback
    factor, outlier skip and in-place snapshot restore (DESIGN.md item 25)."""
    cfg = tiny_config()
    cfg.learn.warmup_steps, cfg.learn.conv_lr_scale, cfg.learn.skip_norm = 4, 0.5, 100.0
    cfg.learn.lr_hold_iters, cfg.learn.lr_decay_iters, cfg.learn.lr_min = 2, 4, cfg.learn.lr / 5
    net = ConvNet(cfg.net).to(DEVICE)
    ema = ConvNet(cfg.net).to(DEVICE)
    learner = Learner(cfg, net, ema, DEVICE)
    scales = sorted(g["lr_scale"] for g in learner.opt.param_groups)
    assert scales == [0.5, 1.0, 1.0]
    conv_group = next(g for g in learner.opt.param_groups if g["lr_scale"] == 0.5)
    assert all(p.ndim == 4 for p in conv_group["params"]) and len(conv_group["params"]) == 2 * cfg.net.pairs + 1
    learner.steps = 1
    learner.set_lr()
    assert abs(conv_group["lr"] - cfg.learn.lr * 0.5 * 2 / 4) < 1e-12
    learner.steps = 99
    learner.iteration = 2
    learner.set_lr()
    assert abs(conv_group["lr"] - cfg.learn.lr * 0.5) < 1e-12
    assert lr_multiplier(cfg.learn, 2) == 1.0
    assert abs(lr_multiplier(cfg.learn, 3) - 0.6) < 1e-12
    assert abs(lr_multiplier(cfg.learn, 4) - 0.2) < 1e-12
    learner.iteration, learner.rollback_factor = 3, 0.5
    learner.set_lr()
    assert abs(conv_group["lr"] - cfg.learn.lr * 0.6 * 0.5 * 0.5) < 1e-12

    for p in learner.params:
        p.grad = torch.full_like(p, 2**-10)
    total, skipped = learner.clip_grads()
    assert skipped.item() == 0.0 and total.item() < cfg.learn.grad_clip
    assert learner.params[0].grad.abs().max().item() == 2**-10
    for p in learner.params:
        p.grad = torch.full_like(p, 10.0)
    total, skipped = learner.clip_grads()
    assert skipped.item() == 1.0 and total.item() > cfg.learn.skip_norm
    assert all(p.grad.abs().max().item() == 0.0 for p in learner.params)

    before = [p.detach().clone() for p in learner.params]
    snap = learner.snapshot()
    with torch.no_grad():
        for p in learner.params:
            p.add_(1.0)
    learner.restore(snap)
    assert all(torch.equal(a, b) for a, b in zip(before, learner.params, strict=True))
    path = str(tmp_path / "rollback.pt")
    save_checkpoint(path, net, ema, None, 0, cfg, {"rollback_factor": learner.rollback_factor})
    assert load_checkpoint(path)["rollback_factor"] == 0.5


def test_outcome_labels_back_fill_earlier_windows(tmp_path):
    """Outcomes (DESIGN.md item 31): a won game labels its rows +1 / -1 back
    from the winner's last move, a clock draw labels 0, a game cut by the ply
    cap stays unlabelled, and a game that began in an earlier window labels
    that window too."""
    from types import SimpleNamespace

    from conv.env import END_CLOCK, END_MAX_PLIES, END_NONE, END_WIN
    from conv.replay import Outcomes

    T, N = 3, 3

    def window(iteration):
        rows = T * N
        return SimpleNamespace(
            iteration=iteration,
            outcome=torch.zeros(rows, dtype=torch.int8),
            outcome_ok=torch.zeros(rows, dtype=torch.bool),
            action=torch.zeros(rows, dtype=torch.int16),
            reply=torch.zeros(rows, dtype=torch.int16),
            reply_ok=torch.zeros(rows, dtype=torch.bool),
            played_q=torch.zeros(rows),
            ret=torch.zeros(rows),
            regret=torch.zeros(rows),
            regret_ok=torch.zeros(rows, dtype=torch.bool),
        )

    w0, w1 = window(0), window(1)
    replay = SimpleNamespace(find=lambda it: {0: w0, 1: w1}.get(it))
    outcomes = Outcomes(N, T)
    # Env 0: a win at step 1 of window 0, then a game running into window 1 that draws at its step 0.
    # Env 1: one long game, cut by the ply cap at step 2 of window 1.
    # Env 2: a game from window 0 won at step 2 of window 1.
    reason0 = torch.full((T, N), END_NONE, dtype=torch.uint8)
    reason0[1, 0] = END_WIN
    reason1 = torch.full((T, N), END_NONE, dtype=torch.uint8)
    reason1[0, 0], reason1[2, 1], reason1[2, 2] = END_CLOCK, END_MAX_PLIES, END_WIN
    assert outcomes.label(reason0 != END_NONE, reason0, w0, replay) == set()
    assert outcomes.label(reason1 != END_NONE, reason1, w1, replay) == {0}
    o0, k0 = w0.outcome.view(T, N), w0.outcome_ok.view(T, N)
    o1, k1 = w1.outcome.view(T, N), w1.outcome_ok.view(T, N)
    assert o0[:, 0].tolist() == [-1, 1, 0] and k0[:, 0].tolist() == [True, True, True]
    assert o1[0, 0] == 0 and k1[0, 0] and not k1[1:, 0].any()
    assert not k0[:, 1].any() and not k1[:, 1].any()
    assert o0[:, 2].tolist() == [-1, 1, -1] and o1[:, 2].tolist() == [1, -1, 1] and k0[:, 2].all() and k1[:, 2].all()


def test_bias_excess_labels_use_parity_q_return_suffix_and_clock_penalty():
    from types import SimpleNamespace

    from conv.env import END_CLOCK, END_NONE, END_WIN
    from conv.replay import Outcomes

    T, N = 3, 2
    rows = T * N
    window = SimpleNamespace(
        iteration=0,
        outcome=torch.zeros(rows, dtype=torch.int8),
        outcome_ok=torch.zeros(rows, dtype=torch.bool),
        action=torch.zeros(rows, dtype=torch.int16),
        reply=torch.zeros(rows, dtype=torch.int16),
        reply_ok=torch.zeros(rows, dtype=torch.bool),
        played_q=torch.tensor([[0.5, 0.5], [-0.25, -0.25], [0.75, 0.75]]).flatten(),
        ret=torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, -0.2]]).flatten(),
        regret=torch.zeros(rows),
        regret_ok=torch.zeros(rows, dtype=torch.bool),
    )
    reason = torch.full((T, N), END_NONE, dtype=torch.uint8)
    reason[-1] = torch.tensor([END_WIN, END_CLOCK], dtype=torch.uint8)
    replay = SimpleNamespace(find=lambda _: None)
    Outcomes(N, T).label(reason != END_NONE, reason, window, replay)

    q = window.played_q.view(T, N)
    expected = torch.empty_like(q)
    for n, terminal in enumerate((1.0, -0.2)):
        z_q = torch.tensor([terminal, -terminal, terminal])
        terms = 2 * q[:, n] * (q[:, n] - z_q)
        expected[:, n] = torch.stack([terms.mean(), terms[1:].mean(), terms[2]])
    assert torch.allclose(window.regret.view(T, N), expected)
    assert window.regret_ok.all()
    assert window.outcome.view(T, N)[:, 0].tolist() == [1, -1, 1]
    assert window.outcome.view(T, N)[:, 1].tolist() == [0, 0, 0]


def test_old_regret_version_invalidates_only_regret_labels(tmp_path):
    import io

    from compression import zstd

    from conv.replay import REGRET_VERSION

    cfg = tiny_config(envs=1, steps=1)
    rollout = Rollout.allocate(1, 1, cfg.search.candidates, "cpu")
    rollout.board.copy_(torch.tensor(initial_board(), dtype=torch.int8))
    window = build_window(rollout, cfg, iteration=0)
    window.regret.fill_(0.75)
    window.regret_ok.fill_(True)
    replay = Replay(cfg.learn, str(tmp_path), seed=0)
    replay.persist(window)
    path = replay.path(0)
    payload = torch.load(io.BytesIO(zstd.decompress(path.read_bytes())), map_location="cpu", weights_only=True)
    assert payload.pop("regret_version") == REGRET_VERSION
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path.write_bytes(zstd.compress(buffer.getvalue(), replay.LEVEL))

    loaded = replay.load(0)
    assert loaded is not None and not loaded.regret_ok.any()
    assert torch.equal(loaded.board, window.board) and torch.equal(loaded.outcome_ok, window.outcome_ok)


def test_ranking_loss_excludes_dummy_when_labelled_and_is_zero_when_empty():
    from conv.train import ranking_loss

    gamma = torch.tensor([0.2, -0.4, 2.0])
    label = torch.tensor([-0.5, 0.7, 9.0])
    ok = torch.tensor([True, True, False])
    expected = torch.logsumexp(gamma[ok], 0) - torch.logsumexp(gamma[ok] + label[ok], 0)
    assert torch.allclose(ranking_loss(gamma, label, ok), expected)
    assert ranking_loss(gamma, label, torch.zeros_like(ok)).item() == 0.0


def test_bias_resume_removes_regret_weights_and_moments(tmp_path):
    from conv.model import share_fresh_heads
    from conv.train import load_optimizer_state, optimizer_state, regret_parameter_indices
    from scripts.derive_bias_resume import main as derive

    cfg = tiny_config(envs=1, steps=1)
    net, ema = ConvNet(cfg.net), ConvNet(cfg.net)
    ema.load_state_dict(net.state_dict())
    learner = Learner(cfg, net, ema, "cpu")
    for p in net.parameters():
        learner.opt.state[p] = {
            "step": torch.tensor(3.0),
            "exp_avg": torch.ones_like(p),
            "exp_avg_sq": torch.ones_like(p),
        }
    src, dst, backup = (str(tmp_path / name) for name in ("src.pt", "dst.pt", "backup.pt"))
    save_checkpoint(src, net, ema, optimizer_state(learner.opt, net), 17, cfg)
    derive(["derive_bias_resume.py", src, dst, backup])
    assert (tmp_path / "src.pt").read_bytes() == (tmp_path / "backup.pt").read_bytes()

    ck = load_checkpoint(dst)
    indices = regret_parameter_indices(net)
    assert ck["iteration"] == 17 and all(index not in ck["optimizer"]["state"] for index in indices)
    assert ck["control"]["warmup_left"] == 0 and not ck["control"]["heads_active"]
    fresh_net, fresh_ema = build(cfg, ck, "cpu", "model"), build(cfg, ck, "cpu", "ema")
    assert fresh_net.fresh and all(key.startswith("regret.") for key in fresh_net.fresh)
    share_fresh_heads(fresh_net, fresh_ema)
    for key in fresh_net.fresh:
        assert torch.equal(fresh_net.state_dict()[key], fresh_ema.state_dict()[key])
    fresh_learner = Learner(cfg, fresh_net, fresh_ema, "cpu")
    load_optimizer_state(fresh_learner.opt, fresh_net, ck["optimizer"], "cpu")
    params = list(fresh_net.parameters())
    assert all(fresh_learner.opt.state[params[index]]["exp_avg"].count_nonzero() == 0 for index in indices)


def test_search_control_labels_train_and_resume(tmp_path):
    """A collection with the search control: regret labels are suffix means
    of played-Q calibration excess, the buffer fills from
    finished games, the regret and reply losses are finite, and a checkpoint
    from before the heads resumes with fresh heads shared by the EMA, zero
    optimizer moments for them and the buffer restored."""
    from conv.control import SearchControl
    from conv.model import share_fresh_heads
    from conv.replay import Outcomes
    from conv.train import load_optimizer_state, optimizer_state

    cfg = tiny_config(steps=48)
    cfg.control.capacity, cfg.control.warmup = 16, 0
    net = ConvNet(cfg.net).to(DEVICE)
    ema = ConvNet(cfg.net).to(DEVICE)
    ema.load_state_dict(net.state_dict())
    ema.eval()
    actor = Actor(cfg, DEVICE)
    control = SearchControl(cfg.control, cfg.learn.envs, cfg.learn.steps, DEVICE, seed=0, warmup=0)
    actor.control = control
    with torch.no_grad():
        rollout = actor.collect(student_evaluator(ema))
    window = build_window(rollout, cfg, iteration=0)
    replay = Replay(cfg.learn, str(tmp_path / "run"), seed=0)
    control.observe(rollout, actor.env.origin)
    Outcomes(cfg.learn.envs, cfg.learn.steps).label(rollout.done, rollout.end_reason, window, replay, control)
    T, N = cfg.learn.steps, cfg.learn.envs
    ok, regret = window.regret_ok.view(T, N), window.regret.view(T, N)
    q = window.played_q.view(T, N)
    assert ok.any() and (window.outcome_ok == window.regret_ok).all()
    for n in range(N):
        rows = torch.nonzero(ok[:, n]).squeeze(1).tolist()
        if not rows:
            continue
        # Within one game the label is the suffix mean of Q-domain calibration excess.
        t = rows[0]
        end = t
        while end + 1 < T and ok[end + 1, n] and not rollout.done[end, n]:
            end += 1
        distance = torch.arange(end - t, -1, -1)
        z_q = window.ret.view(T, N)[end, n] * torch.where(distance % 2 == 0, 1.0, -1.0)
        excess = 2 * q[t : end + 1, n] * (q[t : end + 1, n] - z_q)
        assert abs(regret[t, n].item() - excess.mean().item()) < 1e-4
    assert len(control.buffer) == 0 and control.pending
    control.upload()
    assert len(control.buffer) > 0 and not control.pending
    control.draw(actor.env)
    assert (actor.env.restart_origin >= 0).any()
    replay.add(window)
    learner = Learner(cfg, net, ema, DEVICE)
    stats = learner.train(replay)
    assert all(math.isfinite(stats[k]) for k in ("regret", "rank", "reply")) and stats["reply"] > 0
    # No game of the first window started from the buffer; the next window's rows record their restarts.
    assert not window.replayed.any() and stats["value_fresh"] == stats["value"] and stats["value_replay"] == 0
    with torch.no_grad():
        rollout = actor.collect(student_evaluator(ema))
    again = build_window(rollout, cfg, iteration=1)
    assert again.replayed.any() and torch.equal(again.replayed, (rollout.origin >= 0).flatten().cpu())
    replay.add(again)
    stats = learner.train(replay)
    assert stats["value_replay"] > 0 and stats["value_fresh"] > 0

    # A checkpoint from before the heads resumes with them fresh and shared.
    old = tiny_config(steps=48)
    old.net.reply = old.net.regret = False
    old_net, old_ema = ConvNet(old.net).to(DEVICE), ConvNet(old.net).to(DEVICE)
    old_learner = Learner(old, old_net, old_ema, DEVICE)
    old_learner.train(replay)
    path = str(tmp_path / "old.pt")
    save_checkpoint(path, old_net, old_ema, optimizer_state(old_learner.opt, old_net), 0, old)
    ck = load_checkpoint(path)
    assert "control" not in ck
    new_net, new_ema = build(cfg, ck, DEVICE, "model"), build(cfg, ck, DEVICE, "ema")
    assert new_net.fresh and all(k.startswith(("reply.", "regret.")) for k in new_net.fresh)
    share_fresh_heads(new_net, new_ema)
    for k in new_net.fresh:
        assert torch.equal(new_net.state_dict()[k], new_ema.state_dict()[k])
    fresh = Learner(cfg, new_net, new_ema, DEVICE)
    load_optimizer_state(fresh.opt, new_net, ck["optimizer"], DEVICE)
    assert len(fresh.opt.state) == len(list(new_net.parameters()))
    assert all(fresh.opt.state[p]["exp_avg"].abs().sum() == 0 for p in list(new_net.parameters())[-11:])
    fresh.snapshot()
    # The control's state round-trips through the checkpoint.
    save_checkpoint(path, new_net, new_ema, None, 1, cfg, {"control": control.state()})
    again = SearchControl(cfg.control, cfg.learn.envs, cfg.learn.steps, DEVICE, seed=0, warmup=2)
    again.load(load_checkpoint(path)["control"])
    assert again.buffer.entries.keys() == control.buffer.entries.keys() and again.warmup_left == 0
    assert again.heads_active == control.heads_active and again.rank_lift == control.rank_lift
    assert torch.allclose(torch.tensor(again.buffer.priorities()), torch.tensor(control.buffer.priorities()))


def test_wdl_head_trains_when_enabled(tmp_path):
    cfg = tiny_config(steps=48)  # long enough for games to end under the 6-12 ply clock
    cfg.net.wdl = True
    net = ConvNet(cfg.net).to(DEVICE)
    ema = ConvNet(cfg.net).to(DEVICE)
    ema.load_state_dict(net.state_dict())
    ema.eval()
    actor = Actor(cfg, DEVICE)
    with torch.no_grad():
        rollout = actor.collect(student_evaluator(ema))
    window = build_window(rollout, cfg, iteration=0)
    replay = Replay(cfg.learn, str(tmp_path / "run"), seed=0)
    from conv.replay import Outcomes

    Outcomes(cfg.learn.envs, cfg.learn.steps).label(rollout.done, rollout.end_reason, window, replay)
    assert window.outcome_ok.any() and (window.outcome[~window.outcome_ok] == 0).all()
    replay.add(window)
    assert replay.load(0).outcome_ok.sum() == window.outcome_ok.sum()
    learner = Learner(cfg, net, ema, DEVICE)
    stats = learner.train(replay)
    assert math.isfinite(stats["wdl"]) and stats["wdl"] > 0
    # A checkpoint without the head loads into a network with it.
    path = str(tmp_path / "old.pt")
    cfg.net.wdl = False
    save_checkpoint(path, ConvNet(cfg.net), ConvNet(cfg.net), None, 0, cfg)
    cfg.net.wdl = True
    assert build(cfg, load_checkpoint(path), "cpu", "model").wdl is not None
