"""One tiny collection and learner iteration end to end: labels have the right
shapes and masks, losses are finite, the checkpoint round-trips, and the
export matches torch. Runs on the CPU (Triton interpreter) without a GPU."""

import os

import torch

from sq.config import Config, Learn, Net, Search
from sq.model import SqNet, build, load_checkpoint, save_checkpoint
from sq.replay import Replay, build_window, lambda_returns, plies_to_end
from sq.train import Actor, Learner, acting, conversion_metrics

# TRITON_INTERPRET=1 in the environment runs everything on the CPU.
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


def tiny_config(tmp_path) -> Config:
    cfg = Config(run="test", seed=3, checkpoint_every=1)
    cfg.net = Net(
        width=32,
        blocks=2,
        heads=2,
        ffn=64,
        smolgen_every=1,
        smolgen_dim=4,
        smolgen_hidden=16,
        smolgen_latent=8,
        atoms=11,
        head_dim=8,
    )
    cfg.search = Search(
        sims=6, candidates=3, cheap_sims=3, cheap_candidates=2, full_fraction=0.5, reuse_nodes=6, temperature_plies=2
    )
    cfg.learn = Learn(
        envs=4,
        steps=6,
        replay_windows=2,
        epochs=1,
        batch=8,
        compile=False,
        graphs=False,
        occupancy_horizon=2,
        danger_horizon=3,
    )
    cfg.rules.max_plies = 40
    return cfg


def test_lambda_returns_and_plies_to_end():
    reward = torch.tensor([[0.0], [0.0], [1.0]])
    done = torch.tensor([[False], [False], [True]])
    value = torch.zeros(4, 1)
    ret = lambda_returns(reward, done, value, lam=1.0)
    assert ret.flatten().tolist() == [1.0, -1.0, 1.0]
    terminal = torch.tensor([[False], [False], [True]])
    assert plies_to_end(done, terminal).flatten().tolist() == [3, 2, 1]
    assert plies_to_end(done, torch.zeros_like(done)).flatten().tolist() == [-1, -1, -1]


def test_collection_labels_learner_step_and_checkpoint(tmp_path):
    cfg = tiny_config(tmp_path)
    net = SqNet(cfg.net).to(DEVICE)
    ema = SqNet(cfg.net).to(DEVICE)
    ema.load_state_dict(net.state_dict())
    ema.eval()
    actor = Actor(cfg, DEVICE)
    forward = acting(ema)
    with torch.no_grad():
        rollout = actor.collect(forward)
    window = build_window(rollout, cfg, iteration=0)
    rows = cfg.learn.envs * cfg.learn.steps
    assert window.board.shape == (rows, 81) and window.target.shape == (rows, 648)
    assert window.target_ok.dtype == torch.bool and window.plies_to_end.min() >= -1
    assert (window.material[window.plies_to_end > 0] >= 0).all()
    assert torch.isfinite(window.ret).all() and window.ret.abs().max() <= 1.0
    assert window.candidate_visited[window.target_ok].any()
    metrics = conversion_metrics(rollout)
    assert 0.0 <= metrics["win_end_frac"] <= 1.0

    learner = Learner(cfg, net, ema, DEVICE)
    replay = Replay(cfg.learn.replay_windows, cfg.learn.epochs, cfg.learn.batch, str(tmp_path / "run"), seed=0)
    replay.add(window)
    stats = learner.train(replay)
    assert stats["batches"] == rows // cfg.learn.batch
    assert all(abs(v) < 1e6 for k, v in stats.items() if k != "batches")
    assert stats["policy"] > 0 and stats["q"] > 0
    assert replay.restore(0) == 1

    path = str(tmp_path / "ckpt.pt")
    from sq.train import load_optimizer_state, optimizer_state

    save_checkpoint(path, net, ema, optimizer_state(learner.opt, net), 0, cfg)
    ck = load_checkpoint(path)
    again = build(cfg, ck, DEVICE, "model")
    for a, b in zip(again.state_dict().values(), net.state_dict().values(), strict=True):
        assert torch.equal(a.cpu(), b.cpu())
    fresh = Learner(cfg, again, build(cfg, ck, DEVICE, "ema"), DEVICE)
    load_optimizer_state(fresh.opt, again, ck["optimizer"], DEVICE)
    assert len(fresh.opt.state) == len(learner.opt.state)


def test_export_matches_torch(tmp_path):
    from sq.export import check, export

    cfg = tiny_config(tmp_path)
    net = SqNet(cfg.net)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, net, net, None, 0, cfg)
    out = str(tmp_path / "model.onnx")
    meta = export(path, out, "ema")
    assert meta["atoms"] == cfg.net.atoms and os.path.exists(out + ".json")
    deltas = check(path, out, "ema", positions=8)
    assert max(deltas.values()) < 2e-3


def test_append_row_widens_header(tmp_path):
    import csv

    from sq.train import append_row

    path = str(tmp_path / "log.csv")
    append_row(path, {"iter": 0, "a": 1.0})
    append_row(path, {"iter": 1, "a": 2.0, "b": 3})
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [list(r.values()) for r in rows] == [["0", "1.0", ""], ["1", "2.0", "3"]]
    assert list(rows[0]) == ["iter", "a", "b"]
