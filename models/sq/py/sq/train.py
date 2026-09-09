"""Self-play training loop: collect one window with the search, build labels,
train on the replay, checkpoint. One process, one GPU, strictly alternating
collection and learning.

    python -m sq.train --run <name> --init weights/sq_g128.pt
    python -m sq.train --run <name> --resume runs/<name>/latest.pt

Writes `<workspace root>/runs/<run>/`: `config.json`, `latest.pt`, `ckpt_NNNNNN.pt`, the replay
windows, and `log.csv` with one row per iteration: losses, throughput and the
conversion metrics (share of games won outright, ended by the clock or the ply
cap, mean lengths, repetition-draw and tree-reuse rates)."""

from __future__ import annotations

import csv
import json
import os
import time

import torch
import torch.nn.functional as F

from . import kernels, paths
from .config import Config, parse, to_dict
from .env import END_CLOCK, END_MAX_PLIES, END_WIN, Env
from .model import TTE_EDGES, SqNet, build, config_of, count_params, hl_gauss, load_checkpoint, save_checkpoint
from .planes import CYCLE_CELL, DIAG_DIR_PERM, DIAG_SQUARE_PERM, N_ACTIONS, N_DIRS, N_SQUARES
from .replay import Replay, Rollout, Window, build_window
from .search import GumbelSearch, RepetitionHistory, raw_policy


def conversion_metrics(rollout: Rollout) -> dict:
    """How collected games end: outright wins against artificial endings, and lengths."""
    reason, ply = rollout.end_reason, rollout.ply.float() + 1.0
    win, clock, cap = reason == END_WIN, reason == END_CLOCK, reason == END_MAX_PLIES
    ended = win | clock | cap
    n = ended.sum().item()
    return {
        "episodes": n,
        "win_end_frac": win.sum().item() / max(n, 1),
        "clock_end_frac": clock.sum().item() / max(n, 1),
        "max_plies_end_frac": cap.sum().item() / max(n, 1),
        "episode_len_mean": (ply * ended).sum().item() / max(n, 1),
        "win_len_mean": (ply * win).sum().item() / max(win.sum().item(), 1),
    }


def acting(net):
    """The (logits, q) pair the actor's search consumes, from a network or its compiled form."""

    def forward(planes):
        out = net(planes)
        return out.logits, out.q

    return forward


def compile_net(net: SqNet, cfg: Config):
    if not cfg.learn.compile:
        return net
    import torch._functorch.config as functorch_config
    import torch._inductor.config as inductor_config

    inductor_config.layout_optimization = False
    functorch_config.backward_pass_autocast = "off"
    # Each block is compiled on its own. In one whole-network graph Inductor hoists every
    # block's recomputation to the start of the backward pass and the memory saving of
    # `Block.recompute` is lost; per block, autograd recomputes one block at a time.
    # The actor (envs boards) and the learner (batch rows) compile the same code; left
    # to itself, dynamo answers the second batch size with one symbolic-batch graph that
    # both sides then run, so shapes are pinned static.
    for block in [*net.blocks, net.q_block]:
        block.forward = torch.compile(block.forward, dynamic=False)
    net.embed = torch.compile(net.embed, dynamic=False)
    net.heads = torch.compile(net.heads, dynamic=False)
    return net


class Actor:
    """Owns the environment, the search arena and the repetition history;
    collects one rollout per call with the network it is given."""

    def __init__(self, cfg: Config, device):
        self.cfg = cfg
        self.device = torch.device(device)
        n = cfg.learn.envs
        self.capture_clock = torch.tensor(cfg.rules.capture_clock, dtype=torch.int32, device=device)
        self.env = Env(n, device, cfg.rules.max_plies, self.capture_clock)
        self.history = RepetitionHistory(n, cfg.rules.capture_clock, device)
        self.history.reset(self.env.board, self.env.ply)
        self.search = GumbelSearch(
            cfg.search,
            n,
            device,
            self.capture_clock,
            cfg.rules.clock_penalty,
            cfg.rules.max_plies,
            cfg.seed + 1,
            self.history,
        )
        self.rollout = Rollout.allocate(cfg.learn.steps, n, cfg.search.candidates, device)
        self.mode_rng = torch.Generator().manual_seed(cfg.seed + 2)

    @torch.no_grad()
    def collect(self, forward) -> Rollout:
        """One window of `steps` plies on every board with `forward` (planes -> logits, q)."""
        env, search, buf = self.env, self.search, self.rollout
        search.forget()
        for t in range(self.cfg.learn.steps):
            full = bool(torch.rand((), generator=self.mode_rng) < self.cfg.search.full_fraction)
            buf.board[t].copy_(env.board)
            buf.since_capture[t].copy_(env.since_capture.to(torch.int16))
            buf.ply[t].copy_(env.ply.to(torch.int16))
            target, value, action, candidates, candidate_q, visited = search(
                env.board, env.since_capture, env.ply, env.legal, env.planes, forward, full
            )
            buf.target[t].copy_(target.to(torch.bfloat16))
            buf.target_ok[t].fill_(full)
            buf.value[t].copy_(value)
            buf.action[t].copy_(action.to(torch.int16))
            buf.candidates[t].copy_(candidates.to(torch.int16))
            buf.candidate_q[t].copy_(candidate_q)
            buf.candidate_visited[t].copy_(visited)
            env.step(action.to(torch.int32))
            self.history.push(env.board, env.since_capture, env.ply, env.done)
            search.advance(action, env.done)
            buf.reward[t].copy_(env.reward)
            buf.done[t].copy_(env.done)
            buf.end_reason[t].copy_(env.end_reason)
        _, value, *_ = search(env.board, env.since_capture, env.ply, env.legal, env.planes, forward, True)
        buf.value[self.cfg.learn.steps].copy_(value)
        return buf


class Augment:
    """Diagonal reflection and cyclic type relabelling, both exact symmetries."""

    def __init__(self, device):
        self.dir_perm = torch.tensor(DIAG_DIR_PERM, device=device)
        self.square_perm = torch.tensor(DIAG_SQUARE_PERM, device=device)
        self.cycle = torch.tensor(CYCLE_CELL, dtype=torch.int64, device=device)

    def board(self, b, flip, k):
        b = torch.where(flip[:, None], b.view(-1, 9, 9).transpose(1, 2).reshape(-1, N_SQUARES), b)
        return self.cycle[k].gather(1, b.long()).to(b.dtype)

    def squares(self, m, flip):
        return torch.where(flip[:, None], m.view(-1, 9, 9).transpose(1, 2).reshape(-1, N_SQUARES), m)

    def actions(self, a, flip):
        d, s = a // N_SQUARES, a % N_SQUARES
        flipped = self.dir_perm[d] * N_SQUARES + self.square_perm[s]
        return torch.where(flip.view(-1, *([1] * (a.ndim - 1))), flipped, a)

    def policy(self, p, flip):
        flipped = p.view(-1, N_DIRS, 9, 9)[:, self.dir_perm].transpose(2, 3).reshape(-1, N_ACTIONS)
        return torch.where(flip[:, None], flipped, p)


def optimizer_state(opt, net: SqNet) -> dict:
    """Optimizer moments keyed by parameter index in `net.parameters()` order."""
    return {
        "state": {
            i: {k: v.detach().cpu() for k, v in opt.state[p].items()}
            for i, p in enumerate(net.parameters())
            if p in opt.state
        }
    }


def load_optimizer_state(opt, net: SqNet, state: dict, device) -> None:
    params = list(net.parameters())
    for i, entry in state["state"].items():
        opt.state[params[int(i)]] = {
            k: v.to(device).contiguous() if isinstance(v, torch.Tensor) else v for k, v in entry.items()
        }


class Learner:
    """Owns the network, its EMA copy, the optimizer and the training step;
    trains on the replay one iteration at a time."""

    METRICS = (
        "policy",
        "q",
        "candidate_q",
        "occupancy",
        "plies_to_end",
        "reply",
        "danger",
        "material",
        "entropy",
        "kl",
        "grad_norm",
        "abs_return",
    )
    WARMUP = 3

    def __init__(self, cfg: Config, net: SqNet, ema: SqNet, device):
        self.cfg, self.net, self.ema = cfg, net, ema
        self.device = torch.device(device)
        cuda = self.device.type == "cuda"
        decay = [p for p in net.parameters() if p.ndim >= 2]
        no_decay = [p for p in net.parameters() if p.ndim < 2]
        self.opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.learn.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.learn.lr,
            betas=(0.9, 0.999),
            fused=cuda,
            capturable=cuda and cfg.learn.graphs,
        )
        self.forward = compile_net(net, cfg)
        # Eager steps before a graph capture run on a side stream, as the capture will.
        self.side = torch.cuda.Stream() if cuda and cfg.learn.graphs else None
        self.augment = Augment(device)
        self.tte_edges = torch.tensor(TTE_EDGES, dtype=torch.int32, device=device)
        self.ema_params = list(ema.parameters())
        self.params = list(net.parameters())
        self.ema_buffers = [b for b in ema.buffers() if b.dtype.is_floating_point or b.dtype == torch.int64]
        self.buffers = [b for b in net.buffers() if b.dtype.is_floating_point or b.dtype == torch.int64]
        B, C = cfg.learn.batch, cfg.search.candidates

        def z(*shape, dtype):
            return torch.zeros(*shape, dtype=dtype, device=device)

        self.st = {
            "board": z(B, N_SQUARES, dtype=torch.int8),
            "since_capture": z(B, dtype=torch.int16),
            "ply": z(B, dtype=torch.int16),
            "action": z(B, dtype=torch.int16),
            "target": z(B, N_ACTIONS, dtype=torch.bfloat16),
            "target_ok": z(B, dtype=torch.bool),
            "ret": z(B, dtype=torch.float32),
            "candidates": z(B, C, dtype=torch.int16),
            "candidate_q": z(B, C, dtype=torch.float32),
            "candidate_visited": z(B, C, dtype=torch.bool),
            "occupancy": z(B, N_SQUARES, dtype=torch.int8),
            "occupancy_ok": z(B, dtype=torch.bool),
            "plies_to_end": z(B, dtype=torch.int32),
            "reply": z(B, dtype=torch.int16),
            "reply_ok": z(B, dtype=torch.bool),
            "danger": z(B, N_SQUARES, dtype=torch.bool),
            "danger_ok": z(B, dtype=torch.bool),
            "material": z(B, 2, dtype=torch.int8),
            "flip": z(B, dtype=torch.bool),
            "cycle": z(B, dtype=torch.int64),
        }
        self.host = {
            k: torch.empty(v.shape, dtype=v.dtype, pin_memory=cuda)
            for k, v in self.st.items()
            if k not in ("flip", "cycle")
        }
        self.sums = torch.zeros(len(self.METRICS), device=device)
        # Marks the end of a batch's host-to-device copies; the pinned buffers are rewritten
        # only after it, or a batch still queued behind the GPU would read the next one.
        self.copied = torch.cuda.Event() if cuda else None
        self.graph = None
        self.eager_steps = 0
        self.aug_rng = torch.Generator(device=device).manual_seed(cfg.seed + 3)

    def stage(self, window: Window, idx: torch.Tensor) -> None:
        """Gather one batch of window rows into the static device inputs."""
        if self.copied is not None:
            self.copied.synchronize()
        for name in self.host:
            source = (
                window.occupancy(idx, self.cfg.learn.occupancy_horizon)
                if name == "occupancy"
                else getattr(window, name)[idx]
            )
            self.host[name].copy_(source)
            self.st[name].copy_(self.host[name], non_blocking=True)
        if self.copied is not None:
            self.copied.record()
        self.st["flip"].copy_(torch.rand(self.cfg.learn.batch, generator=self.aug_rng, device=self.device) < 0.5)
        self.st["cycle"].copy_(torch.randint(3, (self.cfg.learn.batch,), generator=self.aug_rng, device=self.device))

    def losses(self) -> torch.Tensor:
        """Forward and backward on the staged batch; returns the metric contributions."""
        cfg, st, aug = self.cfg.learn, self.st, self.augment
        flip, k = st["flip"], st["cycle"]
        board = aug.board(st["board"], flip, k)
        target = aug.policy(st["target"].float(), flip)
        action = aug.actions(st["action"].long(), flip)
        candidates = aug.actions(st["candidates"].long(), flip)
        occupancy = aug.board(st["occupancy"], flip, k).long()
        reply = aug.actions(st["reply"].long(), flip)
        danger = aug.squares(st["danger"], flip).float()
        planes, legal, _ = kernels.derive_batch(board, st["since_capture"].int(), st["ply"].int())
        if planes.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                out = self.forward(planes, aux=True)
        else:
            out = self.forward(planes.float(), aux=True)
        _, logp = raw_policy(out.logits, legal)
        target_ok = st["target_ok"]
        policy = -(target * logp.masked_fill(~legal, 0.0)).sum(-1)
        policy_loss = (policy * target_ok).sum() / target_ok.sum().clamp_min(1)
        atoms = out.q_logits.shape[-1]
        played = out.q_logits.gather(1, action[:, None, None].expand(-1, 1, atoms)).squeeze(1).float()
        ret = st["ret"]
        q_loss = -(hl_gauss(ret, atoms, cfg.hl_gauss_sigma) * played.log_softmax(-1)).sum(-1).mean()
        visited = st["candidate_visited"] & target_ok[:, None]
        chosen = out.q_logits.gather(1, candidates[..., None].expand(-1, -1, atoms)).float()
        cq_target = hl_gauss(st["candidate_q"].reshape(-1), atoms, cfg.hl_gauss_sigma).reshape_as(chosen)
        cq_ce = -(cq_target * chosen.log_softmax(-1)).sum(-1)
        cq_loss = (cq_ce * visited).sum() / visited.sum().clamp_min(1)
        occ_ok = st["occupancy_ok"]
        occ_ce = F.cross_entropy(out.occupancy.float(), occupancy, reduction="none").mean(1)
        occ_loss = (occ_ce * occ_ok).sum() / occ_ok.sum().clamp_min(1)
        tte = st["plies_to_end"]
        tte_ok = tte > 0
        tte_class = (torch.searchsorted(self.tte_edges, tte.clamp_min(1), right=True) - 1).long()
        tte_ce = F.cross_entropy(out.plies_to_end.float(), tte_class, reduction="none")
        tte_loss = (tte_ce * tte_ok).sum() / tte_ok.sum().clamp_min(1)
        reply_ok = st["reply_ok"]
        reply_ce = F.cross_entropy(out.reply.float(), reply, reduction="none")
        reply_loss = (reply_ce * reply_ok).sum() / reply_ok.sum().clamp_min(1)
        danger_ok = st["danger_ok"]
        danger_bce = F.binary_cross_entropy_with_logits(out.danger.float(), danger, reduction="none").mean(1)
        danger_loss = (danger_bce * danger_ok).sum() / danger_ok.sum().clamp_min(1)
        material_se = (out.material.float() - st["material"].float() / 10.0).pow(2).mean(1)
        material_loss = (material_se * tte_ok).sum() / tte_ok.sum().clamp_min(1)
        loss = (
            policy_loss
            + q_loss
            + cfg.w_candidate_q * cq_loss
            + cfg.w_occupancy * occ_loss
            + cfg.w_plies_to_end * tte_loss
            + cfg.w_reply * reply_loss
            + cfg.w_danger * danger_loss
            + cfg.w_material * material_loss
        )
        loss.backward()
        with torch.no_grad():
            log_target = target.clamp_min(1e-12).log().masked_fill(~legal, 0.0)
            entropy = -(target * log_target).sum(-1)
            kl = (target * (log_target - logp.masked_fill(~legal, 0.0))).sum(-1)
            entropy = (entropy * target_ok).sum() / target_ok.sum().clamp_min(1)
            kl = (kl * target_ok).sum() / target_ok.sum().clamp_min(1)
            return torch.stack(
                [
                    policy_loss,
                    q_loss,
                    cq_loss,
                    occ_loss,
                    tte_loss,
                    reply_loss,
                    danger_loss,
                    material_loss,
                    entropy,
                    kl,
                    torch.zeros((), device=loss.device),
                    ret.abs().mean(),
                ]
            ).detach()

    def step(self) -> None:
        """One optimizer step on the staged batch, then the EMA update."""
        metrics = self.losses()
        metrics[10] = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.learn.grad_clip).float()
        self.opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(self.ema_params, self.params, 1.0 - self.cfg.learn.ema)
            torch._foreach_copy_(self.ema_buffers, self.buffers)
        self.sums.add_(metrics)

    def run_step(self) -> None:
        if self.graph is not None:
            self.graph.replay()
        elif self.cfg.learn.graphs and self.device.type == "cuda" and self.eager_steps >= self.WARMUP:
            self.opt.zero_grad(set_to_none=True)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
                self.step()
            # The warm-up steps' cached blocks cannot serve the graph's private pool.
            torch.cuda.empty_cache()
            self.graph.replay()
        elif self.side is not None:
            self.opt.zero_grad(set_to_none=True)
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                self.step()
            torch.cuda.current_stream().wait_stream(self.side)
            self.eager_steps += 1
        else:
            self.opt.zero_grad(set_to_none=True)
            self.step()
            self.eager_steps += 1

    def train(self, replay: Replay) -> dict:
        """One iteration over the replay; returns the averaged metrics."""
        self.net.train()
        self.sums.zero_()
        batches = 0
        for window, idx in replay.batches():
            self.stage(window, idx)
            self.run_step()
            batches += 1
        self.net.eval()
        means = (self.sums / max(batches, 1)).tolist()
        return {**dict(zip(self.METRICS, means, strict=True)), "batches": batches}


def append_row(path: str, row: dict) -> None:
    """Append one row; when the row has columns the file lacks, the file is rewritten
    with the widened header and blanks in the old rows."""
    fields = list(row)
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            old_fields, old_rows = list(reader.fieldnames or []), list(reader)
        if old_fields != fields:
            fields = old_fields + [k for k in fields if k not in old_fields]
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(old_rows)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = str(paths.run_dir(cfg.run))
    os.makedirs(run_dir, exist_ok=True)
    start = 0
    checkpoint = None
    if cfg.resume or cfg.init:
        checkpoint = load_checkpoint(cfg.resume or cfg.init)
        saved = config_of(checkpoint)
        cfg.net, cfg.play = saved.net, saved.play
        if cfg.resume:
            cfg.rules = saved.rules
            start = checkpoint["iteration"] + 1
    net = build(cfg, checkpoint, device, "model")
    ema = build(cfg, checkpoint, device, "ema")
    learner = Learner(cfg, net, ema, device)
    replay = Replay(cfg.learn.replay_windows, cfg.learn.epochs, cfg.learn.batch, run_dir, cfg.seed + 4)
    if cfg.resume and checkpoint.get("optimizer"):
        load_optimizer_state(learner.opt, net, checkpoint["optimizer"], device)
        print(f"resumed {cfg.resume} at iteration {start}; replay windows restored: {replay.restore(start - 1)}")
    elif checkpoint is not None:
        print(f"initialised from {cfg.init} (iteration {checkpoint['iteration']}); the optimizer starts fresh")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(to_dict(cfg), f, indent=2)
    print(f"params {count_params(net) / 1e6:.2f}M; device {device}")
    actor = Actor(cfg, device)
    ema.eval()
    act = compile_net(ema, cfg)
    net.eval()
    forward = acting(act)
    log_path = os.path.join(run_dir, "log.csv")
    for it in range(start, cfg.iters):
        t0 = time.time()
        rollout = actor.collect(forward)
        window = build_window(rollout, cfg, it)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        replay.add(window)
        stats = learner.train(replay)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        row = {
            "iter": it,
            "t_collect": round(t1 - t0, 2),
            "t_train": round(t2 - t1, 2),
            "steps_per_s": round(cfg.learn.envs * cfg.learn.steps / max(t1 - t0, 1e-6)),
            "replay_positions": replay.positions(),
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in stats.items()},
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in conversion_metrics(rollout).items()},
            **{k: round(v, 5) for k, v in actor.search.metrics().items()},
            "gpu_reserved_mb": torch.cuda.memory_reserved() >> 20 if device.type == "cuda" else 0,
        }
        append_row(log_path, row)
        print(f"[it {it}] {row}", flush=True)
        save_checkpoint(os.path.join(run_dir, "latest.pt"), net, ema, optimizer_state(learner.opt, net), it, cfg)
        if it % cfg.checkpoint_every == 0:
            save_checkpoint(
                os.path.join(run_dir, f"ckpt_{it:06d}.pt"), net, ema, optimizer_state(learner.opt, net), it, cfg
            )
        replay.prune(it)


if __name__ == "__main__":
    main(parse())
