"""Self-play training loop: collect one window with the search, build labels,
train on the replay, checkpoint. One process, one GPU, strictly alternating
collection and learning.

    python -m conv.train --run <name>
    python -m conv.train --run <name> --resume runs/<name>/latest.pt

The actor driving self-play is the student's EMA; a run usually starts with
`--init runs/<name>/distil.pt` from `conv.distil` (DESIGN item 26).

Writes `<workspace root>/runs/<run>/`: `config.json`, `latest.pt`,
`ckpt_NNNNNN.pt`, the replay windows, and `log.csv` with one row per iteration:
losses, throughput, the conversion metrics (share of games won outright,
ended by the clock or the ply cap, mean lengths, repetition-draw and tree-reuse
rates), the share of batches skipped for an outlier gradient, whether the
iteration was rolled back, and the learning rate in force."""

from __future__ import annotations

import csv
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernels, paths
from .config import Config, parse, to_dict
from .control import SearchControl
from .env import END_CLOCK, END_MAX_PLIES, END_WIN, Env
from .model import (
    TTE_EDGES,
    ConvNet,
    atom_values,
    build,
    config_of,
    count_params,
    hl_gauss,
    load_checkpoint,
    save_checkpoint,
    share_fresh_heads,
)
from .planes import CYCLE_CELL, DIAG_DIR_PERM, DIAG_SQUARE_PERM, N_SQUARES
from .replay import Outcomes, Replay, Rollout, Window, build_window
from .search import SLOTS, GumbelSearch, RepetitionHistory, raw_policy, spread, unpack_tactics


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


def compile_net(module: nn.Module, cfg: Config):
    if not cfg.learn.compile:
        return module
    import torch._functorch.config as functorch_config
    import torch._inductor.config as inductor_config

    inductor_config.layout_optimization = False
    functorch_config.backward_pass_autocast = "off"
    return torch.compile(module)


def student_evaluator(forward, draw_kernel_width: float = 0.15, contempt_source: str = "wdl"):
    """The self-play actor: conv's 46 planes and the network without its
    training-only heads; a leaf is worth its state value V. WDL contempt uses
    the outcome head's draw probability per state; `q` keeps the per-action Q
    mass that the export computes for the Rust player (DESIGN.md item 17)."""
    if contempt_source not in ("wdl", "q"):
        raise ValueError(f"unknown contempt draw source {contempt_source!r}")
    kernel = None

    def evaluate(board, since, ply, clock):
        nonlocal kernel
        planes, legal, count = kernels.derive_batch(board, since, ply, clock)
        if planes.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = forward(planes)
        else:
            out = forward(planes.float())
        if contempt_source == "wdl":
            if out.wdl is None:
                raise ValueError("WDL contempt requires Net.wdl")
            draw = torch.softmax(out.wdl.float(), -1)[:, 1]
        else:
            if kernel is None:
                kernel = torch.exp(
                    -((atom_values(out.q_logits.shape[-1], out.q_logits.device) / draw_kernel_width) ** 2)
                )
            draw = torch.softmax(out.q_logits.float(), -1) @ kernel
        if out.regret is None:
            return out.logits, out.q, out.v, draw, legal, count
        regret = out.regret.float()
        return out.logits, out.q, out.v, draw, legal, count, regret[:, 0], regret[:, 1]

    return evaluate


class Actor:
    """Owns the environment, the search arena and the repetition history;
    collects one rollout per call with the actor it is given."""

    def __init__(self, cfg: Config, device):
        self.cfg = cfg
        self.device = torch.device(device)
        n = cfg.learn.envs
        self.env = Env(n, device, cfg.rules.max_plies, cfg.rules.clock_min, cfg.rules.clock_max, cfg.seed + 5)
        self.history = RepetitionHistory(n, cfg.rules.clock_max, device)
        self.history.reset(self.env.board, self.env.ply)
        self.search = GumbelSearch(
            cfg.search,
            n,
            device,
            self.env.clock,
            cfg.rules.clock_penalty,
            cfg.rules.max_plies,
            cfg.seed + 1,
            self.history,
        )
        self.rollout = Rollout.allocate(cfg.learn.steps, n, cfg.search.candidates, device)
        self.mode_rng = torch.Generator().manual_seed(cfg.seed + 2)
        self.control = None  # the search control, when restarts come from its buffer

    @torch.no_grad()
    def collect(self, evaluate) -> Rollout:
        """One window of `steps` plies on every board with `evaluate`."""
        env, search, buf = self.env, self.search, self.rollout
        search.forget()
        for t in range(self.cfg.learn.steps):
            full = bool(torch.rand((), generator=self.mode_rng) < self.cfg.search.full_fraction)
            buf.board[t].copy_(env.board)
            buf.since_capture[t].copy_(env.since_capture.to(torch.int16))
            buf.ply[t].copy_(env.ply.to(torch.int16))
            buf.clock[t].copy_(env.clock.to(torch.int16))
            buf.origin[t].copy_(env.origin)
            root = search(env.board, env.since_capture, env.ply, env.legal, evaluate, full)
            buf.moves[t].copy_(root.moves)
            buf.target[t].copy_(root.target.to(torch.bfloat16))
            buf.target_ok[t].fill_(full)
            buf.tactics[t].copy_(root.tactics)
            buf.value[t].copy_(root.value)
            buf.action[t].copy_(root.action.to(torch.int16))
            buf.candidates[t].copy_(root.candidates.to(torch.int16))
            buf.candidate_q[t].copy_(root.candidate_q)
            buf.candidate_visited[t].copy_(root.visited)
            buf.played_q[t].copy_(root.played_q)
            buf.rank[t].copy_(root.rank)
            tree_score, tree_regret, tree_board, tree_since, tree_ply = search.sample_node()
            buf.tree_score[t].copy_(tree_score)
            buf.tree_regret[t].copy_(tree_regret)
            buf.tree_board[t].copy_(tree_board)
            buf.tree_since[t].copy_(tree_since.to(torch.int16))
            buf.tree_ply[t].copy_(tree_ply.to(torch.int16))
            if self.control is not None:
                self.control.draw(env)
            env.step(root.action.to(torch.int32))
            self.history.push(env.board, env.since_capture, env.ply, env.done)
            search.advance(env.done)
            buf.reward[t].copy_(env.reward)
            buf.done[t].copy_(env.done)
            buf.end_reason[t].copy_(env.end_reason)
        buf.value[self.cfg.learn.steps].copy_(search(env.board, env.since_capture, env.ply, env.legal, evaluate).value)
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

    def actions(self, a, flip):
        """Action indices of any shape `(B, ...)`: the played move, the
        candidates and the root moves whose slots carry the target and labels."""
        d, s = a // N_SQUARES, a % N_SQUARES
        flipped = self.dir_perm[d] * N_SQUARES + self.square_perm[s]
        return torch.where(flip.view(-1, *([1] * (a.ndim - 1))), flipped, a)


def optimizer_state(opt, net: ConvNet) -> dict:
    """Optimizer moments keyed by parameter index in `net.parameters()` order."""
    return {
        "state": {
            i: {k: v.detach().cpu() for k, v in opt.state[p].items()}
            for i, p in enumerate(net.parameters())
            if p in opt.state
        }
    }


def regret_parameter_indices(net: ConvNet) -> list[int]:
    """Indices of the regret head in `net.parameters()` checkpoint order."""
    return [i for i, (name, _) in enumerate(net.named_parameters()) if name.startswith("regret.")]


def ranking_loss(gamma: torch.Tensor, regret: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """Listwise regret-ranking loss over labelled rows, or exact zero when none are labelled."""
    dummy = (torch.arange(len(ok), device=ok.device) == 0) & ~ok.any()
    members = ok | dummy
    scores = torch.where(ok, gamma, torch.zeros_like(gamma)).masked_fill(~members, -torch.inf)
    labels = torch.where(ok, regret, torch.zeros_like(regret))
    return torch.logsumexp(scores, 0) - torch.logsumexp(scores + labels, 0)


def load_optimizer_state(opt, net: ConvNet, state: dict, device) -> int:
    """Returns the number of steps the state has taken. Parameters the state
    does not cover (fresh heads) get zero moments in the same layout, so the
    first rollback snapshot holds every parameter's state."""
    params = list(net.parameters())
    steps = 0
    template = None
    for i, entry in state["state"].items():
        opt.state[params[int(i)]] = template = {
            k: v.to(device).contiguous() if isinstance(v, torch.Tensor) else v for k, v in entry.items()
        }
        steps = max(steps, int(entry["step"]))
    if template is not None:
        for p in params:
            if p not in opt.state:
                opt.state[p] = {
                    k: torch.zeros_like(v) if k == "step" else torch.zeros_like(p) for k, v in template.items()
                }
    return steps


def lr_multiplier(learn, iteration: int) -> float:
    """Iteration schedule of DESIGN.md item 25: hold, cosine decay, then floor."""
    if learn.lr_decay_iters <= learn.lr_hold_iters:
        raise ValueError("lr_decay_iters must be greater than lr_hold_iters")
    if iteration <= learn.lr_hold_iters:
        return 1.0
    minimum = learn.lr_min / learn.lr
    if iteration >= learn.lr_decay_iters:
        return minimum
    progress = (iteration - learn.lr_hold_iters) / (learn.lr_decay_iters - learn.lr_hold_iters)
    return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * progress))


class Learner:
    """Owns the network, its EMA copy, the optimizer and the training step;
    trains on the replay one iteration at a time."""

    METRICS = (
        "policy",
        "q",
        "candidate_q",
        "exact_q",
        "value",
        "tactics",
        "occupancy",
        "plies_to_end",
        "material",
        "entropy",
        "kl",
        "grad_norm",
        "abs_return",
        "skipped_frac",
        "wdl",
        "reply",
        "regret",
        "rank",
        "value_fresh",  # the value loss on rows of games from the initial position
        "value_replay",  # and on rows of games the search control restarted
    )
    WARMUP = 3

    def __init__(self, cfg: Config, net: ConvNet, ema: ConvNet, device):
        self.cfg, self.net, self.ema = cfg, net, ema
        self.device = torch.device(device)
        cuda = self.device.type == "cuda"
        capturable = cuda and cfg.learn.graphs
        conv = {id(m.weight) for m in net.modules() if isinstance(m, nn.Conv2d)}
        matrices = [p for p in net.parameters() if p.ndim >= 2 and id(p) not in conv]
        vectors = [p for p in net.parameters() if p.ndim < 2]
        convs = [p for p in net.parameters() if id(p) in conv]
        assert len(matrices) + len(vectors) + len(convs) == len(list(net.parameters()))
        groups = [
            {"params": matrices, "weight_decay": cfg.learn.weight_decay, "lr_scale": 1.0},
            {"params": vectors, "weight_decay": 0.0, "lr_scale": 1.0},
            {"params": convs, "weight_decay": cfg.learn.weight_decay, "lr_scale": cfg.learn.conv_lr_scale},
        ]
        # A captured graph reads the learning rate from a tensor that the
        # schedule, warm-up and rollback factor rewrite in place.
        for g in groups:
            value = cfg.learn.lr * g["lr_scale"]
            g["lr"] = torch.full((), value, device=device) if capturable else value
        self.opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), fused=cuda, capturable=capturable)
        self.steps = 0
        self.iteration = 0
        self.rollback_factor = 1.0
        self.forward = compile_net(net, cfg)
        # Eager steps before a graph capture run on a side stream, as the capture will.
        self.side = torch.cuda.Stream() if cuda and cfg.learn.graphs else None
        self.augment = Augment(device)
        self.tte_edges = torch.tensor(TTE_EDGES, dtype=torch.int32, device=device)
        # Atom distributions of a certain loss and a certain win, for the exact-tactics Q loss.
        self.exact_atoms = hl_gauss(torch.tensor([-1.0, 1.0], device=device), cfg.net.atoms, cfg.learn.hl_gauss_sigma)
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
            "clock": z(B, dtype=torch.int16),
            "action": z(B, dtype=torch.int16),
            "moves": z(B, SLOTS, dtype=torch.int16),
            "target": z(B, SLOTS, dtype=torch.bfloat16),
            "target_ok": z(B, dtype=torch.bool),
            "tactics": z(B, SLOTS, dtype=torch.uint8),
            "ret": z(B, dtype=torch.float32),
            "candidates": z(B, C, dtype=torch.int16),
            "candidate_q": z(B, C, dtype=torch.float32),
            "candidate_visited": z(B, C, dtype=torch.bool),
            "occupancy": z(B, N_SQUARES, dtype=torch.int8),
            "occupancy_ok": z(B, dtype=torch.bool),
            "plies_to_end": z(B, dtype=torch.int32),
            "material": z(B, 2, dtype=torch.int8),
            "material_ok": z(B, dtype=torch.bool),
            "outcome": z(B, dtype=torch.int8),
            "outcome_ok": z(B, dtype=torch.bool),
            "reply": z(B, dtype=torch.int16),
            "reply_ok": z(B, dtype=torch.bool),
            "regret": z(B, dtype=torch.float32),
            "regret_ok": z(B, dtype=torch.bool),
            "replayed": z(B, dtype=torch.bool),
            "flip": z(B, dtype=torch.bool),
            "cycle": z(B, dtype=torch.int64),
        }
        # Two host staging sets: the gather of the next batch overlaps the step
        # in flight, and each set is rewritten only once its device copy is done.
        self.host = [
            {
                k: torch.empty(v.shape, dtype=v.dtype, pin_memory=cuda)
                for k, v in self.st.items()
                if k not in ("flip", "cycle")
            }
            for _ in range(2)
        ]
        self.copied = [torch.cuda.Event() if cuda else None for _ in range(2)]
        self.staged = 0
        # Metric sums and what each averages over: batches, or rows for the split value losses.
        self.sums = torch.zeros(len(self.METRICS), device=device)
        self.counts = torch.zeros(len(self.METRICS), device=device)
        self.graph = None
        self.eager_steps = 0
        self.aug_rng = torch.Generator(device=device).manual_seed(cfg.seed + 3)

    def effective_base_lr(self, iteration: int | None = None) -> float:
        """Configured rate after the iteration schedule and persistent rollbacks."""
        at = self.iteration if iteration is None else iteration
        return self.cfg.learn.lr * lr_multiplier(self.cfg.learn, at) * self.rollback_factor

    def set_lr(self) -> None:
        """Every group's learning rate for the coming step: the configured rate
        times the iteration schedule, rollback and group factors, warmed up
        linearly from a fresh optimizer."""
        warmup = self.cfg.learn.warmup_steps
        factor = min(1.0, (self.steps + 1) / warmup) if warmup > 0 else 1.0
        for g in self.opt.param_groups:
            value = self.effective_base_lr() * g["lr_scale"] * factor
            if isinstance(g["lr"], torch.Tensor):
                g["lr"].fill_(value)
            else:
                g["lr"] = value

    def clip_grads(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale the gradients to the clip norm; a batch whose norm is not
        finite or above `skip_norm` is zeroed instead. Returns the norm before
        clipping and whether the batch was skipped."""
        grads = [p.grad for p in self.params if p.grad is not None]
        total = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
        ok = torch.isfinite(total) & (total < self.cfg.learn.skip_norm)
        coef = torch.where(ok, (self.cfg.learn.grad_clip / (total + 1e-6)).clamp(max=1.0), torch.zeros_like(total))
        torch._foreach_mul_(grads, coef)
        return total, (~ok).float()

    def snapshot(self) -> dict:
        """Copies of the weights, the EMA and the optimizer state, for `restore`."""
        return {
            "params": [p.detach().clone() for p in self.params],
            "ema": [p.detach().clone() for p in self.ema_params],
            "buffers": [b.clone() for b in self.buffers],
            "ema_buffers": [b.clone() for b in self.ema_buffers],
            "opt": {
                i: {k: v.clone() for k, v in self.opt.state[p].items() if isinstance(v, torch.Tensor)}
                for i, p in enumerate(self.params)
                if p in self.opt.state
            },
        }

    @torch.no_grad()
    def restore(self, snap: dict) -> None:
        """Copy a snapshot back in place, so a captured graph keeps its addresses."""
        torch._foreach_copy_(self.params, snap["params"])
        torch._foreach_copy_(self.ema_params, snap["ema"])
        torch._foreach_copy_(self.buffers, snap["buffers"])
        torch._foreach_copy_(self.ema_buffers, snap["ema_buffers"])
        for i, entry in snap["opt"].items():
            for k, v in entry.items():
                self.opt.state[self.params[i]][k].copy_(v)

    def stage(self, window: Window, idx: torch.Tensor) -> None:
        """Gather one batch of window rows into the static device inputs."""
        host, copied = self.host[self.staged % 2], self.copied[self.staged % 2]
        self.staged += 1
        if copied is not None:
            copied.synchronize()
        for name in host:
            source = (
                window.occupancy(idx, self.cfg.learn.occupancy_horizon)
                if name == "occupancy"
                else getattr(window, name)[idx]
            )
            host[name].copy_(source)
            self.st[name].copy_(host[name], non_blocking=True)
        if copied is not None:
            copied.record()
        self.st["flip"].copy_(torch.rand(self.cfg.learn.batch, generator=self.aug_rng, device=self.device) < 0.5)
        self.st["cycle"].copy_(torch.randint(3, (self.cfg.learn.batch,), generator=self.aug_rng, device=self.device))

    def losses(self) -> torch.Tensor:
        """Forward and backward on the staged batch; returns the metric contributions."""
        cfg, st, aug = self.cfg.learn, self.st, self.augment
        flip, k = st["flip"], st["cycle"]
        board = aug.board(st["board"], flip, k)
        moves = aug.actions(st["moves"].long(), flip)
        target = spread(moves, st["target"].float())
        tactics = unpack_tactics(spread(moves, st["tactics"]))
        action = aug.actions(st["action"].long(), flip)
        candidates = aug.actions(st["candidates"].long(), flip)
        reply = aug.actions(st["reply"].long(), flip)
        occupancy = aug.board(st["occupancy"], flip, k).long()
        planes, legal, _ = kernels.derive_batch(board, st["since_capture"].int(), st["ply"].int(), st["clock"].int())
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
        q_logits = out.q_logits.float()
        ret = st["ret"]
        ret_target = hl_gauss(ret, atoms, cfg.hl_gauss_sigma)
        played = q_logits.gather(1, action[:, None, None].expand(-1, 1, atoms)).squeeze(1)
        q_loss = -(ret_target * played.log_softmax(-1)).sum(-1).mean()
        visited = st["candidate_visited"] & target_ok[:, None]
        chosen = q_logits.gather(1, candidates[..., None].expand(-1, -1, atoms))
        cq_target = hl_gauss(st["candidate_q"].reshape(-1), atoms, cfg.hl_gauss_sigma).reshape_as(chosen)
        cq_ce = -(cq_target * chosen.log_softmax(-1)).sum(-1)
        cq_loss = (cq_ce * visited).sum() / visited.sum().clamp_min(1)
        # Every action known to win or to lose pulls its Q distribution to the corresponding atom.
        exact_ce = q_logits.logsumexp(-1)[..., None] - q_logits @ self.exact_atoms.t()
        winning = (tactics[..., 0] | tactics[..., 2]) & legal
        losing = tactics[..., 1] & legal
        exact_ok = winning | losing
        exact_loss = (torch.where(winning, exact_ce[..., 1], exact_ce[..., 0]) * exact_ok).sum()
        exact_loss = exact_loss / exact_ok.sum().clamp_min(1)
        v_ce = -(ret_target * out.v_logits.float().log_softmax(-1)).sum(-1)
        v_loss = v_ce.mean()
        replayed = st["replayed"]
        v_fresh = (v_ce * ~replayed).sum() / (~replayed).sum().clamp_min(1)
        v_replay = (v_ce * replayed).sum() / replayed.sum().clamp_min(1)
        tactics_bce = F.binary_cross_entropy_with_logits(out.tactics.float(), tactics.float(), reduction="none")
        tactics_loss = (tactics_bce.mean(-1) * legal).sum() / legal.sum().clamp_min(1)
        occ_ok = st["occupancy_ok"]
        occ_ce = F.cross_entropy(out.occupancy.float(), occupancy, reduction="none").mean(1)
        occ_loss = (occ_ce * occ_ok).sum() / occ_ok.sum().clamp_min(1)
        tte = st["plies_to_end"]
        tte_ok = tte > 0
        tte_class = (torch.searchsorted(self.tte_edges, tte.clamp_min(1), right=True) - 1).long()
        tte_ce = F.cross_entropy(out.plies_to_end.float(), tte_class, reduction="none")
        tte_loss = (tte_ce * tte_ok).sum() / tte_ok.sum().clamp_min(1)
        material_se = (out.material.float() - st["material"].float() / 10.0).pow(2).mean(1)
        material_ok = st["material_ok"]
        material_loss = (material_se * material_ok).sum() / material_ok.sum().clamp_min(1)
        if out.wdl is not None:
            # Classes win / draw / loss from the outcome +1 / 0 / -1, on rows whose game has ended.
            wdl_ok = st["outcome_ok"]
            wdl_ce = F.cross_entropy(out.wdl.float(), (1 - st["outcome"].long()), reduction="none")
            wdl_loss = (wdl_ce * wdl_ok).sum() / wdl_ok.sum().clamp_min(1)
        else:
            wdl_loss = torch.zeros((), device=policy_loss.device)
        if out.reply is not None:
            reply_ok = st["reply_ok"]
            reply_ce = F.cross_entropy(out.reply.float(), reply, reduction="none")
            reply_loss = (reply_ce * reply_ok).sum() / reply_ok.sum().clamp_min(1)
        else:
            reply_loss = torch.zeros((), device=policy_loss.device)
        if out.regret is not None:
            gamma, predicted = out.regret.float().unbind(-1)
            regret_ok, regret = st["regret_ok"], st["regret"]
            regret_loss = ((predicted - regret).pow(2) * regret_ok).sum() / regret_ok.sum().clamp_min(1)
            # Ranking (DESIGN.md item 33): -log sum_s softmax(gamma)_s exp(e_s)
            # over labelled rows alone; a wholly unlabelled batch costs zero.
            rank_loss = ranking_loss(gamma, regret, regret_ok)
        else:
            regret_loss = rank_loss = torch.zeros((), device=policy_loss.device)
        loss = (
            policy_loss
            + q_loss
            + cfg.w_candidate_q * cq_loss
            + exact_loss
            + cfg.w_value * v_loss
            + cfg.w_tactics * tactics_loss
            + cfg.w_occupancy * occ_loss
            + cfg.w_plies_to_end * tte_loss
            + cfg.w_material * material_loss
            + cfg.w_wdl * wdl_loss
            + cfg.w_reply * reply_loss
            + cfg.w_regret * regret_loss
            + cfg.w_rank * rank_loss
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
                    exact_loss,
                    v_loss,
                    tactics_loss,
                    occ_loss,
                    tte_loss,
                    material_loss,
                    entropy,
                    kl,
                    torch.zeros((), device=loss.device),
                    ret.abs().mean(),
                    torch.zeros((), device=loss.device),
                    wdl_loss.detach(),
                    reply_loss.detach(),
                    regret_loss.detach(),
                    rank_loss.detach(),
                    v_fresh,
                    v_replay,
                ]
            ).detach()

    def step(self) -> None:
        """One optimizer step on the staged batch, then the EMA update."""
        metrics = self.losses()
        metrics[11], metrics[13] = self.clip_grads()
        self.opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(self.ema_params, self.params, 1.0 - self.cfg.learn.ema)
            torch._foreach_copy_(self.ema_buffers, self.buffers)
        replayed = self.st["replayed"].sum()
        weights = torch.cat(
            [
                torch.ones(len(self.METRICS) - 2, device=metrics.device),
                (~self.st["replayed"]).sum()[None],
                replayed[None],
            ]
        )
        self.sums.add_(metrics * weights)
        self.counts.add_(weights)

    def run_step(self) -> None:
        self.set_lr()
        self.steps += 1
        if self.graph is not None:
            self.graph.replay()
        elif self.cfg.learn.graphs and self.device.type == "cuda" and self.eager_steps >= self.WARMUP:
            self.opt.zero_grad(set_to_none=True)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
                self.step()
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

    def train(self, replay: Replay, iteration: int = 0) -> dict:
        """One iteration over the replay; returns the averaged metrics."""
        self.iteration = iteration
        self.net.train()
        self.sums.zero_()
        self.counts.zero_()
        batches = 0
        for window, idx in replay.batches():
            self.stage(window, idx)
            self.run_step()
            batches += 1
        self.net.eval()
        means = (self.sums / self.counts.clamp_min(1)).tolist()
        return {**dict(zip(self.METRICS, means, strict=True)), "batches": batches}


def append_row(path: str, row: dict) -> None:
    """Append one row; a row with new columns rewrites the file under the wider header."""
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = list(reader.fieldnames or row)
        if any(k not in header for k in row):
            header = header + [k for k in row if k not in header]
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
    else:
        header = list(row)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=header).writerow(row)


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
        # The network and play settings are the checkpoint's; `--net.wdl true`
        # may turn the outcome head on for a running run (it loads fresh).
        wdl = cfg.net.wdl or saved.net.wdl
        cfg.net, cfg.play = saved.net, saved.play
        cfg.net.wdl = wdl
        if cfg.resume:
            cfg.rules = saved.rules
            start = checkpoint["iteration"] + 1
    net = build(cfg, checkpoint, device, "model")
    ema = build(cfg, checkpoint, device, "ema")
    share_fresh_heads(net, ema)
    learner = Learner(cfg, net, ema, device)
    if checkpoint is not None:
        learner.rollback_factor = float(checkpoint.get("rollback_factor", 1.0))
    replay = Replay(cfg.learn, run_dir, cfg.seed + 4)
    if cfg.resume and checkpoint.get("optimizer"):
        learner.steps = load_optimizer_state(learner.opt, net, checkpoint["optimizer"], device)
        print(f"resumed {cfg.resume} at iteration {start}; replay windows restored: {replay.restore(start - 1)}")
    elif checkpoint is not None:
        print(f"initialised from {cfg.init} (iteration {checkpoint['iteration']}); the optimizer starts fresh")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(to_dict(cfg), f, indent=2)
    print(f"params {count_params(net) / 1e6:.2f}M; device {device}; contempt draw {cfg.search.contempt_source}")
    actor = Actor(cfg, device)
    outcomes = Outcomes(cfg.learn.envs, cfg.learn.steps)
    control = SearchControl(cfg.control, cfg.learn.envs, cfg.learn.steps, device, cfg.seed + 6, cfg.control.warmup)
    if checkpoint is not None and "control" in checkpoint:
        control.load(checkpoint["control"])
    actor.control = control
    control.upload()
    net.eval()
    ema.eval()
    student = student_evaluator(compile_net(ema, cfg), cfg.play.draw_kernel_width, cfg.search.contempt_source)
    log_path = os.path.join(run_dir, "log.csv")
    good, good_policy, good_it = learner.snapshot(), None, start - 1
    for it in range(start, cfg.iters):
        t0 = time.time()
        rollout = actor.collect(student)
        window = build_window(rollout, cfg, it)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        control.observe(rollout, actor.env.origin)
        touched = outcomes.label(rollout.done, rollout.end_reason, window, replay, control)
        # Rewrite the windows that gained labels before the new one can trim them.
        for k in sorted(touched):
            replay.persist(replay.find(k))
        replay.add(window)
        control.upload()
        iteration_lr = learner.effective_base_lr(it)
        stats = learner.train(replay, it)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        diverged = (
            not math.isfinite(stats["policy"])
            or stats["grad_norm"] > cfg.learn.skip_norm
            or (good_policy is not None and stats["policy"] > cfg.learn.rollback_ratio * good_policy)
        )
        if diverged:
            learner.restore(good)
            learner.rollback_factor *= 0.5
            print(
                f"[it {it}] diverged (policy {stats['policy']:.3f} after {good_policy},"
                f" grad_norm {stats['grad_norm']:.3g}): restored the weights and optimizer"
                f" of iteration {good_it}; next lr {learner.effective_base_lr(it + 1):.2e}",
                flush=True,
            )
        else:
            good, good_policy, good_it = learner.snapshot(), stats["policy"], it
        row = {
            "iter": it,
            "t_collect": round(t1 - t0, 2),
            "t_train": round(t2 - t1, 2),
            "steps_per_s": round(cfg.learn.envs * cfg.learn.steps / max(t1 - t0, 1e-6)),
            "replay_positions": replay.positions(),
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in stats.items()},
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in conversion_metrics(rollout).items()},
            **{k: round(v, 5) for k, v in actor.search.metrics().items()},
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in control.end_iteration().items()},
            "rollback": int(diverged),
            "lr": iteration_lr,
            "material_ok_frac": round(window.material_ok.float().mean().item(), 5),
            "replayed_frac": round(window.replayed.float().mean().item(), 5),
        }
        labelled_fresh = window.regret_ok & ~window.replayed
        row["label_mean"] = round(window.regret[labelled_fresh].mean().item(), 5) if labelled_fresh.any() else 0.0
        row["label_high_frac"] = (
            round((window.regret[labelled_fresh] > 0.5).float().mean().item(), 5) if labelled_fresh.any() else 0.0
        )
        append_row(log_path, row)
        print(f"[it {it}] {row}", flush=True)
        extra = {"control": control.state(), "rollback_factor": learner.rollback_factor}
        save_checkpoint(os.path.join(run_dir, "latest.pt"), net, ema, optimizer_state(learner.opt, net), it, cfg, extra)
        if it % cfg.checkpoint_every == 0:
            save_checkpoint(
                os.path.join(run_dir, f"ckpt_{it:06d}.pt"), net, ema, optimizer_state(learner.opt, net), it, cfg, extra
            )
        replay.prune(it)


if __name__ == "__main__":
    main(parse())
