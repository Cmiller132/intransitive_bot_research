"""Direct distillation from a conv or sq reference (DESIGN.md item 26). The
reference plays games with its raw policy on the batched environment; a conv
teacher supplies its full policy, categorical Q, categorical V and WDL
distributions, while the legacy sq path supplies policy, scalar Q and its
policy-weighted Q as V. Exact tactics always come from the current clock-aware
kernels. There is no search: each position streams through once, then the games
advance one ply.

    python -m conv.distil --run <name> [--distil.teacher conv|sq] [--distil.steps N]

Writes `<workspace root>/runs/<run>/distil.pt`, a checkpoint at iteration 0
for `conv.train --init`, and `distil.csv` with one row per logged step. `sq`
is imported only when it is the selected teacher."""

from __future__ import annotations

import json
import os
import time

import torch
import torch.nn.functional as F

from . import kernels, paths
from .config import Config, parse, to_dict
from .env import Env
from .model import ConvNet, build, config_of, count_params, hl_gauss, load_checkpoint, save_checkpoint
from .search import pack_tactics, raw_policy, unpack_tactics
from .train import append_row, compile_net


def sq_policy(path: str, cfg: Config, device):
    """The sq checkpoint's policy and scalar Q on its own 25 planes; V and
    WDL are absent and returned as `None`."""
    import sq.kernels
    from sq.model import build as sq_build
    from sq.model import config_of as sq_config_of
    from sq.model import load_checkpoint as sq_load_checkpoint

    checkpoint = sq_load_checkpoint(path, device)
    net = sq_build(sq_config_of(checkpoint), checkpoint, device, "ema").eval()
    forward = compile_net(net, cfg)

    @torch.no_grad()
    def play(board, since, ply, clock):
        planes, _, _ = sq.kernels.derive_batch(board, since, ply)
        if planes.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = forward(planes)
        else:
            out = forward(planes.float())
        return out.logits.float(), out.q.float(), None, None

    return play


def conv_policy(path: str, cfg: Config, device):
    """The EMA conv teacher on its own checkpoint config and the same 46
    derived planes the student sees; returns its full target distributions."""
    checkpoint = load_checkpoint(path, device)
    teacher_cfg = config_of(checkpoint)
    net = build(teacher_cfg, checkpoint, device, "ema").eval()
    # Compiled under the run's setting, not the teacher checkpoint's.
    forward = compile_net(net, cfg)

    @torch.no_grad()
    def play(board, since, ply, clock):
        planes, _, _ = kernels.derive_batch(board, since, ply, clock)
        if planes.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = forward(planes)
        else:
            out = forward(planes.float())
        if out.wdl is None:
            raise ValueError("conv distillation requires a teacher checkpoint with Net.wdl")
        return (
            out.logits.float(),
            torch.softmax(out.q_logits.float(), -1),
            torch.softmax(out.v_logits.float(), -1),
            torch.softmax(out.wdl.float(), -1),
        )

    return play


class Distiller:
    """Owns the environment, the reference, the student and its optimizer;
    one `step` trains on the positions in play, then advances every game by a
    move sampled from the reference."""

    METRICS = ("policy", "q", "value", "wdl", "tactics", "kl", "teacher_entropy")

    def __init__(self, cfg: Config, net: ConvNet, ema: ConvNet, device):
        self.cfg, self.net, self.ema = cfg, net, ema
        self.device = torch.device(device)
        d = cfg.distil
        self.env = Env(d.envs, device, cfg.rules.max_plies, cfg.rules.clock_min, cfg.rules.clock_max, cfg.seed + 5)
        path = d.ckpt if os.path.isabs(d.ckpt) else str(paths.workspace_root() / d.ckpt)
        if d.teacher == "conv":
            self.play = conv_policy(path, cfg, device)
        elif d.teacher == "sq":
            self.play = sq_policy(path, cfg, device)
        else:
            raise ValueError(f"unknown distillation teacher {d.teacher!r}")
        self.distribution_teacher = d.teacher == "conv"
        self.forward = compile_net(net, cfg)
        decay = [p for p in net.parameters() if p.ndim >= 2]
        no_decay = [p for p in net.parameters() if p.ndim < 2]
        self.opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.learn.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=d.lr,
            betas=(0.9, 0.999),
            fused=self.device.type == "cuda",
        )
        self.params, self.ema_params = list(net.parameters()), list(ema.parameters())
        self.rng = torch.Generator(device=device).manual_seed(cfg.seed + 6)
        n = d.envs
        self.win_flags = torch.zeros(n, 648, dtype=torch.bool, device=device)
        self.loss_flags = torch.zeros(n, 648, dtype=torch.int8, device=device)
        self.win3_flags = torch.zeros(n, 648, dtype=torch.int8, device=device)
        self.sums = torch.zeros(len(self.METRICS), device=device)
        self.count = 0

    def labels(self, board, since, clock, legal):
        """Packed exact tactics of every board over the 648 actions, under its
        capture clock."""
        kernels.exact_wins(board, legal, self.win_flags)
        kernels.loses_in_two(board, legal, since, clock, self.loss_flags)
        kernels.wins_in_three(board, legal, since, clock, self.win3_flags)
        win1 = self.win_flags & legal
        return pack_tactics(win1, self.loss_flags.bool() & legal, self.win3_flags.bool() & legal)

    def losses(self, planes, legal, logits_t, q_t, v_t, wdl_t, tactics) -> torch.Tensor:
        """Student against the reference on one batch; returns the metrics."""
        cfg = self.cfg.learn
        if planes.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                out = self.forward(planes, aux=True)
        else:
            out = self.forward(planes.float(), aux=True)
        p_t, logp_t = raw_policy(logits_t, legal)
        _, logp_s = raw_policy(out.logits, legal)
        logp_t = logp_t.masked_fill(~legal, 0.0)
        logp_s = logp_s.masked_fill(~legal, 0.0)
        kl = (p_t * (logp_t - logp_s)).sum(-1).mean()
        policy_loss = -(p_t * logp_s).sum(-1).mean()
        atoms = out.q_logits.shape[-1]
        q_logits = out.q_logits.float()
        if self.distribution_teacher:
            if q_t.shape != q_logits.shape or v_t.shape != out.v_logits.shape:
                raise ValueError("conv teacher and student must use the same Q and V atoms")
            q_target = q_t
        else:
            q_target = hl_gauss(q_t.reshape(-1), atoms, cfg.hl_gauss_sigma).reshape_as(q_logits)
        q_ce = -(q_target * q_logits.log_softmax(-1)).sum(-1)
        q_loss = (q_ce * legal).sum() / legal.sum().clamp_min(1)
        if self.distribution_teacher:
            v_target = v_t
        else:
            v_target = hl_gauss((p_t * q_t).sum(-1), atoms, cfg.hl_gauss_sigma)
        v_loss = -(v_target * out.v_logits.float().log_softmax(-1)).sum(-1).mean()
        if self.distribution_teacher:
            if out.wdl is None or wdl_t.shape != out.wdl.shape:
                raise ValueError("conv teacher and student must both have matching WDL heads")
            wdl_loss = -(wdl_t * out.wdl.float().log_softmax(-1)).sum(-1).mean()
        else:
            wdl_loss = torch.zeros((), device=policy_loss.device)
        tactics_bce = F.binary_cross_entropy_with_logits(out.tactics.float(), tactics.float(), reduction="none")
        tactics_loss = (tactics_bce.mean(-1) * legal).sum() / legal.sum().clamp_min(1)
        loss = policy_loss + q_loss + cfg.w_value * v_loss + cfg.w_wdl * wdl_loss + cfg.w_tactics * tactics_loss
        loss.backward()
        with torch.no_grad():
            entropy = -(p_t * logp_t).sum(-1).mean()
            return torch.stack([policy_loss, q_loss, v_loss, wdl_loss, tactics_loss, kl, entropy]).detach()

    def step(self) -> None:
        """Train on the positions in play, then advance every game one ply."""
        env = self.env
        board, since, ply = env.board.clone(), env.since_capture.clone(), env.ply.clone()
        logits_t, q_t, v_t, wdl_t = self.play(board, since, ply, env.clock)
        planes, legal, _ = kernels.derive_batch(board, since, ply, env.clock)
        tactics = unpack_tactics(self.labels(board, since, env.clock, legal))
        self.net.train()
        self.opt.zero_grad(set_to_none=True)
        metrics = self.losses(planes, legal, logits_t, q_t, v_t, wdl_t, tactics)
        torch.nn.utils.clip_grad_norm_(self.params, self.cfg.learn.grad_clip)
        self.opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(self.ema_params, self.params, 1.0 - self.cfg.learn.ema)
        self.net.eval()
        self.sums.add_(metrics)
        self.count += 1
        # The reference's move: sampled from its policy at the distillation temperature.
        p_t, _ = raw_policy(logits_t / self.cfg.distil.temperature, legal)
        action = torch.multinomial(p_t, 1, generator=self.rng).squeeze(1)
        env.step(action.to(torch.int32))

    def metrics(self) -> dict:
        means = (self.sums / max(self.count, 1)).tolist()
        self.sums.zero_()
        self.count = 0
        return dict(zip(self.METRICS, means, strict=True))


def main(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = str(paths.run_dir(cfg.run))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "distil_config.json"), "w") as f:
        json.dump(to_dict(cfg), f, indent=2)
    net = build(cfg, None, device, "model")
    ema = build(cfg, None, device, "model")
    ema.load_state_dict(net.state_dict())
    ema.eval()
    distiller = Distiller(cfg, net, ema, device)
    print(f"params {count_params(net) / 1e6:.2f}M; device {device}; reference {cfg.distil.teacher}:{cfg.distil.ckpt}")
    log_path = os.path.join(run_dir, "distil.csv")
    out_path = os.path.join(run_dir, "distil.pt")
    t0 = time.time()
    for step in range(1, cfg.distil.steps + 1):
        distiller.step()
        if step % cfg.distil.log_every == 0 or step == cfg.distil.steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            row = {
                "step": step,
                "positions": step * cfg.distil.envs,
                "t": round(time.time() - t0, 1),
                **{k: round(v, 5) for k, v in distiller.metrics().items()},
            }
            append_row(log_path, row)
            print(f"[distil {step}] {row}", flush=True)
            save_checkpoint(out_path, net, ema, None, 0, cfg)


if __name__ == "__main__":
    main(parse())
