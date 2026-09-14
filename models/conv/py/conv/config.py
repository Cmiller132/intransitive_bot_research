"""Every tunable of the conv model and its training, with the DESIGN.md values as
defaults. `train` parses these from the command line; a run's resolved config is
saved beside its checkpoints and inside every checkpoint."""

from __future__ import annotations

import argparse
import math
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass


@dataclass
class Net:
    planes: int = 46
    width: int = 160
    # Trunk pairs, each a convolution block then an attention block.
    pairs: int = 6
    heads: int = 5
    ffn: int = 640
    # QK-norm bounds the logits so SDPA can safely use bf16 under CUDA autocast.
    fused_attention: bool = True
    # Smolgen runs on the attention blocks with `i % smolgen_every == smolgen_first`.
    smolgen_every: int = 3
    smolgen_first: int = 1
    smolgen_dim: int = 8
    smolgen_hidden: int = 64
    smolgen_latent: int = 16
    atoms: int = 51
    head_dim: int = 64
    value_squares: int = 32
    value_hidden: int = 256
    # State features behind V, WDL and regret (DESIGN.md item 10): "spatial"
    # (per-square projection, flatten, dense 256; 2026-09-10) or "pooled" (the
    # older mean + a1 + i9, which checkpoints without this field carry).
    state_head: str = "spatial"
    # Win/draw/loss head trained on final game outcomes (DESIGN.md item 31).
    wdl: bool = True
    # Opponent-reply policy head (DESIGN.md item 32) and the regret ranking
    # and value head of the search control (item 33); a checkpoint without
    # them loads with the heads fresh.
    reply: bool = True
    regret: bool = True


@dataclass
class Search:
    sims: int = 128
    candidates: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    cheap_sims: int = 16
    cheap_candidates: int = 4
    # Share of steps searched with `sims` and `candidates`; the rest use the cheap sizes.
    full_fraction: float = 0.5
    repetition_draw: bool = True
    # A rule draw is worth `-contempt` to the side to move at the root: the move
    # choice and the policy target charge every move `contempt` times its draw
    # mass, while the Q and V labels stay zero-sum (DESIGN.md item 17).
    contempt: float = 0.05
    # `wdl` uses the state outcome head; `q` keeps the per-action Q-mass proxy.
    contempt_source: str = "wdl"
    temperature: float = 1.5
    temperature_plies: int = 10
    reuse_nodes: int = 64


@dataclass
class Rules:
    """The capture clock is drawn per game from `[clock_min, clock_max]` plies."""

    clock_min: int = 50
    clock_max: int = 200
    clock_penalty: float = 0.05
    max_plies: int = 1000


@dataclass
class Learn:
    envs: int = 1024
    steps: int = 192
    # Replay window in iterations (one window of `steps x envs` positions
    # each), grown as a power law of the windows produced so far the way
    # KataGo grows its window: `window_min` at first, then
    # min + (n^e - min^e) / (e min^(e-1)) x per_iter, capped at `window_max`.
    window_min: int = 4
    window_max: int = 16
    window_exponent: float = 0.75
    window_per_iter: float = 0.4
    # Learner steps per iteration, each a batch drawn uniformly from the window.
    batches: int = 1368
    batch: int = 768
    lr: float = 5e-4
    # Hold the base rate through this iteration, then cosine-decay to `lr_min`
    # at `lr_decay_iters` and hold it there (DESIGN.md item 25).
    lr_hold_iters: int = 40
    lr_decay_iters: int = 120
    lr_min: float = 1e-4
    weight_decay: float = 1e-4
    ema: float = 0.999
    grad_clip: float = 10.0
    # Learning-rate multiplier of the dense 3x3 convolution weights (stem and
    # convolution blocks): Adam moves every weight by about the learning rate,
    # so a layer's pre-activations drift in proportion to its fan-in, and the
    # convolutions' fan-in is nine times the attention layers'.
    conv_lr_scale: float = 1 / 3
    # Linear warm-up of the learning rate over the first optimizer steps.
    warmup_steps: int = 500
    # A batch whose gradient norm is not finite or exceeds this is skipped.
    skip_norm: float = 1000.0
    # An iteration whose policy loss exceeds the last good one by this factor
    # (or whose mean gradient norm exceeds `skip_norm`) is rolled back and the
    # learning rate halved.
    rollback_ratio: float = 1.5
    lam: float = math.exp(-1 / 8)
    hl_gauss_sigma: float = 0.75
    w_candidate_q: float = 1.0
    w_value: float = 1.0
    w_tactics: float = 0.25
    w_occupancy: float = 0.5
    w_plies_to_end: float = 0.25
    w_material: float = 0.25
    w_wdl: float = 0.5
    w_reply: float = 0.15
    w_regret: float = 0.25
    w_rank: float = 0.25
    occupancy_horizon: int = 16
    compile: bool = True
    graphs: bool = True


@dataclass
class Control:
    """Regret-guided search control (DESIGN.md item 33): a finished board
    restarts from an opening of the prioritised regret buffer with
    probability `restart`, drawn in proportion to priority to the power
    `1 / temperature`; a replayed opening's excess moves toward each new
    game's by `ema`, and from two games on its priority is the corrected
    squared mean residual. The paper's 0.1 was tuned for per-game resampling;
    a whole window draws from one copy here, so 0.5 spreads the draws over
    the entries instead of the top few."""

    restart: float = 0.5
    temperature: float = 0.5
    # No entry takes more than this share of the restarts drawn from one
    # copy of the buffer, which a window's draws all come from.
    share: float = 1 / 16
    capacity: int = 512
    ema: float = 0.5
    # Iterations during which nothing restarts from a newly built buffer.
    warmup: int = 5


@dataclass
class Distil:
    """Direct distillation from the reference before self-play (`conv.distil`)."""

    teacher: str = "conv"
    ckpt: str = "runs/conv_g128/ckpt_000150.pt"
    envs: int = 1024
    # Training steps, one batch of `envs` positions each.
    steps: int = 3000
    lr: float = 1e-3
    # Softmax temperature of the reference's move sampling.
    temperature: float = 1.0
    log_every: int = 100


@dataclass
class Play:
    """Constants the exported model carries for the Rust player."""

    alpha: float = 0.0032981350479294397
    beta: float = 0.00989440514378832
    draw_kernel_width: float = 0.15


@dataclass
class Config:
    run: str = "conv"
    init: str = ""
    resume: str = ""
    iters: int = 100000
    checkpoint_every: int = 5
    seed: int = 0
    net: Net = field(default_factory=Net)
    search: Search = field(default_factory=Search)
    rules: Rules = field(default_factory=Rules)
    learn: Learn = field(default_factory=Learn)
    control: Control = field(default_factory=Control)
    distil: Distil = field(default_factory=Distil)
    play: Play = field(default_factory=Play)


def _flags(section, prefix: str) -> list[tuple[str, type, object]]:
    out = []
    for f in fields(section):
        value = getattr(section, f.name)
        if is_dataclass(value):
            out.extend(_flags(value, f"{prefix}{f.name}."))
        else:
            out.append((f"{prefix}{f.name}", type(value), value))
    return out


def _parse_bool(text: str) -> bool:
    if text.lower() in ("1", "true", "yes", "on"):
        return True
    if text.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")


def parse(argv: list[str] | None = None) -> Config:
    """Command line flags `--<section>.<field>` for nested sections, plain
    `--<field>` for the top level."""
    cfg = Config()
    parser = argparse.ArgumentParser(description="conv training")
    for name, kind, default in _flags(cfg, ""):
        parser.add_argument(
            f"--{name}", type=_parse_bool if kind is bool else kind, default=None, help=f"default {default}"
        )
    args = vars(parser.parse_args(argv))
    for name, value in args.items():
        if value is None:
            continue
        target = cfg
        parts = name.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return cfg


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)


def from_dict(d: dict) -> Config:
    def build(kind, values):
        kwargs = {}
        for f in fields(kind):
            if f.name not in values:
                continue
            default = f.default_factory() if f.default_factory is not MISSING else None
            kwargs[f.name] = build(type(default), values[f.name]) if is_dataclass(default) else values[f.name]
        return kind(**kwargs)

    return build(Config, d)
