"""Every tunable of the sq model and its training, with the DESIGN.md values as
defaults. `train` parses these from the command line; a run's resolved config is
saved beside its checkpoints and inside every checkpoint."""

from __future__ import annotations

import argparse
import math
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass


@dataclass
class Net:
    planes: int = 25
    width: int = 128
    blocks: int = 16
    heads: int = 4
    ffn: int = 512
    smolgen_every: int = 3
    smolgen_dim: int = 8
    smolgen_hidden: int = 64
    smolgen_latent: int = 16
    atoms: int = 51
    head_dim: int = 64


@dataclass
class Search:
    sims: int = 128
    candidates: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    cheap_sims: int = 16
    cheap_candidates: int = 4
    full_fraction: float = 0.25
    repetition_draw: bool = True
    temperature: float = 1.5
    temperature_plies: int = 10
    reuse_nodes: int = 64


@dataclass
class Rules:
    capture_clock: int = 75
    clock_penalty: float = 0.05
    max_plies: int = 1000


@dataclass
class Learn:
    envs: int = 1024
    steps: int = 128
    replay_windows: int = 4
    epochs: int = 2
    batch: int = 768
    lr: float = 5e-4
    weight_decay: float = 1e-4
    ema: float = 0.999
    grad_clip: float = 10.0
    lam: float = math.exp(-1 / 8)
    hl_gauss_sigma: float = 0.75
    w_candidate_q: float = 1.0
    w_occupancy: float = 0.5
    w_plies_to_end: float = 0.25
    w_reply: float = 0.25
    w_danger: float = 0.25
    w_material: float = 0.25
    occupancy_horizon: int = 16
    danger_horizon: int = 8
    compile: bool = True
    graphs: bool = True


@dataclass
class Play:
    """Constants the exported model carries for the Rust player."""

    alpha: float = 0.0032981350479294397
    beta: float = 0.00989440514378832
    draw_kernel_width: float = 0.15


@dataclass
class Config:
    run: str = "sq"
    init: str = ""
    resume: str = ""
    iters: int = 100000
    checkpoint_every: int = 5
    seed: int = 0
    net: Net = field(default_factory=Net)
    search: Search = field(default_factory=Search)
    rules: Rules = field(default_factory=Rules)
    learn: Learn = field(default_factory=Learn)
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
    parser = argparse.ArgumentParser(description="sq training")
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
