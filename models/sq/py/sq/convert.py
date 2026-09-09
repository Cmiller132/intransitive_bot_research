"""Convert a checkpoint written by the previous trainer into this package's
format: the same weights under this model's parameter names, the optimizer
moments carried over, the EMA weights initialised to the model.

    python -m sq.convert --old <old.pt> --out weights/sq_g128.pt
"""

from __future__ import annotations

import argparse
import copy

import torch

from .config import Config
from .config import Net as NetConfig
from .model import SqNet, save_checkpoint


def name_map(net_cfg: NetConfig) -> dict[str, str]:
    """Old parameter name -> new parameter name."""
    m = {"pos": "position", "stem.weight": "stem.weight"}

    def block(old: str, new: str) -> None:
        m[f"{old}.pair_w"] = f"{new}.relation"
        m[f"{old}.bias.table"] = f"{new}.position"
        for a, b in (
            ("ln1", "norm1"),
            ("ln2", "norm2"),
            ("mix_ln", "mix_norm"),
            ("dw", "mix"),
            ("qkv", "qkv"),
            ("proj", "proj"),
            ("mlp.0", "ffn_in"),
            ("mlp.2", "ffn_out"),
            ("sm_compress", "compress"),
        ):
            m[f"{old}.{a}.weight"] = f"{new}.{b}.weight"
            m[f"{old}.{a}.bias"] = f"{new}.{b}.bias"

    for i in range(net_cfg.blocks):
        block(f"trunk.{i}", f"blocks.{i}")
    block("q_block", "q_block")
    for a, b in (("dense1", "dense1"), ("ln1", "ln1"), ("dense2", "dense2"), ("ln2", "ln2")):
        m[f"smolgen.{a}.weight"] = f"smolgen.{b}.weight"
        m[f"smolgen.{a}.bias"] = f"smolgen.{b}.bias"
    m["smolgen.out.weight"] = "smolgen.out.weight"
    for old, new in (("head_bn", "head_norm"), ("q_bn", "q_norm")):
        for field in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            m[f"{old}.{field}"] = f"{new}.{field}"
    for old, new in (("policy", "policy"), ("q_out", "q_head")):
        m[f"{old}.bias"] = f"{new}.bias"
        for a in ("q", "k", "from_term"):
            m[f"{old}.{a}.weight"] = f"{new}.{a}.weight"
            m[f"{old}.{a}.bias"] = f"{new}.{a}.bias"
    m["q_out.tilt"] = "q_head.tilt"
    for old, new in (("occ_out", "occupancy"), ("reply_out", "reply"), ("danger_out", "danger")):
        m[f"{old}.weight"] = f"{new}.weight"
        m[f"{old}.bias"] = f"{new}.bias"
    for old, new in (("tte_out", "plies_to_end"), ("material_out", "material")):
        for layer in ("0", "2"):
            m[f"{old}.{layer}.weight"] = f"{new}.{layer}.weight"
            m[f"{old}.{layer}.bias"] = f"{new}.{layer}.bias"
    return m


def reshape(new_name: str, value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Old tensor layout -> new layout for the parameters whose shape changed."""
    if value.shape == target.shape:
        return value
    if new_name == "position":  # (1, C, 9, 9) -> (1, 81, C)
        return value.flatten(2).transpose(1, 2).contiguous()
    if new_name.endswith(".position"):  # (1, H, 81, 81) -> (H, 81, 81)
        return value[0].contiguous()
    if value.ndim == 4 and value.shape[2:] == (1, 1):  # 1x1 conv -> linear
        return value[:, :, 0, 0].contiguous()
    raise ValueError(f"cannot map {tuple(value.shape)} onto {new_name} {tuple(target.shape)}")


def convert(old: dict, cfg: Config) -> tuple[SqNet, dict]:
    """The network with the old weights and the optimizer state under new indices."""
    net = SqNet(cfg.net)
    names = name_map(cfg.net)
    own = net.state_dict()
    old_state = old["model"]
    missing = [k for k in old_state if k not in names]
    if missing:
        raise ValueError(f"old checkpoint has unmapped tensors: {missing[:5]}")
    covered = set()
    for old_name, value in old_state.items():
        new_name = names[old_name]
        own[new_name].copy_(reshape(new_name, value, own[new_name]))
        covered.add(new_name)
    uncovered = sorted(set(own) - covered)
    if uncovered:
        raise ValueError(f"old checkpoint leaves tensors uninitialised: {uncovered[:5]}")
    net.load_state_dict(own)

    buffers = ("running_mean", "running_var", "num_batches_tracked")
    old_params = [k for k in old_state if not k.endswith(buffers)]
    new_index = {name: i for i, (name, _) in enumerate(net.named_parameters())}
    old_opt = old["opt"]
    state = {}
    for old_i, old_name in enumerate(old_params):
        new_name = names[old_name]
        entry = dict(old_opt["state"][old_i])
        target = dict(net.named_parameters())[new_name]
        for key in ("exp_avg", "exp_avg_sq"):
            entry[key] = reshape(new_name, entry[key], target)
        state[new_index[new_name]] = entry
    optimizer = {"state": state, "param_groups": [{"params": list(range(len(new_index)))}]}
    return net, optimizer


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--old", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    old = torch.load(args.old, map_location="cpu", weights_only=False)
    cfg = Config()
    old_cfg = old["cfg"]
    cfg.play.alpha, cfg.play.beta = float(old_cfg["alpha"]), float(old_cfg["beta"])
    net, optimizer = convert(old, cfg)
    save_checkpoint(args.out, net, copy.deepcopy(net), optimizer, int(old["iter"]), cfg)
    print(
        f"converted {args.old} (iteration {old['iter']}) -> {args.out}: "
        f"{sum(p.numel() for p in net.parameters())} parameters, {len(optimizer['state'])} optimizer entries"
    )


if __name__ == "__main__":
    main()
