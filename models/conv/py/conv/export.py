"""Export a checkpoint for the Rust player.

    python -m conv.export --ckpt runs/<name>/latest.pt --out runs/<name>/export.onnx [--weights ema]

Writes `<out>` (input `planes` f32 [B, 46, 81]; outputs `logits` [B, 648],
`q` [B, 648], `value` [B], `plies_to_end` [B, 16], `draw` [B, 648]; dynamic
batch; plus `wdl` [B, 3] when enabled; opset 17) and `<out>.json` with the
fields `src/net.rs` reads. Then checks ONNX Runtime against torch on random
positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from .model import TTE_CENTERS, ConvNet, build, config_of, load_checkpoint

OUTPUTS = ("logits", "q", "value", "plies_to_end", "draw")
TOLERANCES = (2e-3, 5e-4, 5e-4, 2e-3, 5e-4)
# With the WDL head (Net.wdl): its win / draw / loss probabilities as a sixth output.
WDL_OUTPUT, WDL_TOLERANCE = "wdl", 5e-4


def outputs_of(net: ConvNet) -> tuple[str, ...]:
    return OUTPUTS + ((WDL_OUTPUT,) if net.wdl is not None else ())


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Exported(torch.nn.Module):
    """The network plus each action's draw mass: its value distribution weighted
    by `exp(-(atom / width)^2)`."""

    def __init__(self, net: ConvNet, width: float):
        super().__init__()
        self.net = net
        # Opset 17 does not portably lower SDPA with an arbitrary float mask.
        # The mathematically equivalent explicit path consists only of stable
        # ONNX operators and is also used for the torch parity reference.
        for block in self.net.attn_blocks:
            block.fused_attention = False
        self.register_buffer("kernel", torch.exp(-((net.atoms / width) ** 2)))

    def forward(self, planes):
        out = self.net(planes, aux=True)
        draw = torch.softmax(out.q_logits.float(), -1) @ self.kernel
        heads = (out.logits, out.q, out.v, out.plies_to_end, draw)
        if out.wdl is not None:
            heads += (torch.softmax(out.wdl.float(), -1),)
        return heads


def export(ckpt: str, out: str, weights: str = "ema") -> dict:
    checkpoint = load_checkpoint(ckpt)
    cfg = config_of(checkpoint)
    net = build(cfg, checkpoint, "cpu", weights).eval()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    dummy = torch.zeros(4, cfg.net.planes, 81)
    with torch.no_grad():
        torch.onnx.export(
            Exported(net, cfg.play.draw_kernel_width),
            (dummy,),
            out,
            input_names=["planes"],
            output_names=list(outputs_of(net)),
            dynamic_axes={name: {0: "b"} for name in ("planes", *outputs_of(net))},
            opset_version=17,
            dynamo=False,
        )
    run = os.path.basename(os.path.dirname(os.path.abspath(ckpt)))
    meta = {
        "name": f"{run}@{checkpoint['iteration']}",
        "checkpoint_sha256": sha256(ckpt),
        "weights": weights,
        "iteration": checkpoint["iteration"],
        "planes": cfg.net.planes,
        "atoms": cfg.net.atoms,
        "alpha": cfg.play.alpha,
        "beta": cfg.play.beta,
        "draw_kernel_width": cfg.play.draw_kernel_width,
        "wdl": net.wdl is not None,
        "tte_centers": TTE_CENTERS,
        "onnx_sha256": sha256(out),
    }
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def check(ckpt: str, out: str, weights: str = "ema", positions: int = 64) -> dict:
    """Max deviation of ONNX Runtime from torch on random positions; raises beyond tolerance."""
    import onnxruntime as ort

    from .planes import random_positions, reference

    checkpoint = load_checkpoint(ckpt)
    cfg = config_of(checkpoint)
    net = build(cfg, checkpoint, "cpu", weights).eval()
    encoded = [reference(b, sc, p, 200) for b, sc, p in random_positions(positions, seed=7)]
    planes = torch.tensor(np.stack(encoded))
    with torch.no_grad():
        ref = Exported(net, cfg.play.draw_kernel_width)(planes)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(out, options, providers=["CPUExecutionProvider"])
    got = session.run(None, {"planes": planes.numpy()})
    return deviations(ref, got)


def deviations(expected, actual) -> dict:
    """Max |onnx - torch| per output; raises past the tolerance or on a non-finite value."""
    deltas = {}
    names, tolerances = OUTPUTS, TOLERANCES
    if len(expected) > len(OUTPUTS):
        names, tolerances = names + (WDL_OUTPUT,), tolerances + (WDL_TOLERANCE,)
    for name, want, got, tol in zip(names, expected, actual, tolerances, strict=True):
        delta = float(np.abs(want.numpy() - got).max())
        deltas[name] = delta
        if not np.isfinite(got).all() or delta > tol:
            raise RuntimeError(f"ONNX parity failed for {name}: max error {delta:.3g} > {tol}")
    return deltas


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", default="ema", choices=("ema", "model"))
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    meta = export(args.ckpt, args.out, args.weights)
    deltas = check(args.ckpt, args.out, args.weights)
    print(
        f"exported {args.out} ({os.path.getsize(args.out) // 1024} KB) {meta['name']}; "
        "max |onnx - torch| " + " ".join(f"{k} {v:.1e}" for k, v in deltas.items())
    )


if __name__ == "__main__":
    main()
