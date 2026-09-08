"""Export a checkpoint for the Rust player.

    python -m sq.export --ckpt runs/<name>/latest.pt --out runs/<name>/export.onnx [--weights ema]

Writes `<out>` (input `planes` f32 [B, 25, 81]; outputs `logits` [B, 648],
`q` [B, 648], `plies_to_end` [B, 16], `draw` [B, 648]; dynamic batch; opset 17)
and `<out>.json` with the fields `src/net.rs` reads. Then checks ONNX Runtime
against torch on random positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from .model import TTE_CENTERS, SqNet, build, config_of, load_checkpoint
from .planes import random_positions, reference


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Exported(torch.nn.Module):
    """The network plus each action's draw mass: its value distribution weighted
    by `exp(-(atom / width)^2)`."""

    def __init__(self, net: SqNet, width: float):
        super().__init__()
        self.net = net
        self.register_buffer("kernel", torch.exp(-((net.atoms / width) ** 2)))

    def forward(self, planes):
        out = self.net(planes, aux=True)
        draw = torch.softmax(out.q_logits.float(), -1) @ self.kernel
        return out.logits, out.q, out.plies_to_end, draw


def export(ckpt: str, out: str, weights: str = "ema") -> dict:
    checkpoint = load_checkpoint(ckpt)
    cfg = config_of(checkpoint)
    net = build(cfg, checkpoint, "cpu", weights).eval()
    net.bf16_stream = False
    net.set_explicit_attention(True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    dummy = torch.zeros(4, cfg.net.planes, 81)
    with torch.no_grad():
        torch.onnx.export(
            Exported(net, cfg.play.draw_kernel_width),
            (dummy,),
            out,
            input_names=["planes"],
            output_names=["logits", "q", "plies_to_end", "draw"],
            dynamic_axes={name: {0: "b"} for name in ("planes", "logits", "q", "plies_to_end", "draw")},
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
        "tte_centers": TTE_CENTERS,
        "onnx_sha256": sha256(out),
    }
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def check(ckpt: str, out: str, weights: str = "ema", positions: int = 64) -> dict:
    """Max deviation of ONNX Runtime from torch on random positions; raises beyond tolerance."""
    import onnxruntime as ort

    checkpoint = load_checkpoint(ckpt)
    cfg = config_of(checkpoint)
    net = build(cfg, checkpoint, "cpu", weights).eval()
    net.bf16_stream = False
    planes = torch.tensor(np.stack([reference(b, sc, p) for b, sc, p in random_positions(positions, seed=7)]))
    with torch.no_grad():
        ref = Exported(net, cfg.play.draw_kernel_width)(planes)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(out, options, providers=["CPUExecutionProvider"])
    got = session.run(None, {"planes": planes.numpy()})
    deltas = {}
    for name, expected, actual, tol in zip(
        ("logits", "q", "plies_to_end", "draw"), ref, got, (2e-3, 5e-4, 2e-3, 5e-4), strict=True
    ):
        delta = float(np.abs(expected.numpy() - actual).max())
        deltas[name] = delta
        if not np.isfinite(actual).all() or delta > tol:
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
