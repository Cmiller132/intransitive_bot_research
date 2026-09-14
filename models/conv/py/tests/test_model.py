"""The conv network: forward shapes with and without the training-only heads,
the parameter budget of DESIGN.md, the checkpoint round trip and ONNX Runtime
parity with torch. Runs on the CPU."""

import os

import onnxruntime as ort
import torch

from conv.config import Config, Net, to_dict
from conv.export import Exported, deviations, export, outputs_of
from conv.model import (
    N_OCC,
    N_TACTICS,
    N_TTE,
    AttnBlock,
    ConvNet,
    build,
    config_of,
    count_params,
    load_checkpoint,
    offset_index,
    save_checkpoint,
)
from conv.planes import N_ACTIONS, N_PLANES, N_SQUARES


def tiny_config() -> Config:
    """A few thousand parameters: two pairs, Smolgen on the second block only."""
    cfg = Config(run="test", seed=3)
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
        value_squares=4,
        value_hidden=16,
    )
    return cfg


def test_forward_shapes():
    cfg = tiny_config()
    net = ConvNet(cfg.net).eval()
    planes = torch.rand(3, N_PLANES, N_SQUARES)
    with torch.no_grad():
        plain = net(planes)
        full = net(planes, aux=True)
    assert plain.logits.shape == (3, N_ACTIONS)
    assert plain.q_logits.shape == (3, N_ACTIONS, cfg.net.atoms)
    assert plain.q.shape == (3, N_ACTIONS) and plain.q.dtype == torch.float32
    assert plain.v_logits.shape == (3, cfg.net.atoms)
    assert plain.v.shape == (3,) and plain.v.abs().max() <= 1.0
    assert plain.tactics is None and plain.occupancy is None
    assert plain.plies_to_end is None and plain.material is None
    assert plain.wdl.shape == (3, 3) and torch.equal(plain.wdl, full.wdl)
    assert full.tactics.shape == (3, N_ACTIONS, N_TACTICS)
    assert full.occupancy.shape == (3, N_OCC, N_SQUARES)
    assert full.plies_to_end.shape == (3, N_TTE) and full.material.shape == (3, 2)
    assert plain.reply is None and full.reply.shape == (3, N_ACTIONS)
    assert plain.regret.shape == (3, 2) and torch.equal(plain.regret, full.regret)
    assert torch.equal(plain.logits, full.logits) and torch.equal(plain.v, full.v)
    assert all(torch.isfinite(t).all() for t in full if t is not None)


def test_qk_norm_bounds_the_attention_logits():
    """Normalised queries and keys keep every scaled dot product within
    sqrt(head_dim) times the product of the learned scales."""
    from conv.model import rms_normalize

    x = torch.randn(3, 5, 81, 32) * 50.0
    y = rms_normalize(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(3, 5, 81), atol=1e-4)
    att = torch.matmul(y, y.transpose(-2, -1)) * 32**-0.5
    assert att.abs().max() <= 32**0.5 + 1e-4


def test_fused_attention_matches_explicit_attention():
    cfg = tiny_config().net
    block = AttnBlock(cfg, smolgen=False).eval()
    with torch.no_grad():
        block.position.normal_(std=0.1)
        block.relation.normal_(std=0.1)
        x = torch.randn(2, N_SQUARES, cfg.width)
        relations = torch.randint(0, 1 << 7, (2, N_SQUARES, N_SQUARES), dtype=torch.uint8)
        block.fused_attention = True
        fused = block(x, relations, offset_index(), None)
        block.fused_attention = False
        explicit = block(x, relations, offset_index(), None)
    assert torch.allclose(fused, explicit, atol=1e-4, rtol=1e-4)


def test_default_parameter_budget():
    cfg = Config().net
    assert (cfg.width, cfg.heads, cfg.ffn, cfg.pairs) == (160, 5, 640, 6)
    assert cfg.head_dim == 64 and cfg.value_squares == 32 and cfg.value_hidden == 256
    assert cfg.wdl and cfg.fused_attention
    assert count_params(ConvNet(cfg)) == 6_031_492


def test_checkpoint_round_trip(tmp_path):
    cfg = tiny_config()
    net, ema = ConvNet(cfg.net), ConvNet(cfg.net)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, net, ema, None, 7, cfg)
    checkpoint = load_checkpoint(path)
    assert checkpoint["iteration"] == 7 and config_of(checkpoint) == cfg
    again = build(config_of(checkpoint), checkpoint, "cpu", "model")
    for a, b in zip(again.state_dict().values(), net.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_checkpoint_without_optional_heads_loads_them_fresh(tmp_path):
    cfg = tiny_config()
    cfg.net.wdl = cfg.net.reply = cfg.net.regret = False
    net = ConvNet(cfg.net)
    path = str(tmp_path / "old.pt")
    save_checkpoint(path, net, net, None, 0, cfg)
    cfg.net.wdl = cfg.net.reply = cfg.net.regret = True
    loaded = build(cfg, load_checkpoint(path), "cpu", "model")
    assert loaded.fresh
    assert all(key.startswith(("wdl.", "reply.", "regret.")) for key in loaded.fresh)


def test_export_matches_torch(tmp_path):
    cfg = tiny_config()
    net = ConvNet(cfg.net)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, net, net, None, 0, cfg)
    out = str(tmp_path / "model.onnx")
    meta = export(path, out, "ema")
    assert meta["planes"] == N_PLANES and meta["atoms"] == cfg.net.atoms
    assert os.path.exists(out + ".json")

    checkpoint = load_checkpoint(path)
    torch_net = build(config_of(checkpoint), checkpoint, "cpu", "ema").eval()
    planes = torch.rand(5, N_PLANES, N_SQUARES)
    exported = Exported(torch_net, cfg.play.draw_kernel_width)
    assert not any(block.fused_attention for block in torch_net.attn_blocks)
    with torch.no_grad():
        expected = exported(planes)
    session = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    got = session.run(None, {"planes": planes.numpy()})
    deltas = deviations(expected, got)
    assert tuple(deltas) == outputs_of(torch_net) and max(deltas.values()) < 2e-3


def test_pooled_state_head_loads_old_checkpoints():
    """A checkpoint config without `state_head` (every conv_g128 checkpoint)
    builds the pooled mean + a1 + i9 heads with the old parameter names, so
    the distillation teacher and the arena export still load."""
    old = Net(width=112, heads=4, ffn=448, wdl=True)
    old_cfg = Config(net=old)
    saved = to_dict(old_cfg)
    del saved["net"]["state_head"]
    cfg = config_of({"config": saved})
    assert cfg.net.state_head == "pooled"
    net = build(cfg, None, "cpu")
    assert net.state_encoder is None
    keys = set(net.state_dict())
    assert {"value.0.weight", "value.2.weight", "wdl.0.weight", "wdl.2.weight", "regret.0.weight"} <= keys
    assert net.value[0].weight.shape == (256, 336)
    payload = {"model": net.state_dict(), "ema": net.state_dict(), "config": to_dict(cfg)}
    again = build(config_of(payload), payload, "cpu", "ema")
    out = again(torch.rand(2, 46, 81), aux=True)
    assert out.wdl.shape == (2, 3) and out.regret.shape == (2, 2) and out.v.shape == (2,)
