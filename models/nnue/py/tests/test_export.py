"""The exported file against the fake-quantised forward pass, the round trip
through `load`, the widenings and the version 6 to 8 and 8 to 9 conversions."""

import struct

import numpy as np
import pytest
import torch

from nnue.export import HEADER, convert, export, file_size, integer_eval, load, read
from nnue.features import (
    CONTEXTS,
    FORMAT8,
    FORMAT9,
    HEADS,
    PIECE_ROWS,
    context,
    feature_ids,
    feature_ids9,
    piece_bucket,
)
from nnue.model import NNUE

from .test_features import random_boards


def positions(seed, count):
    boards = random_boards(seed, count, 2, 21)
    rng = np.random.default_rng(seed + 1)
    clock = rng.integers(50, 201, count)
    since = (rng.random(count) * clock).astype(np.int64)
    return boards, since, clock


def trained_like(net, seed):
    """A network whose residual head and context rows are not at their zero start."""
    torch.manual_seed(seed)
    with torch.no_grad():
        net.delta.weight.normal_(std=0.3)
        if net.context is not None:
            net.context.normal_(std=0.02)
    net.constrain()
    return net


def evaluate(path, boards, since, clock):
    return integer_eval(path, boards, since, clock)


@pytest.mark.parametrize("hidden,version", [(32, 8), (64, 8), (512, 6), (32, 6)])
def test_integer_file_reproduces_the_quantised_forward(tmp_path, hidden, version):
    torch.set_num_threads(1)
    net = trained_like(NNUE(hidden, version), 5)
    boards, since, clock = positions(21, 96)
    path = tmp_path / "net.nnue"
    info = export(net, path)
    assert info["bytes"] == file_size(hidden, version) == path.stat().st_size
    assert info["version"] == version and info["features"] == net.layout.features
    assert struct.unpack_from("<II", path.read_bytes(), 8) == (version, net.layout.features)
    assert struct.unpack_from("<I", path.read_bytes(), 32)[0] == 1
    assert read(path)["weights"].shape == (net.layout.features, hidden)
    ids = torch.from_numpy(net.layout.rows(feature_ids(boards, since, clock)))
    with torch.no_grad():
        expected = net(ids, qat=True).numpy()
    got = evaluate(path, boards, since, clock)
    assert np.max(np.abs(got - expected)) < 2e-6
    again = load(path)
    assert again.version == version and again.hidden == hidden
    with torch.no_grad():
        assert np.max(np.abs(again(ids, qat=True).numpy() - expected)) < 2e-6
    # The load/export round trip reproduces the bytes (a version 8 row is split into factor and residual).
    assert export(again, tmp_path / "again.nnue")["sha256"] == info["sha256"]
    assert path.with_suffix(".nnue.json").is_file()


def test_sizes_match_the_contract_and_damaged_files_are_rejected(tmp_path):
    assert file_size(512, 6) == 1_064_172
    assert file_size(512, 8) == 14_003_436
    assert HEADER.size == 36
    path = tmp_path / "net.nnue"
    export(NNUE(32, 6), path)
    good = path.read_bytes()
    for bad in (
        good[:-1],
        good + b"\0",
        good[:32] + struct.pack("<I", 4) + good[36:],
        good[:16] + struct.pack("<I", 48) + good[20:],
    ):
        path.write_bytes(bad)
        with pytest.raises(ValueError):
            read(path)


def test_conversion_copies_every_piece_row_into_all_contexts_and_keeps_the_values(tmp_path):
    net = trained_like(NNUE(32, 6), 13)
    six, eight = tmp_path / "six.nnue", tmp_path / "eight.nnue"
    export(net, six)
    info = convert(six, eight)
    assert info["version"] == 8 and info["bytes"] == file_size(32, 8) == eight.stat().st_size
    assert info["converted_from"]["version"] == 6
    a, b = read(six), read(eight)
    piece = b["weights"][: FORMAT8.attack_base].reshape(CONTEXTS, PIECE_ROWS, 32)
    assert np.array_equal(piece, np.broadcast_to(a["weights"][:PIECE_ROWS], piece.shape))
    assert np.array_equal(b["weights"][FORMAT8.attack_base :], a["weights"][PIECE_ROWS:])
    for key in ("bias", "output", "output_bias", "dense", "dense_bias", "residual"):
        assert np.array_equal(a[key], b[key]), key
    boards, since, clock = positions(6, 300)  # more than one evaluation chunk
    assert np.array_equal(evaluate(six, boards, since, clock), evaluate(eight, boards, since, clock))
    # The trainer's widening is the same conversion, byte for byte.
    wide = load(six).widen_contexts()
    assert export(wide, tmp_path / "wide.nnue")["sha256"] == info["sha256"]
    with pytest.raises(ValueError):
        wide.widen_contexts()
    with pytest.raises(ValueError):
        convert(eight, tmp_path / "again.nnue")
    # A trained context residual changes only the positions where a perspective sees that context.
    ids = torch.from_numpy(feature_ids(boards, since, clock))
    with torch.no_grad():
        before = wide(ids, qat=True)
        wide.context[26].fill_(0.02)
        after = wide(ids, qat=True)
    touched = torch.from_numpy((context(boards) == 26).any(axis=1))
    assert touched.any() and not touched.all()
    assert torch.equal(before[~touched], after[~touched]) and not torch.equal(before[touched], after[touched])


def test_widened_hidden_evaluates_identically_with_distinct_new_columns(tmp_path):
    net = trained_like(NNUE(32), 11)
    wide = net.widen_hidden(64, seed=3)
    assert wide.hidden == 64 and wide.version == net.version
    boards, since, clock = positions(4, 40)
    a = evaluate(export(net, tmp_path / "a.nnue") and tmp_path / "a.nnue", boards, since, clock)
    b = evaluate(export(wide, tmp_path / "b.nnue") and tmp_path / "b.nnue", boards, since, clock)
    assert np.array_equal(a, b)
    new = wide.piece[:, 32:]
    assert new.std() > 0.01 and not torch.equal(new[:, 0], new[:, 1])
    assert not wide.context[..., 32:].any()
    assert not wide.output.weight[:, 32:64].any() and not wide.output.weight[:, 96:].any()
    assert not wide.dense.weight[:, 32:64].any() and not wide.dense.weight[:, 96:].any()
    assert torch.equal(net.widen_hidden(64, seed=3).piece, wide.piece)
    with pytest.raises(ValueError):
        net.widen_hidden(32)


def test_widened_hidden_with_seeded_outgoing_deviates_boundedly(tmp_path):
    net = trained_like(NNUE(32), 11)
    wide = net.widen_hidden(64, seed=3, outgoing=1 / 64)
    boards, since, clock = positions(4, 40)
    a = evaluate(export(net, tmp_path / "a.nnue") and tmp_path / "a.nnue", boards, since, clock)
    b = evaluate(export(wide, tmp_path / "b.nnue") and tmp_path / "b.nnue", boards, since, clock)
    assert not np.array_equal(a, b) and np.abs(a - b).max() < 0.5
    for name in ("output", "dense"):
        layer, source = getattr(wide, name), getattr(net, name)
        new = torch.cat([layer.weight[:, 32:64], layer.weight[:, 96:]], 1)
        if name == "output":
            assert torch.all(new.abs() == 1 / 64) and (new > 0).any() and (new < 0).any()
        else:
            assert not new.any()
        assert torch.equal(layer.weight[:, :32], source.weight[:, :32])
        assert torch.equal(layer.weight[:, 64:96], source.weight[:, 32:])
    assert torch.equal(net.widen_hidden(64, seed=3).piece, wide.piece)  # the rows do not depend on the amplitude
    assert torch.equal(net.widen_hidden(64, seed=3, outgoing=1 / 64).output.weight, wide.output.weight)
    assert not torch.equal(net.widen_hidden(64, seed=4, outgoing=1 / 64).output.weight, wide.output.weight)


def trained_like9(net, seed):
    """A version 9 network whose goal rows and head residuals have left zero."""
    trained_like(net, seed)
    with torch.no_grad():
        net.goal.normal_(std=0.3)
        for name in ("output_head", "output_head_bias", "dense_head", "dense_head_bias", "delta_head"):
            getattr(net, name).normal_(std=0.05)
    net.constrain()
    return net


def test_version9_file_reproduces_the_quantised_forward_and_round_trips(tmp_path):
    torch.set_num_threads(1)
    net = trained_like9(NNUE(32, 9), 7)
    boards, since, clock = positions(23, 128)
    path = tmp_path / "nine.nnue"
    info = export(net, path)
    assert info["bytes"] == file_size(32, 9) == path.stat().st_size == 1604 + 2 * FORMAT9.features * 32 + 546 * 32
    assert struct.unpack_from("<III", path.read_bytes(), 8) == (9, FORMAT9.features, 32)
    assert struct.unpack_from("<I", path.read_bytes(), 32)[0] == HEADS
    assert read(path)["weights"].shape == (FORMAT9.features, 32)
    assert read(path)["dense"].shape == (HEADS, 32, 64) and read(path)["output_bias"].shape == (HEADS,)
    ids = torch.from_numpy(feature_ids9(boards, since, clock))
    assert ids.shape == (128, 2, 44)
    with torch.no_grad():
        expected = net(ids, qat=True).numpy()
    got = evaluate(path, boards, since, clock)
    assert np.max(np.abs(got - expected)) < 2e-6
    again = load(path)
    assert again.version == 9 and again.hidden == 32
    with torch.no_grad():
        assert np.max(np.abs(again(ids, qat=True).numpy() - expected)) < 2e-6
    # The load/export round trip reproduces the bytes (a head is split into the shared layer and its residual).
    assert export(again, tmp_path / "again.nnue")["sha256"] == info["sha256"]
    # The positions reach every head, and a network with one head everywhere
    # differs exactly where the bucket is not the one it was flattened to.
    bucket = piece_bucket(np.count_nonzero(boards, axis=1))
    assert set(bucket.tolist()) == set(range(HEADS))
    with torch.no_grad():
        for name in ("output_head", "output_head_bias", "dense_head", "dense_head_bias", "delta_head"):
            getattr(again, name).copy_(getattr(again, name)[:1].expand_as(getattr(again, name)).clone())
    flat = evaluate(export(again, tmp_path / "flat.nnue") and tmp_path / "flat.nnue", boards, since, clock)
    assert np.array_equal(flat[bucket == 0], got[bucket == 0])
    assert not np.array_equal(flat[bucket != 0], got[bucket != 0])
    # The goal rows are read: zeroing them moves the positions with an occupied corner.
    with torch.no_grad():
        again.goal.zero_()
    blind = evaluate(export(again, tmp_path / "blind.nnue") and tmp_path / "blind.nnue", boards, since, clock)
    assert not np.array_equal(blind, flat)
    for bad in (
        path.read_bytes()[:-1],
        path.read_bytes() + bytes(1),
        path.read_bytes()[:32] + struct.pack("<I", 1) + path.read_bytes()[36:],
        path.read_bytes()[:12] + struct.pack("<I", FORMAT8.features) + path.read_bytes()[16:],
    ):
        (tmp_path / "bad.nnue").write_bytes(bad)
        with pytest.raises(ValueError):
            read(tmp_path / "bad.nnue")


def test_conversion_to_version9_repeats_the_head_and_zeroes_the_goal_rows(tmp_path):
    net = trained_like(NNUE(32, 6), 13)
    six, eight, nine = tmp_path / "six.nnue", tmp_path / "eight.nnue", tmp_path / "nine.nnue"
    export(net, six)
    convert(six, eight)
    info = convert(eight, nine, to=9)
    assert info["version"] == 9 and info["buckets"] == HEADS and info["bytes"] == file_size(32, 9)
    assert info["converted_from"]["version"] == 8
    a, b = read(eight), read(nine)
    assert np.array_equal(b["weights"][: FORMAT9.goal_base], a["weights"])
    assert not b["weights"][FORMAT9.goal_base :].any()
    for index in range(HEADS):
        for key in ("output", "dense", "dense_bias", "residual"):
            assert np.array_equal(b[key][index], a[key]), key
        assert b["output_bias"][index] == a["output_bias"][0]
    boards, since, clock = positions(6, 300)  # more than one evaluation chunk
    assert np.array_equal(evaluate(eight, boards, since, clock), evaluate(nine, boards, since, clock))
    # The trainer's widening is the same conversion, byte for byte.
    wide = load(eight).widen_buckets()
    assert export(wide, tmp_path / "wide.nnue")["sha256"] == info["sha256"]
    for bad in (wide.widen_buckets, load(six).widen_buckets):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):
        convert(six, tmp_path / "no.nnue", to=9)
    with pytest.raises(ValueError):
        convert(nine, tmp_path / "no.nnue", to=9)
