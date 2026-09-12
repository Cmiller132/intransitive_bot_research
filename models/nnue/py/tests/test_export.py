"""The exported file against the fake-quantised forward pass, the round trip
through `load`, the widenings and the version 6 to 8 conversion."""

import struct

import numpy as np
import pytest
import torch

from nnue.export import HEADER, convert, export, file_size, integer_eval, load, read
from nnue.features import CONTEXTS, FORMAT8, PIECE_ROWS, context, feature_ids
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
