"""The exported file against the fake-quantised forward pass, the round trip
through `load`, the bucketed head and the prototype conversion."""

import struct

import numpy as np
import pytest
import torch

from nnue.export import HEADER, convert_v3, export, file_size, integer_eval, load, read
from nnue.features import (
    BUCKETS,
    CLOCK_BUCKETS,
    FEATURES,
    FORMAT6_FEATURES,
    PIECE_ROWS,
    RACE_BASE,
    feature_ids,
    piece_bucket,
)
from nnue.model import DENSE, NNUE, QA, QB

from .test_features import random_boards


def positions(seed, count):
    boards = random_boards(seed, count, 2, 21)
    rng = np.random.default_rng(seed + 1)
    clock = rng.integers(50, 201, count)
    since = (rng.random(count) * clock).astype(np.int64)
    return boards, since, clock


@pytest.mark.parametrize(
    "hidden,buckets,features",
    [(32, 1, FEATURES), (64, BUCKETS, FEATURES), (512, 1, FEATURES), (32, BUCKETS, FORMAT6_FEATURES)],
)
def test_integer_file_reproduces_the_quantised_forward(tmp_path, hidden, buckets, features):
    torch.manual_seed(5)
    torch.set_num_threads(1)
    net = NNUE(hidden, buckets, features)
    with torch.no_grad():
        net.delta.weight.normal_(std=0.3)
    net.constrain()
    boards, since, clock = positions(21, 96)
    path = tmp_path / "net.nnue"
    info = export(net, path)
    assert info["bytes"] == file_size(hidden, buckets, features) == path.stat().st_size
    assert info["version"] == (7 if features == FEATURES else 6) and info["features"] == features
    assert struct.unpack_from("<I", path.read_bytes(), 32)[0] == buckets
    assert read(path)["weights"].shape == (features, hidden)
    ids = torch.from_numpy(feature_ids(boards, since, clock))
    bucket = torch.from_numpy(piece_bucket(boards))
    with torch.no_grad():
        expected = net(ids, bucket, qat=True).numpy()
    got = integer_eval(path, boards, since, clock)
    assert np.max(np.abs(got - expected)) < 2e-6
    again = load(path)
    with torch.no_grad():
        assert np.max(np.abs(again(ids, bucket, qat=True).numpy() - expected)) < 2e-6
    assert (path.with_suffix(".nnue.json")).is_file()


def test_size_at_the_migration_shape_matches_the_contract():
    assert file_size(512, 1, FORMAT6_FEATURES) == 1_064_172
    assert file_size(512, 1) == 1_064_172 + 36 * 512  # format 7 adds the 18 race rows
    assert HEADER.size == 36


def test_widened_features_keep_every_byte_and_evaluate_identically(tmp_path):
    torch.manual_seed(13)
    net = NNUE(32, BUCKETS, FORMAT6_FEATURES)
    with torch.no_grad():
        net.delta.weight.normal_(std=0.3)
    net.constrain()
    six, seven = tmp_path / "six.nnue", tmp_path / "seven.nnue"
    export(net, six)
    wide = load(six).widen_features()
    assert wide.features == FEATURES
    export(wide, seven)
    a, b = six.read_bytes(), seven.read_bytes()
    rows = 36 + 2 * 32 + 2 * FORMAT6_FEATURES * 32
    assert struct.unpack_from("<II", b, 8) == (7, FEATURES) and struct.unpack_from("<II", a, 8) == (6, 1004)
    assert a[16:rows] == b[16:rows] and a[rows:] == b[rows + 36 * 32 :]
    assert b[rows : rows + 36 * 32] == bytes(36 * 32)
    boards, since, clock = positions(6, 40)
    assert np.array_equal(integer_eval(six, boards, since, clock), integer_eval(seven, boards, since, clock))
    with pytest.raises(ValueError):
        wide.widen_features()
    # A format 6 model ignores the race ids in its forward pass; a trained race row changes the value.
    ids = torch.from_numpy(feature_ids(boards, since, clock))
    bucket = torch.from_numpy(piece_bucket(boards))
    with torch.no_grad():
        before = wide(ids, bucket, qat=True)
        wide.embedding.weight[RACE_BASE + 8].fill_(0.5)
        assert not torch.equal(wide(ids, bucket, qat=True), before)
        narrow = load(six)
        narrow.embedding.weight[RACE_BASE + 8].fill_(0.5)
        assert torch.equal(narrow(ids, bucket, qat=True), before)
        narrow.constrain()
        assert not narrow.embedding.weight[FORMAT6_FEATURES:].any()


def test_widened_buckets_evaluate_like_the_single_head(tmp_path):
    torch.manual_seed(9)
    net = NNUE(32)
    with torch.no_grad():
        net.delta.weight.normal_(std=0.3)
    wide = net.widen_buckets()
    boards, since, clock = positions(4, 40)
    a = integer_eval(export(net, tmp_path / "a.nnue") and tmp_path / "a.nnue", boards, since, clock)
    b = integer_eval(export(wide, tmp_path / "b.nnue") and tmp_path / "b.nnue", boards, since, clock)
    assert np.array_equal(a, b)


def test_widened_hidden_evaluates_like_the_narrow_network(tmp_path):
    torch.manual_seed(11)
    net = NNUE(32)
    with torch.no_grad():
        net.delta.weight.normal_(std=0.3)
    wide = net.widen_hidden(64)
    assert wide.hidden == 64 and wide.buckets == net.buckets
    boards, since, clock = positions(4, 40)
    a = integer_eval(export(net, tmp_path / "a.nnue") and tmp_path / "a.nnue", boards, since, clock)
    b = integer_eval(export(wide, tmp_path / "b.nnue") and tmp_path / "b.nnue", boards, since, clock)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        net.widen_hidden(32)


def prototype_v3_bytes(hidden, rng):
    """A synthetic prototype file: version 3, 972 features, dense 32."""
    raw = b"RPSNNUE1" + struct.pack("<IIIIIf", 3, 2 * PIECE_ROWS, hidden, QA, QB, 600.0)
    raw += rng.integers(-300, 300, hidden, dtype=np.int16).astype("<i2").tobytes()
    raw += rng.integers(-300, 300, 2 * PIECE_ROWS * hidden, dtype=np.int16).astype("<i2").tobytes()
    raw += rng.integers(-64, 64, 2 * hidden, dtype=np.int16).astype("<i2").tobytes()
    raw += struct.pack("<i", 7)
    raw += struct.pack("<I", DENSE)
    raw += rng.integers(-127, 128, DENSE * 2 * hidden, dtype=np.int8).astype("i1").tobytes()
    raw += rng.integers(-5000, 5000, DENSE, dtype=np.int32).astype("<i4").tobytes()
    raw += rng.integers(-64, 64, DENSE, dtype=np.int16).astype("<i2").tobytes()
    return raw


def test_prototype_conversion_keeps_the_weights_and_zeroes_the_clock_rows(tmp_path):
    rng = np.random.default_rng(2)
    source = tmp_path / "v3.nnue"
    source.write_bytes(prototype_v3_bytes(32, rng))
    target = tmp_path / "v6.nnue"
    info = convert_v3(source, target)
    net = read(target)
    assert info["bytes"] == file_size(32, 1, FORMAT6_FEATURES) == target.stat().st_size
    assert np.all(net["weights"][2 * PIECE_ROWS :] == 0) and net["weights"].shape == (FORMAT6_FEATURES, 32)
    assert net["output_bias"][0] == 7
    assert 2 * CLOCK_BUCKETS == FORMAT6_FEATURES - 2 * PIECE_ROWS
    boards, since, clock = positions(8, 20)
    values = integer_eval(target, boards, since, clock)
    assert np.all(np.isfinite(values))
