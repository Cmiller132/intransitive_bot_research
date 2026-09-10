"""The labeller preserves root evaluation and all consumed leaf fields."""

import inspect
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nnue.gpu_label import _ConvActor, searcher


@pytest.mark.parametrize("with_regret", [False, True])
def test_leaf_omits_draw_reduction_and_preserves_consumed_fields(monkeypatch, with_regret):
    board = torch.ones(2, 81, dtype=torch.int8)
    since, ply, clock = (torch.zeros(2, dtype=torch.int32) for _ in range(3))
    legal, count = torch.ones(2, 4, dtype=torch.bool), torch.full((2,), 4)
    out = SimpleNamespace(
        logits=torch.randn(2, 4),
        q=torch.randn(2, 4),
        v=torch.randn(2),
        regret=torch.randn(2, 2).bfloat16() if with_regret else None,
    )
    draw = torch.full_like(out.q, 0.25)
    expected = out.logits, out.q, out.v, draw, legal, count
    if with_regret:
        expected = *expected, out.regret[:, 0].float(), out.regret[:, 1].float()

    def forward(planes):
        assert planes.dtype == torch.float32
        return out

    kernels = SimpleNamespace(derive_batch=lambda *args: (board.bfloat16(), legal, count))
    actor = _ConvActor(lambda *args: expected, forward, kernels)
    assert actor(board, since, ply, clock) is expected

    def forbidden(*args, **kwargs):
        raise AssertionError("the leaf computed a discarded softmax")

    monkeypatch.setattr(torch, "softmax", forbidden)
    actual = actor.leaf(board, since, ply, clock)
    assert actual[3] is out.q
    for i in range(len(expected)):
        if i != 3:
            assert torch.equal(actual[i], expected[i])


@pytest.mark.parametrize("fail", [False, True])
def test_simulation_restores_root_actor(monkeypatch, fail):
    seen = []

    class FakeSearch:
        def __init__(self, *args):
            self.evaluate = None

        def forget(self):
            pass

        def __call__(self, board, since, ply, legal, actor, full):
            self.evaluate = actor
            self._simulate()
            return SimpleNamespace(value=torch.zeros(len(board)))

        def _simulate(self):
            seen.append(self.evaluate)
            if fail:
                raise RuntimeError("expansion failed")

    package, module = ModuleType("conv"), ModuleType("conv.search")
    package.__path__ = []
    module.GumbelSearch = FakeSearch
    monkeypatch.setitem(sys.modules, "conv", package)
    monkeypatch.setitem(sys.modules, "conv.search", module)

    @dataclass
    class Config:
        sims: int = 128
        cheap_sims: int = 16

    cfg = SimpleNamespace(search=Config(), rules=SimpleNamespace(clock_penalty=0.05, max_plies=1000))
    leaf = object()
    actor = SimpleNamespace(leaf=leaf)
    kernels = SimpleNamespace(derive_batch=lambda *args: (None, torch.ones(2, 4), None))
    values, _ = searcher(("conv", actor, cfg, kernels), torch.device("cpu"), 128, 2, 0)
    search = inspect.getclosurevars(values).nonlocals["search"]
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError, match="expansion failed"):
                values(torch.zeros(2, 81), None, None)
        else:
            assert torch.equal(values(torch.zeros(2, 81), None, None), torch.zeros(2))
        assert search.evaluate is actor
    assert seen == [leaf, leaf]
