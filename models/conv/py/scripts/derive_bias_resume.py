"""Derive a bias-excess RGSC resume checkpoint with fresh regret heads."""

from __future__ import annotations

import os
import shutil
import sys

import torch

from conv.control import Buffer
from conv.model import ConvNet, config_of, load_checkpoint
from conv.train import regret_parameter_indices


def main(argv: list[str]) -> None:
    if len(argv) not in (3, 4):
        raise SystemExit("usage: derive_bias_resume.py SRC DST [BACKUP]")
    src, dst = argv[1:3]
    if len(argv) == 4:
        shutil.copy2(src, argv[3])

    ck = load_checkpoint(src)
    iteration = ck["iteration"]
    retained = {}
    for group in ("model", "ema"):
        retained[group] = {key: value for key, value in ck[group].items() if not key.startswith("regret.")}
        for key in [key for key in ck[group] if key.startswith("regret.")]:
            del ck[group][key]

    cfg = config_of(ck)
    indices = regret_parameter_indices(ConvNet(cfg.net))
    optimizer = ck["optimizer"]["state"]
    before = len(optimizer)
    assert all(index in optimizer for index in indices)
    retained_optimizer = {index: entry for index, entry in optimizer.items() if index not in indices}
    for index in indices:
        del optimizer[index]
    assert len(optimizer) == before - len(indices)

    ck["control"] = {
        "buffer": Buffer(cfg.control.capacity, cfg.control.ema).state(),
        "warmup_left": 0,
        "heads_active": False,
        "rank_lift": [],
    }
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = dst + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, dst)

    derived = load_checkpoint(dst)
    assert derived["iteration"] == iteration
    assert all(not key.startswith("regret.") for group in ("model", "ema") for key in derived[group])
    assert len(derived["optimizer"]["state"]) == before - len(indices)
    assert derived["optimizer"]["state"].keys() == retained_optimizer.keys()
    for index, entry in retained_optimizer.items():
        assert entry.keys() == derived["optimizer"]["state"][index].keys()
        assert all(torch.equal(value, derived["optimizer"]["state"][index][key]) for key, value in entry.items())
    for group in ("model", "ema"):
        assert derived[group].keys() == retained[group].keys()
        assert all(torch.equal(derived[group][key], value) for key, value in retained[group].items())


if __name__ == "__main__":
    main(sys.argv)
