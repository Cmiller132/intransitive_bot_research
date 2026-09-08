# sq: square-attention model

An all-attention network over the 81 squares with a rule-aware pair bias, a
from-to policy head and a categorical action-value head, trained by self-play
whose targets come from a batched GPU Gumbel search. DESIGN.md holds the
numbered design items.

The package has two halves that share only the observation encoding and the
export format:

- `src/` (Rust crate `sq`): observation planes, the ONNX evaluator with a
  position cache, and `SqPlayer`, which implements `match::Player` on top of
  `search::Gumbel` with this line's play settings.
- `py/` (Python package `sq`): the network, the batched GPU environment and
  search (Triton kernels), replay, the training loop, checkpoint conversion
  and the ONNX export.

## Rust interface

- `SqPlayer::load(onnx_path, threads)`: the default play settings (contempt,
  repetition penalty, moves-left, batch, cache size); `with_settings` takes a
  `PlaySettings`.
- `planes::encode(state) -> [f32; 25 * 81]`.
- `OrtNet`: `search::Evaluator` over an exported graph and its sidecar.

## Python interface

`pip install -e ".[dev]"` from `py/` installs the package with pytest, ruff
and maturin; the `engine` wheel comes from `cargo xtask wheel`. On Windows,
`kernels.py` points Triton's C compiler (`CC`) at the tcc it bundles when
`CC` is unset. Commands, from the workspace root:

```
python -m sq.train   --run <name> [--init weights/sq_g128.pt | --resume runs/<name>/latest.pt] [--iters N] [--<section>.<field> value ...]
python -m sq.export  --ckpt runs/<name>/latest.pt --out runs/<name>/export.onnx [--weights ema|model]
python -m sq.convert --old <previous-format.pt> --out weights/sq_g128.pt [--clock-from-iter N]
```

A run is written to `<workspace root>/runs/<name>/` whatever the working
directory (`paths.py`); `--init` and `--resume` paths are resolved from the
working directory.

Every config field is a flag, for example `--search.sims 64` or
`--learn.batch 512`; `runs/<name>/config.json` records the values used.

A run directory holds `config.json`, `latest.pt`, `ckpt_NNNNNN.pt` every
`checkpoint_every` iterations, `log.csv` with one row per iteration (losses,
conversion metrics, timings) and the replay windows.

A checkpoint holds the trained weights, their EMA, the optimizer moments and
the config. `--init` starts a fresh run from a checkpoint's weights;
`--resume` continues one with its optimizer state and iteration count.

Export writes `<out>` (input `planes`; outputs `logits`, `q`, `plies_to_end`,
`draw`) and `<out>.json` (name, checkpoint hash, plane and atom counts, the
prior temperatures alpha and beta, plies-to-end class centres), then checks
ONNX Runtime against torch. The Rust side reads both files.

The tests check the kernels against the engine on random positions, the
search semantics with a tiny network stand-in, one collection and learner
step, and export parity. `cargo xtask check` runs them on the CPU through the
Triton interpreter (`TRITON_INTERPRET=1`), `cargo xtask check --gpu` on the
GPU; `pytest` from `py/` does the same and picks the interpreter itself when
CUDA is unavailable (`tests/conftest.py`).
