# engine

The rules of Intransitive and nothing else. No players, no search, no network,
no site protocol. Every other crate and every model's training code gets its
rules from here; the only permitted second implementation is a model's GPU
kernels, which must be tested against this crate.

## Interface

- `board`: `Board`, `Cell`, `Piece`, `Square`, `Dir`, `Action`, the canonical
  frame (`mirror_anti`, `swap_side`, `flip`) and the initial position.
- `rules`: `State`, `Rules`, `Outcome`, `legal_mask`, `apply`, `beats`.
- `notation`: square and move spelling (`a1`, `a1-b2`) in the canonical frame.
- `python` (feature `python`): a `pyo3` module `engine` exposing
  `initial_board()`, `legal_mask(board)` and
  `apply(board, since_capture, ply, action, capture_clock)` on plain lists of
  cell codes (0 empty, 1-3 own rock/paper/scissors, 4-6 enemy). Used by model
  kernel tests.

`cargo xtask wheel` builds the Python module (maturin, pyproject.toml here)
into target/wheels and installs it.

See docs/rules.md for the rules themselves.
