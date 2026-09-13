# engine

The rules of Intransitive and shared board queries. No players, no search, no network,
no site protocol. Every other crate and every model's training code gets its
rules from here; the only permitted second implementation is a model's GPU
kernels, which must be tested against this crate.

## Interface

- `board`: `Board`, `Cell`, `Piece`, `Square`, `Dir`, `Action`, the canonical
  frame (`mirror_anti`, `swap_side`, `flip`) and the initial position.
- `rules`: `State`, `Rules`, `Outcome`, `legal_mask`, `apply`, `beats`;
  `State::legal_actions_into` generates the legal moves into a caller's buffer
  for searches that visit millions of positions. `State::legal_moves` builds
  compact capture and quiet masks: `len`, `is_empty`, `contains`, `all_into`,
  `captures_into` and `quiets_into(target_squares, out)` allow a search to
  enumerate captures before quiets. Appended actions are in ascending order;
  the quiet target set is a `u128` square bitmask.
- `tactics`: exact short tactics per legal move, `wins_at_once`,
  `loses_in_two` (the reply wins at once) and `wins_in_three` (every reply
  leaves a win at once), plus `any_win_at_once`; tested against brute force. `can_capture`,
  `is_attacked`, `goal_move` and `goal_threat` supply the NNUE feature and
  quiescence queries without duplicating rules. `attack_candidates(board, action)`
  returns a square bitmask covering every occupancy or attack-status change
  after a legal move, for incremental feature updates.
- `notation`: square and move spelling (`a1`, `a1-b2`) in the canonical frame.
- `python` (feature `python`): a `pyo3` module `engine` exposing
  `initial_board()`, `legal_mask(board)`,
  `apply(board, since_capture, ply, action, capture_clock)` and
  `tactics(board)` (three 648-flag lists: wins at once, loses in two, wins in
  three) on plain lists of cell codes (0 empty, 1-3 own rock/paper/scissors,
  4-6 enemy). Used by model kernel tests.

`cargo xtask wheel` builds the Python module (maturin, pyproject.toml here)
into target/wheels and installs it.

See docs/rules.md for the rules themselves.
