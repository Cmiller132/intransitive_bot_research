# models

One package per model. A model owns all of its logic and touches the rest of
the workspace only through `engine` (the rules), `search` (the generic Gumbel
search, which it may use or extend) and `match::Player`.

A model package `models/<name>/` holds:

- `Cargo.toml`, `src/`: a Rust crate for play time that implements
  `match::Player` (inference, search settings); a workspace member.
- `py/`: a Python package beside it for training and export (network,
  self-play, replay, training loop, ONNX export) with its own pyproject.
- `DESIGN.md`: the numbered design items, each approved before it is built.
- `README.md`: purpose and interface of both halves.

Plugging in: list the crate in the root Cargo.toml members and in `cli`'s
dependencies, and add one arm to `player_from_spec` in cli/src/main.rs so
that `<name>:<path>` builds the player. Eval, play and rpsi then work through
`match::Player` without further changes.

`sq/` is the worked example (the reference line); `conv/` is the second model,
built to the same layout; `nnue/` is the CPU line, a sparse integer evaluator
inside an alpha-beta search, distilled from the others' play and games;
`zero/` holds a design document only.
