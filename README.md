# Intransitive bot

Training and serving a bot for the board game Intransitive. Project.md states
the goals and component boundaries, Agents.md the working rules, docs/rules.md
the game.

## Layout

```
engine/        game rules (Rust; optional Python module for kernel tests)
search/        generic Gumbel MCTS over an Evaluator trait (Rust)
match/         Player trait, game runner, paired eval, RPSI site adapter (Rust)
cli/           the `bot` binary: eval, play and rpsi over every model (Rust)
arena/         the rating runner and the site on the Proxmox container (Python, React)
models/        one package per model: Rust crate, Python package, DESIGN.md
xtask/         workspace tasks: `cargo xtask check | wheel | linux`
weights/       reference checkpoints and their exports (untracked)
runs/          training runs, `runs/<name>/` (untracked)
dist/          release artifacts, `dist/linux/` (untracked)
docs/          the game rules and the architecture specification
```

## Commands

```
cargo xtask check       # the one check: fmt, clippy, tests, engine wheel, ruff, pytest (CPU), the arena's npm gates
cargo xtask linux       # release build of `bot` for Debian 12 x86_64 -> dist/linux/ (bot + libonnxruntime.so.1)
cargo build --release   # Windows binary -> target/release/bot
```

The Rust toolchain is pinned in rust-toolchain.toml. Python 3.11+ with torch
and triton; `pip install -e "models/sq/py[dev]"` and the same for
models/conv/py and arena/backend install the packages with pytest, ruff and
maturin; `npm ci` in arena/frontend installs the site. xtask/README.md
describes each task and the one-time setup of the Linux build environment.

Adding a model: models/README.md.
