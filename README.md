# Intransitive bot

Training and serving bots for [Intransitive](docs/rules.md), a two-player
board game on a 9x9 board: ten rock, paper and scissors pieces a side, king
steps, captures by the rock-paper-scissors rule, and a win by reaching the
far corner or eliminating the opponent. The game is played online at
[rps.henhen1227.com](https://rps.henhen1227.com); the bots here play there
through its RPSI engine protocol.

The workspace holds the rules engine, a generic Gumbel MCTS, the match
runner and evaluation tool, three model lines with their training code, the
`bot` binary that plays and analyses with any of them, and the arena that
rates every model against the rest and serves a public analysis site.
Project.md states the goals and component boundaries, AGENTS.md the working
rules for anyone (or any agent) changing the code.

## Layout

```
engine/        game rules and exact short tactics (Rust; optional Python module for kernel tests)
search/        generic Gumbel MCTS over an Evaluator trait (Rust)
match/         Player trait, game runner, paired eval and SPRT, RPSI adapter and client (Rust)
cli/           the `bot` binary: eval, play, rpsi and analyse over every model, NNUE self-play (Rust)
models/        one package per model: Rust crate for play, Python package for training, DESIGN.md
arena/         the rating runner and the site (Python FastAPI, React)
xtask/         workspace tasks: `cargo xtask check | wheel | release | linux`
docs/          the game rules and the architecture specification
weights/       reference checkpoints and their exports (untracked)
runs/          training runs, `runs/<name>/` (untracked)
dist/          release artifacts (untracked)
```

## Models

| Package | What it is | Trains on |
|---|---|---|
| `models/sq` | all-attention network over the 81 squares; the reference line | GPU self-play with a batched Gumbel search (Triton kernels) |
| `models/conv` | convolution trunk with interleaved attention, 46 rule-derived planes, exact tactics in search | GPU self-play, started by distillation from an earlier checkpoint |
| `models/nnue` | sparse integer evaluator inside an alpha-beta search; the CPU line that plays on the site | positions from the other lines' games, its own CPU self-play and site games |
| `models/zero` | a model designed from the ground up | design document only |

Every model is played through the same `match::Player` trait, so evaluation,
matches, the site adapter and the arena treat them alike. Strength is
measured by one tool, `bot eval`: a paired match, both colours per opening,
against a named reference. models/README.md explains the layout and how to
add a model; each model's DESIGN.md is its numbered specification and its
README the interface of both halves.

## Building

The Rust toolchain is pinned in rust-toolchain.toml. Python 3.12+ with
torch and triton is needed for training and for the Python tests; Node for
the arena frontend.

```
cargo build --release                      # the `bot` binary -> target/release/bot (bot.exe on Windows)
cargo xtask check                          # the one check: fmt, clippy, tests, engine wheel, ruff, pytest (CPU), the frontend gates
cargo xtask release [--portable]           # optimised host build (Zen 4 by default; --portable for other CPUs)
cargo xtask linux [--portable]             # Debian 12 x86_64 build of `bot` -> dist/linux/
```

`pip install -e "models/sq/py[dev]"`, and the same for models/conv/py,
models/nnue/py and arena/backend, install the Python packages with their
test tools (pytest, ruff; maturin for the GPU lines); `cargo xtask wheel` builds and installs the engine's
Python module; `npm ci` in arena/frontend installs the site. xtask/README.md
describes each task and the one-time setup of the Linux build environment.

## Playing and evaluating

```
bot eval  --candidate conv:runs/x/export.onnx --reference sq:weights/sq_g128.onnx --pairs 32
bot play  --first sq:a.onnx --second nnue:b.nnue --sims 32
bot rpsi  --player nnue:model.nnue --move-ms 250      # a seat on the site
bot analyse --engine conv:model.onnx                  # JSON-lines analysis, what the arena site uses
```

A player spec is `<model>:<path>` or `rpsi:<command>` for any external engine
speaking the site protocol. cli/README.md lists every command and flag;
match/README.md defines the evaluation and the record formats.

## Documents

- [Project.md](Project.md): goals, components and their boundaries, the
  documentation policy.
- [AGENTS.md](AGENTS.md): the working rules.
- [docs/rules.md](docs/rules.md): the game.
- [docs/rl-bot-architecture-specification.pdf](docs/rl-bot-architecture-specification.pdf):
  the specification of the model built from the ground up.
- One README per crate and package; one DESIGN.md per model.

## License

MIT, see [LICENSE](LICENSE).
