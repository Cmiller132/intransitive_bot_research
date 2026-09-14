# Intransitive bot workspace

A fresh workspace for training and serving a bot for the game Intransitive.
Code is ported from the old project, then stripped and rewritten so that every
package, class, method and line has a purpose.

## Problems this workspace solves

- Bloat: the old codebase was hard to manage. Ported code is reduced to what the
  current line actually uses; anything bloated or improvable is rewritten.
- Documentation and history overload: agents got stuck on old problems, blamed
  new problems on old docs, and dropped ideas because "it was tried before" on
  thin evidence. This workspace keeps minimal docs and no history, and trims
  them actively.
- Repeated solutions: the old workspace solved the same problem several times
  (evaluation most notably). Here every problem has exactly one solution.
- Unclear structure: the work is split into four components with fixed
  boundaries (below).

## Components

### Engine (Rust)

Game rules only: board, state, legal moves, move application, terminal
outcomes, the capture clock and any other rule of the site's game. It knows
nothing about players, networks, search or the site protocol. It is the single
rules implementation; every other component, including model training, reuses
it rather than reimplementing rules.

### Match runner (Rust)

Runs a game between two players and reports the result. It is the interface
to the engine for anything that plays games: evaluation, arbitrary matches
between any two players, and the site adapter. A player is a Rust trait: new
game, position update, choose a move under a clock. The runner never sees
network internals. The site client is a small adapter built on the runner.

Evaluation is one tool with one definition: a paired match, both colours per
opening, equal simulations (32) per move, no timed clock, against a named
reference model or an older checkpoint. The one variant is the timed eval
(`bot eval --move-ms N`), the same paired match under a per-move wall clock
with a declared thread count per player, for engines whose strength is a
function of time (the NNUE line); its report carries the complete-pair count
and a seeded opening-pair bootstrap interval. Either form can run as the
fixed sequential test (`bot eval --sprt`) instead of a fixed pair count.
Anything else is a diagnostic, not an eval.

### Models (Rust + Python)

Each model is its own package of Rust and Python code and owns all of its
logic: network definition, inference, search, training and self-play.
Models interact with the rest of the workspace only through the engine's rules
and the match runner's player interface.

A generic Gumbel search is defined once in Rust and is what the model serving
the site uses. Models may extend it for their own purposes.

### Arena (Python + React)

Rates every model uploaded to it and serves the site. It is the only place
strengths are compared: each job is the eval tool between two players of the
pool, ratings are a Bayesian Elo fit after the KataGo training server, and the
same engines answer the site's analysis requests. It runs on its own
container; nothing is evaluated on the training machine. Every tenth
checkpoint of a run is exported and uploaded by `arena/watch.sh`. Models reach
it only as `bot` specs (an ONNX export or an RPSI engine).

Planned models, in order:

1. A stripped port of the sq_g128 line (Gumbel-128 search targets, all-attention
   trunk). Ported code is cut down to what that line uses; KLENT-only paths are
   not ported.
2. A convolution network with interleaved transformer blocks.
3. A new model designed from the ground up.

Every aspect of each model is approved by the user before it is built: the
model gets a design document with numbered items, the user approves or edits
each item, and only approved items are implemented. Design documents start
blank.

## Reference artifacts

One checkpoint is brought in as read-only weights: the sq_g128 checkpoint
(weights/README.md). It serves as a distillation starting point and as the
match reference. Nothing else comes from the old run directories.

## Old workspaces

The old workspaces are read-only and ignored. Agents do not visit or read them
unless the user explicitly instructs it for a specific task. Nothing here
references them, except one document recording where the reference checkpoints
came from, if that is ever necessary. Ideas are judged on current evidence;
there is no history here to say something was tried before, and that is
deliberate.

## Documentation

Exactly these documents exist: this file, AGENTS.md, a workspace README, one
game-rules document, the bot architecture specification (docs/), one README
per crate, package and top-level directory stating its purpose and interface,
and one DESIGN.md per model holding its numbered design items.
The user-requested [RGSC diagnosis and experiment plan](models/conv/RGSC.md)
is one maintained working document. Its proposals do not approve changes to
the model design; measured data remains in the linked run artifacts.
The user-approved NNUE experiment plan, kept with its run artifacts in
runs/nnue_plan/ (untracked), defines the active controlled training
comparison; exact settings and measured results remain in those artifacts.
There is no results log, no proposals folder and no history.
Measured results live with their run artifacts and in commit messages.

## Toolchain

- Git repository at this folder; the Rust toolchain is pinned in
  rust-toolchain.toml.
- Cargo workspace: crates `engine`, `search` (the generic Gumbel search),
  `match`, `cli` (the `bot` binary that knows every model), `xtask` (the task
  runner: `cargo xtask check | wheel | release | linux`) and one crate per model under
  `models/`; each model's Python package sits beside its crate.
- Windows native, one RTX 4070 Ti, Triton kernels kept for GPU self-play.
- Deployment targets: the site bot container and the arena container, both
  Debian 12 with CPU inference through ONNX Runtime. `cargo xtask linux`
  builds the Linux binary inside a Debian 12 WSL distro (x86_64, glibc 2.36)
  into dist/linux/.

## TODO (later, do not plan further)

An automated deploy loop that finds the strongest checkpoint and deploys it to
the site without user intervention. Until it exists, deployment is a user
decision.
