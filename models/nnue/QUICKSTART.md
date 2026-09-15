# NNUE self-play quick start

One machine, one command, one iteration: a starting network plays itself,
its searches label the positions, a continuation of the network is trained
on them, and the result is scored against the start. The reference
platform is macOS on Apple silicon; any Unix with the same tools works the
same way. Everything below runs from the repository root.

## What runs

```
models/nnue/quickstart.sh <start.nnue> [name]
```

| step | command | reads | writes |
|---|---|---|---|
| 1 generate | `bot selfplay` | the starting `.nnue` | `runs/nnue_selfplay/<name>/`: gzip JSONL shards, one game per line, plus `manifest.json` |
| 2 import | `python3 -m nnue.importer selfplay` | the shards | `runs/nnue_data/selfplay_<name>/`: NumPy arrays plus `provenance.json` |
| 3 encode | `python3 -m nnue.data encode` | the dataset | `ids8.npy` in it, the feature-id cache the trainer gathers from |
| 4 train | `python3 -m nnue.train` | the starting `.nnue` and the dataset | `runs/<name>/`: `config.json`, `log.csv`, `latest.pt`, `best.pt`, `best.nnue` + `.json` |
| 5 evaluate | `bot eval` | both networks | `runs/<name>/eval.json` and one summary line |

The Rust generator and the Python trainer are connected by files only.
`bot selfplay` writes records; the importer reads them; the trainer writes a
`.nnue` file; the same file is what `bot selfplay --player nnue:<file>`
generates from on the next iteration. No process calls the other.

Each searched root becomes one training row: the position, the search's
score from the mover's view as the target, the game's outcome where it is
real. Rows start at ply 16, the first eight plies are random openings and
two random moves are injected between plies 8 and 40 for diversity, and
duplicate positions are dropped. About 220 rows come out of a game.

## Setup, once

1. The C toolchain: `xcode-select --install`.
2. Rust: install with [rustup](https://rustup.rs); the repository pins its
   toolchain in `rust-toolchain.toml`, which rustup picks up on the first
   `cargo` command.
3. Python 3.11 or newer as `python3` (Homebrew's `python@3.12` or
   python.org). The Apple silicon PyTorch wheels come from PyPI.
4. From the repository root:

   ```
   cargo build --release -p cli
   python3 -m pip install -e "models/nnue/py[dev]"
   python3 -m pip install maturin && cargo xtask wheel
   ```

   The first build downloads the ONNX Runtime library the other models link
   (the `ort` crate's download feature), so it needs the network. `cargo
   xtask wheel` builds and installs the rules engine's Python module, which
   the importer replays games through; it must be built for the same
   `python3` the script uses.
5. Check: `target/release/bot --help` and
   `python3 -c "import engine, nnue, torch"` both succeed.
6. A starting network: any `.nnue` file in format 6 or 8. The repository
   ships an example, `models/nnue/examples/example.nnue`, to start from; a
   format 6 file is converted to format 8 by the trainer on the way in.

## First run

A smoke test of the whole pipeline, under a minute:

```
GAMES=8 NODES=5000 EPOCHS=1 STEPS=5 BATCH=256 WARMUP=2 PAIRS=2 SIMS=1 \
  models/nnue/quickstart.sh models/nnue/examples/example.nnue smoke
```

A first real pass sized for a CPU trainer (generation is the long part,
about 1.5 hours on a 16-core machine; training some minutes):

```
GAMES=3200 EPOCHS=4 STEPS=250 BATCH=2048 models/nnue/quickstart.sh models/nnue/examples/example.nnue first
```

The script prints each step with a timestamp and ends with a line like

```
candidate score 0.532 over 200 games (wins 96, draws 21, losses 83); margin 0.065 with bootstrap interval [-0.03, 0.16]; forfeits 0
```

A score above .5 means the new network beat the starting one at the
evaluation budget; the interval is on the margin (candidate minus
reference, in game points per opening pair), and a result whose interval
excludes zero is a real difference at that sample size. At `PAIRS=100`
(200 games) the score's noise is about plus or minus .07; `PAIRS=400`
halves it.

Rerunning the same command resumes: generation continues from its records
(the same settings are required; a different `NODES` or `GAMES` means a new
name), import and encode are skipped when their outputs exist, training
starts over into the same run directory, evaluation reruns. To retrain from
a clean slate under the same name, delete `runs/<name>/`.

## Configuring

Every knob is an environment variable set on the command line. The
defaults are the research loop's H512 continuation recipe; the script
reads the network's width from the file, so a different width needs no
setting, and a format 6 file trains as format 8.

| variable | default | meaning | when to change |
|---|---|---|---|
| `GAMES` | 3200 | games to generate | more games, more rows: the research loop trains on 12,800-game batches (about 2.9 M rows); 3,200 is a two-hour sample on 16 cores |
| `NODES` | 100000 | search nodes per move, the label's budget | the research loop's value; 250k gives slightly better labels at 2.5 times the cost, below 50k the labels get noisy |
| `THREADS` | all cores | games in flight, and the trainer's threads | one game per core is the model; efficiency cores are slower but add throughput; `sysctl -n hw.perflevel0.logicalcpu` counts performance cores if you want only those |
| `HASH` | 64 | transposition table per game in flight, MiB | 16 games use 1 GiB; lower it on a small machine, it costs little strength |
| `EPOCHS` | 20 | training epochs | the recipe; a CPU pass at 4 first |
| `STEPS` | 1000 | optimiser steps per epoch | with `BATCH`, the compute: `EPOCHS x STEPS x BATCH` rows sampled; aim for three to five passes over the data |
| `BATCH` | 8192 | rows per step | 2048 on a CPU; time per step scales with it |
| `LR` | 0.0001 | peak learning rate, cosine to 15 % after `WARMUP` steps | the continuation rate; from scratch 3e-4 |
| `WARMUP` | 200 | linear warmup steps | shorten for tiny runs |
| `DEVICE` | cpu | the trainer's device | cpu is the tested path; `mps` may work with a recent PyTorch, try it on the smoke test first; `cuda` on a machine with an NVIDIA GPU |
| `EXTRA_DATA` | | more `<set>:<share>` entries in the training mixture, space separated | on later iterations keep the earlier sets, for example `EXTRA_DATA="selfplay_first:1.0"`; shares are relative weights of the sampling mixture |
| `PAIRS` | 100 | evaluation openings, two games each (seats swapped) | 400 for a decision, 100 for a reading |
| `SIMS` | 8 | evaluation budget in units of 2,500 nodes | 8 is 20k nodes (fast); 16 is 40k, the research loop's gate |
| `SEED` | 1 | the generation, training and evaluation seed | change it for another sample of games |
| `PY`, `BOT` | python3, target/release/bot | the interpreter and the binary | a virtual environment's interpreter, a differently built binary |

Rates to plan with: about 130 games per core-hour at 100k nodes (a 16-core
machine makes about 2,200 games/h, so 3,200 games take about 1.5 hours and
12,800 about 6); 250k nodes is 2.5 times slower. Training the full recipe
is 11 minutes on a desktop GPU and hours on a CPU, which is why the first
pass above is scaled down.

## Iterating

```
models/nnue/quickstart.sh runs/first/best.nnue second
EXTRA_DATA="selfplay_first:1.0" models/nnue/quickstart.sh runs/second/best.nnue third
```

Adopt a network as the next start only when its score is above .5 with the
margin interval clear of zero; otherwise generate more games from the same
start (a new name, another `SEED`) and train on both sets with
`EXTRA_DATA`. The research loop keeps the older self-play sets and a share
of human games in every mixture and weights fresh rows three times an old
row; the script's mixture is whatever you pass.

Two more `bot selfplay` flags worth knowing: `--verify` replays a records
directory through the engine and checks every hash and label
(`target/release/bot selfplay --records runs/nnue_selfplay/<name> --verify`),
and `--pv-labels K` also records the first K moves of each search's
principal variation (the importer's `--pv-rows K` turns them into extra
rows; both are off by default and under test in the research loop).

## Troubleshooting

- `No module named engine`: run `cargo xtask wheel` with the same `python3`
  the script uses (`python3 -m pip install maturin` first).
- `No module named nnue` or `torch`: `python3 -m pip install -e
  "models/nnue/py[dev]"`; check `python3 --version` is 3.11 or newer.
- `resume rejected`: the records directory was generated with other
  settings or another network; use a new name.
- `NNUE option must be ?hash=<MiB>`: the network path contains a `?`;
  rename the file.
- Generation looks slow: every move searches `NODES` nodes and a game has
  about 230 moves, so at 100k nodes a core finishes a game in roughly half
  a minute; at the CLI's own default of 250k it takes over a minute. Check
  `NODES`.
- Training looks slow: the default recipe is sized for a GPU; use the CPU
  settings above, or `DEVICE=mps`.
- The score is below .5: a single small batch from a strong network often
  does not improve it; more games, the earlier sets in the mixture, or a
  bigger evaluation before concluding anything.
