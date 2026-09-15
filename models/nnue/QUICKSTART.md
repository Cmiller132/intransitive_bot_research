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

A first real pass (generation is the long part, about 1.5 hours on a
16-core machine; training sizes itself to the data and takes minutes):

```
models/nnue/quickstart.sh models/nnue/examples/example.nnue first
```

On a machine with little memory add `BATCH=2048`. Training makes about
four passes over the rows (`PASSES`), so one 3,200-game batch is a few
hundred optimiser steps; the research loop's fixed 20 x 1,000 steps would
be 200 passes over the same batch and overfits it, which is what the
script did before 2026-09-15. If you ran it before then, rerun with the
new defaults.

The script prints each step with a timestamp and ends with a line like

```
candidate score 0.532 over 200 games (wins 96, draws 21, losses 83); margin 0.065 with bootstrap interval [-0.03, 0.16]; forfeits 0
```

A score above .5 means the new network beat the starting one at the
evaluation budget. The margin is the same result on a different scale:
the mean over opening pairs of (candidate points minus reference points)
divided by two, so margin = 2 x (score - .5), and the interval is its
95 % interval over pairs. The rule is the interval's lower bound: above
zero, the new network is better at that sample size; below zero, worse;
straddling zero, not yet known. In score terms that needs about

| `PAIRS` | games | score for the interval to clear zero |
|---|---|---|
| 100 | 200 | about .57 (margin .14) |
| 400 | 800 | about .535 (margin .07) |

Between .50 and that line, run the evaluation again with `PAIRS=400`
before concluding anything; a gain of 20 Elo is a score of about .53,
which 200 games cannot see. `bootstrap_interval` in `eval.json` is the same
idea computed by resampling pairs and should agree.

Rerunning the same command resumes: an unfinished batch continues from its
records (the same binary and settings are required; a different `NODES` or
`GAMES` means a new name), a finished batch is never regenerated, import
and encode are skipped when their outputs exist, training runs again into
the same run directory, evaluation reruns. Delete `runs/<name>/` before a
retrain so the run starts clean rather than appending to the old log.

That is also how to retrain on batches you already have without new games,
for example after a change to the training defaults, or to train on two
batches together:

```
rm -rf runs/first
EXTRA_DATA="selfplay_second:1.0" PAIRS=400 models/nnue/quickstart.sh models/nnue/examples/example.nnue first
```

Generation is skipped because `first` is complete, training starts from the
example on both datasets, and the evaluation is against the example.

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
| `EPOCHS` | 20 | training epochs, one validation and checkpoint each | the recipe; fewer epochs means coarser `best.nnue` selection, not less training |
| `PASSES` | 4 | passes over the training rows | the amount of training: `STEPS` is derived from it; 3 to 5 is the range that does not memorise a batch |
| `STEPS` | from the data | optimiser steps per epoch: `rows x PASSES / (BATCH x EPOCHS)` | set it only to override the sizing; `EPOCHS x STEPS x BATCH` rows are sampled |
| `BATCH` | 8192 | rows per step | 2048 on a small machine; time per step scales with it |
| `LR` | 0.0001 | peak learning rate, cosine to 15 % after `WARMUP` steps | the continuation rate; from scratch 3e-4 |
| `WARMUP` | from the data | linear warmup steps, a tenth of the run and at most 200 | set it only to override |
| `DEVICE` | cpu | the trainer's device | cpu is the tested path; `mps` may work with a recent PyTorch, try it on the smoke test first; `cuda` on a machine with an NVIDIA GPU |
| `EXTRA_DATA` | | more `<set>:<share>` entries in the training mixture, space separated | on later iterations keep the earlier sets, for example `EXTRA_DATA="selfplay_first:1.0"`; shares are relative weights of the sampling mixture |
| `PAIRS` | 100 | evaluation openings, two games each (seats swapped) | 400 for a decision, 100 for a reading |
| `SIMS` | 8 | evaluation budget in units of 2,500 nodes | 8 is 20k nodes (fast); 16 is 40k, the research loop's gate |
| `SEED` | 1 | the generation, training and evaluation seed | change it for every new batch: the generator is deterministic, so the same seed from the same network replays the same games |
| `PY`, `BOT` | python3, target/release/bot | the interpreter and the binary | a virtual environment's interpreter, a differently built binary |

Rates to plan with: about 130 games per core-hour at 100k nodes (a 16-core
machine makes about 2,200 games/h, so 3,200 games take about 1.5 hours and
12,800 about 6); 250k nodes is 2.5 times slower. Training sized to one
3,200-game batch is a few hundred steps: a few minutes on a GPU or with
`DEVICE=mps`, longer on a CPU. The research loop's own fixed recipe
(20 x 1,000 steps at batch 8192, 11 minutes on a desktop GPU) only makes
sense over its 17 M-row mixture.

## Iterating

Adopt a network as the next start only when the interval's lower bound is
above zero (about score .57 at `PAIRS=100`, .535 at `PAIRS=400`; there is
no other factor). One 3,200-game batch from an already strong network
often lands between .48 and .53, which is neither a gain nor a regression
at that sample size; two or three batches together usually decide it. So
the sequence is: a batch; if it does not clear the line, another batch
from the **same** start with another `SEED`, trained together with the
first; only a network that cleared the line becomes a start.

```
models/nnue/quickstart.sh models/nnue/examples/example.nnue first
# first scores .52 at PAIRS=100: not adopted, so a second batch from the same start, both in the mixture
SEED=2 EXTRA_DATA="selfplay_first:1.0" PAIRS=400 models/nnue/quickstart.sh models/nnue/examples/example.nnue first_v2
# first_v2 clears .535 over 800 games: the next start, with every earlier batch kept in the mixture
SEED=3 EXTRA_DATA="selfplay_first:1.0 selfplay_first_v2:1.0" models/nnue/quickstart.sh runs/first_v2/best.nnue second
```

The dataset names carry the `selfplay_` prefix, the shares are relative
weights, and `STEPS` is sized from the total rows each time, so a growing
mixture gets proportionally more steps at the same four passes.

Keep every batch. The research loop mixes each fresh batch with 13 M
older self-play rows and human games, which anchors the network; the
quick start has only the batches you pass, and a continuation trained on
one small batch alone drifts away from what the starting network knew.
The sign in `log.csv` is an `objective` that bottoms after one or two
epochs and rises while `loss` keeps falling; the cure is fewer passes and
more batches, never more epochs.

Training's own numbers are in `runs/<name>/log.csv`, one row per epoch:
`loss` is the training loss, each `<set>.mse` column is the validation
error on that dataset's held-out rows, and `objective` is their
share-weighted mean, the number that picks `best.nnue` (the epoch where
`objective` was lowest). Its absolute level depends on the data and the
network, so compare epochs within one run, not runs on different data. A
validation error that rises while the training loss keeps falling means
the run is memorising, and more data is worth more than more epochs.

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
- Training looks slow: `BATCH=2048` on a small machine, or `DEVICE=mps`;
  check the step count the script prints, it should be a few hundred for
  one batch, not thousands.
- The score is below .5, or `objective` in `log.csv` rises from epoch one
  or two: the run trained past the data (too many passes, or `STEPS` set
  by hand), or the mixture holds only one small batch. Keep `PASSES` at
  4, pass every earlier batch in `EXTRA_DATA`, start from the last network
  that cleared the line, and evaluate with `PAIRS=400` before concluding.
