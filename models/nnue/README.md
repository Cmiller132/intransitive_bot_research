# nnue: sparse integer evaluator with alpha-beta search

A CPU engine: a sparse, incrementally updated integer network (two
perspectives sharing a table of int16 rows over 13,640 features in format 8,
piece-square rows conditioned on the opponent's material, or the incumbent's
1,004 in format 6; 512 wide by default and any multiple of 32) scores positions inside a
principal-variation search (partial-root selection, cached static
evaluation, history-aware reductions, reverse and late quiet futility)
that visits millions of positions per second. DESIGN.md holds the numbered design items. The line
continues the prototype in the read-only workspace D:/Research/NNUE (its
release network is the migration control, converted once by
`nnue.export.convert_v3`).

The two halves share only the feature definition and the file format:

- `src/` (Rust crate `nnue`): the file reader (format 6; format 8 is DESIGN
  item 38's Rust side), the accumulators and the
  scalar, AVX2 and VNNI kernels, the search, and `NnuePlayer`, which
  implements `match::Player`. `bot` builds it from `nnue:<file.nnue>`.
- `py/` (Python package `nnue`): features, the trainable network with
  quantisation-aware training, the integer export and its NumPy oracle, the
  dataset schema, importers, and the training loop.

## Rust interface

- `NnuePlayer::load(path, threads)`: the player under `Clock::Time` (a host
  deadline less a 20 ms overhead; a seat's own `--movetime` budget, given
  through `choose_self_timed`, is searched in full) or `Clock::Sims(n)` (n x
  2,500 nodes, one thread, DESIGN item 2). The spec is `nnue:<file>[?hash=<MiB>]` (table
  budget, default 64). Files carry learned clock rows, one or four output
  buckets by piece count (DESIGN items 5 and 7) and any hidden width that
  is a multiple of 32 (768 costs about 12 % more per node than 512). When the deadline
  interrupts an iteration, a completed root child inside the aspiration
  window may replace the previous choice (`partial` in the search record).
  Search records include `root_moves` and `iterations`: depth, nodes,
  elapsed milliseconds (including aspiration retries), selected action and
  score, completion, and whether the selected action changed. With several
  workers the iterations belong to the worker whose result was selected;
  top-level nodes count all workers and `aborted` means any worker stopped.
- Builds: `cargo xtask release` and `cargo xtask linux` target Zen 4
  (`-C target-cpu=znver4`; the executable needs AVX-512 and faults at start
  without it) and `--portable` builds baseline x86-64 with the run-time SIMD
  dispatch, about 8 % slower. One search thread visits about 1.4 M nodes per
  second on an idle Zen 4 core with `ft_gpu150a` (staged move generation,
  accumulators materialised only when an evaluation needs them, the table
  entry prefetched before generation, an AVX-512 readout), 1.17 M on the
  arena container under its load; a plain `cargo build --release` keeps
  the generic target.
- `bot eval --move-ms N --player-threads T [--reference-move-ms M]`: the
  timed paired evaluation, the reference optionally on its own budget (the
  doubled-time yardstick for thread measurements).
  An `rpsi:<bot> rpsi --player nnue:<file>` seat receives `go movetime`
  unchanged, so two builds can meet under the same clock; with
  `--movetime <ms>` the seat searches that long (the full time, no deadline
  reserve) whenever the host sends a simulation count instead (`go sims N`),
  which is how the arena seat plays under a clock budget while the pool's
  jobs stay at 32 simulations. A seat
  may send `info json` search statistics after its move; the host copies
  them into the game record, so an external seat's nodes and time appear in
  `bot eval` records like a native player's (match/README.md).
- `bot nnue convert --input <v3.nnue> --output <v6.nnue>`,
  `bot nnue validate --model <v6.nnue> --input <positions.jsonl>`,
  `bot nnue diagnose --model <v6.nnue> --input <positions.jsonl> --nodes N`:
  the migration and parity tools; positions are
  `{"board": [81 codes], "since_capture", "ply", "clock"}` lines. Built
  with `--features nnue/profile`, `diagnose` adds a sampled profile
  (accumulator, dense head, readout, move generation, ordering, table).
- `bot analyse --engine nnue:<file>`: the analyser interface (static
  evaluation as the head, child static values as Q, a `sims` x 2,500-node
  search); `nnue.label --engine nnue:<file>` rescores positions with it.
- `bot play ... --leaves <path>` writes every position the search evaluated
  as JSON lines (the student-collection input of DESIGN item 21).

## Python interface

`pip install -e ".[dev]"` from `py/`; the `engine` wheel comes from
`cargo xtask wheel`. Commands, from the workspace root, all on the CPU:

```
python -m nnue.importer conv --run runs/<run> --out <set> [--iterations A-B] [--children 8]
python -m nnue.importer prototype --root runs/nnue_data/prototype
python -m nnue.importer human --file runs/nnue_data/human/games_export.txt --out human_games
python -m nnue.importer selfplay --records runs/nnue_selfplay/<generation> [--records ...] --out <set> [--min-ply 16]
python -m nnue.data encode <set> [<set> ...]
python -m nnue.train --run <name> --data <set>:<share> [--data <set>:<share> ...] [--init <file.nnue>] [--<field> value ...]
python -m nnue.train --run <name> --data ... --resume runs/<name>/latest.pt
python -m nnue.collect --round <name> --student <a.nnue> [--previous <b.nnue> | --opponent <spec> --move-ms N] [--families 64] [--states 10000] [--random-moves K]
python -m nnue.label --input <set> --out <set> [--engine sq:weights/sq_g128.onnx | nnue:<file>] [--sims 256] [--workers 8] [--rows N] [--quiet-best]
python -m nnue.gpu_label --input <set> --out <set> [--input <set> --out <set> ...] --ckpt runs/conv_g128/ckpt_000150.pt [--teacher conv|sq] [--sims 128] [--batch 4096] [--rows N] [--reuse-nodes K] [--repetition-draw false]
python -m nnue.gauntlet --name <name> --candidate <a.nnue> --incumbent <b.nnue> [--stage screen|accept|confirm|all] [--sims N] [--incumbent-bin <bot.exe>]
```

The data loop is import, train, collect, label, train again: `nnue.collect`
plays the student through `bot play` (two seats per seeded opening family at
2.5k, 7.5k and 20k nodes, leaves traced on the cheapest games; or against
any player spec such as the teacher under a wall clock) and samples
roots, leaves, alternatives at value drops and targeted positions into a raw
set; `nnue.label` gives every row an exact proof from the engine or the
teacher's root value from `bot analyse`, and `nnue.gpu_label` does the same
in bulk when the GPU is free, through conv's or sq's batched search from a
training checkpoint (`--teacher`; about 250 positions per second at 128
simulations on the RTX 4070 Ti, against one or two per second per CPU
thread; the GPU is saturated by the teacher's forward pass, so several sets
given to one call share the loaded teacher and the next set's exact proofs
are computed on the CPU while the GPU searches; the teacher's search scores a
repetition along its own path as a draw when the checkpoint says so, which the
provenance records and `--repetition-draw false` turns off to match the site's
rules; it is not part of the test
suite); `nnue.gauntlet` plays a candidate
against the incumbent in three predeclared stages through `bot eval
--move-ms`, or under a fixed node budget (`--sims`) for evaluator-only
changes, seating a different build of `bot` through `rpsi:` when a search
change is under test. All three run the release `bot`, or the frozen copy
named by `NNUE_BOT`, so a long match never holds the file the next build
replaces.

A dataset is `runs/nnue_data/<set>/`: columnar NumPy arrays (board,
since_capture, ply, capture_clock, target, weight, kind, outcome,
outcome_ok, source, game, orbit, split), `provenance.json` and, after
`nnue.data encode`, the feature-id cache `ids8.npy` (format-8 ids; a
format-6 run maps them onto its table) that lets the trainer gather ids
instead of encoding boards (on the GPU, `--device cuda`, about 300k rows per
second; a batch encodes on the fly when any of its sets lacks the cache;
a cache of another shape is rejected); `nnue.data`
defines the fields and the `Mixture` sampler, `nnue.importer` writes sets
from conv windows, from the prototype's arrays and from the site's export of
human games (every played root labelled with the outcome at weight 0.25;
`nnue.games` decodes the export and replays absolute move tokens through
the engine).

A run is `runs/<name>/`: `config.json`, `log.csv` (one row per epoch:
losses, the selection objective, which scores only rows that are not bare
game outcomes, validation error overall and by stratum under all six
symmetries for every dataset), `latest.pt`, `best.pt` and
`best.nnue` with its JSON sidecar (hashes, shape, the run's settings). Every
config field is a flag (`--batch 2048`, `--loss bce`, `--buckets 4`,
`--hidden 768` with a 512-wide `--init` widens it with seeded new columns
(DESIGN item 33), `--version 6` trains the incumbent's layout where the
default 8 turns a format 6 `--init` into the factorised format 8 network,
which evaluates identically at first (item 38),
`--clock false` for the clock-row ablation, `--quiet true` to train only on
rows where the mover has no capture). Games and verdicts go to
`runs/nnue_collect/<round>/` and `runs/nnue_gauntlets/<name>/`.

`nnue.export` writes and reads formats 6 and 8 (`export`, `read`, `load`),
evaluates a file with NumPy alone (`integer_eval`) and converts a format 6
file to format 8 (`python -m nnue.export convert <in> <out>`: every
piece-square row copied into all 27 contexts, identical raw values on every
position). Format 8 keeps the 36-byte header with version 8 and 13,640
features: rows `486 c + 81 (code - 1) + square` for the perspective's
opponent-material context `c = min(r, 2) + 3 min(p, 2) + 9 min(s, 2)`, then
the 486 attacked-piece rows and the 32 clock rows; width, the single head,
scales and payload order are unchanged (H512: 14,003,436 bytes). The
trainer holds the piece rows factorised, a shared 486-row factor plus 27
zero-started residual tables, and quantises their sum; `NNUE.widen_contexts()`
is the byte-identical model-side conversion. `nnue.features` encodes 42 slots
per perspective (20 pieces, 20 attacked pieces, two clock rows) in the
format 8 layout; `Layout.rows` maps them onto a format 6 table. The exact
shared contract, with the fixtures the Rust side is checked against
(`tests/fixtures/format8_*`), is runs/nnue_plan/format8_contract.md.

## Self-play (the volume loop)

`bot selfplay --player nnue:<frozen.nnue>?hash=64 --nodes 250000 --games G
--threads 16 --seed S --opening-plies 8 --random-moves 2 --random-from 8
--random-to 40 --records <dir>` plays the frozen network against itself,
sixteen one-thread games in flight, and the search that plays each move is
the label: every searched root is written with the last completed
iteration's score and best action, the played action, depth, nodes and
time, into gzip JSONL shards under a manifest (model, binary and rule
hashes, effective settings; `--resume` continues a generation exactly,
`--verify` replays it through the engine). Games end under the site rules
only (capture clock 200, no repetition draw, no score adjudication); a ply
cap is a censored game. `nnue.importer selfplay` turns the published shards
into a dataset: `target = tanh(root_score / 600)` from the mover's view
(kind TEACHER; engine proofs as PROOF), the outcome from the mover's view
with `outcome_ok` only for real results at or after the last random
deviation, rows from ply 16, duplicates by board and clock state dropped,
splits by game and orbit. Measured on eight cores (logical CPUs 16-31 of
the 7950X, below normal): about 400 games and 100k eligible roots per hour
at 250k nodes (median completed depth 7), 210k at 100k nodes, 18k at 1M.
The controlled training study on this data (DESIGN item 37, a null) is in
runs/nnue_plan/selfplay_plan_final.md; the generator measurements are in
runs/nnue_plan/astra_status19.md; the step-change programme that followed
is runs/nnue_plan/step_plan_final.md.

## Retraining on a stronger teacher

When the producing model is clearly stronger than the one the current
network learned from, repeat the loop from the incumbent rather than from
scratch (DESIGN items 26-31):

```
python -m nnue.importer conv --run runs/conv_g128 --out conv_<from>_<to> ...
python -m nnue.gpu_label --input human_games --out human_gpu<n> --ckpt runs/conv_g128/ckpt_<n>.pt   # GPU free
python -m nnue.collect --round student_<k> --student <incumbent.nnue> --previous <parent.nnue> --workers 8
python -m nnue.label --input student_<k> --out student_<k>_self --engine nnue:<incumbent.nnue> --sims 40 --workers 8
python -m nnue.label --input <pool> --rows 20000 --select <pool_shallow> --sims 400 --out <pool>_selected_1m --engine nnue:<incumbent.nnue>   # the rows a 1-sim pass misjudges most (plan item D)
python -m nnue.data encode <every new set>
python -m nnue.train --run <name> --init <incumbent.nnue> --device cuda --batch 8192 --steps_per_epoch 1000 --epochs 20 --lr 0.0003 --data ...
python -m nnue.gauntlet --name <name>_sims8 --candidate runs/<name>/best.nnue --incumbent <incumbent.nnue> --stage all --sims 8
python -m nnue.gauntlet --name <name>_timed --candidate ... --incumbent ... --stage accept      # NNUE_BOT=<deployed search build>
sh arena/upload.sh http://<arena> nnue_1 rpsi:./run.sh --replace <seat.tar.gz>; ssh <arena> systemctl restart arena
```

Selection is by games only: a lower validation loss has not predicted a
gauntlet win here.

## Tests

From `py/`: `python -m pytest -q` (the encoder against a slow enumeration,
the contexts and the perspective exchange, the symmetry tables, export
parity with the fake-quantised forward, the sizes of the contract, the
format 6 to 8 conversion, the format 8 fixtures, the importer on
synthetic windows through the engine, a tiny training run with exact
resume, the collector and the labeller on a scripted `bot`). `cargo xtask
check` runs them on the CPU.
