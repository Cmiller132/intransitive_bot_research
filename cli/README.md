# cli

The `bot` binary. It is the only place that knows every model crate, so the
match runner stays model-agnostic.

```
bot eval  --candidate sq:runs/x/export.onnx [--reference sq:weights/sq_g128.onnx] [--pairs 32] [--sims 32 | --move-ms 100] [--reference-move-ms 100 | --reference-sims 32] [--threads 4] [--player-threads 1] [--seed 0] [--opening-plies 8] [--records games.jsonl] [--stream]
bot play  --first sq:a.onnx --second sq:b.onnx [--sims 32 | --move-ms 100] [--player-threads 1] [--seed 0] [--opening-plies 8] [--leaves leaves.jsonl]
bot rpsi  --player sq:model.onnx [--move-ms 250] [--max-move-ms 250] [--sims 32] [--threads 4] [--name intransitive_bot]
bot analyse --engine conv:model.onnx [--threads 1]   # or sq:<onnx>, nnue:<file.nnue>
bot nnue convert --input prototype.nnue --output model.nnue
bot nnue validate --model model.nnue --input positions.jsonl
bot nnue diagnose --model model.nnue --input positions.jsonl --nodes 20000 [--threads 1]
bot --version
```

`--stream` prints every game's start, moves (with the mover's search stats)
and end as JSON lines while the eval plays, then the report as an
`event: report` line. `analyse` reads JSON requests on stdin and answers one
line each (match/README.md, analysis); `sq:`, `conv:` and `nnue:` engines can
be analysed.

A player spec is `<model>:<path>`, one of `sq:`, `conv:` and `nnue:`, or
`rpsi:<command line>` for an external engine speaking RPSI (the command is
split on whitespace and started as a child process); paths are resolved
from the current directory. `--reference` defaults to the named eval reference,
`sq:weights/sq_g128.onnx`, so `bot eval` is run from the workspace root.
NNUE specs accept `nnue:<path>?hash=<MiB>`, for example
`"nnue:runs/nnue_h768/best.nnue?hash=128"`. The hash budget is 1-2048 MiB,
default 64, per player and shared by its search workers. Table slots round
down to a power of two, so allocation can be below the budget. This applies
to eval, play and RPSI seats; malformed or unknown options are rejected.
Game search records report the requested budget as `hash_mib`. Diagnose
continues to use 64 MiB.

For timed evaluation, --reference-move-ms overrides the reference's budget
while --move-ms remains the candidate budget; it requires --move-ms.
Budgets follow player identity through both seats of each opening pair.
For example, --move-ms 500 --reference-move-ms 250 measures doubled time.
Alternatively, --reference-sims N gives the reference a fixed simulation budget
while the candidate keeps --move-ms; it requires --move-ms and conflicts with
--reference-move-ms. For example, --move-ms 250 --reference-sims 32 compares
a timed candidate with a 32-simulation reference. Reports include move_ms,
sims (0 for a timed candidate), and the effective reference_move_ms and
reference_sims (the unused reference unit is null). Without either override,
both players receive the candidate budget.

Adding a model means adding one arm to `player_from_spec`. Results are printed
as JSON.

`bot --version` prints the crate version followed by the git short hash and
`-dirty` when the checkout had uncommitted changes at build time (build.rs);
built outside a git history, the version alone.

`eval --threads` sets concurrent pairs; `--player-threads` sets threads within
one player. NNUE timed play reserves 20 ms for overhead. Simulation play maps
n simulations to n * 2,500 nodes on one worker (120 s safety ceiling); use timed
eval for equal wall-time comparisons. Its records include nodes, depth, score
and canonical PV under `search`. `partial=true` marks a move found by a fully
searched root child in an unfinished iteration; `depth` remains the last completed
depth. The same fields appear in `nnue diagnose`. `play --leaves` writes the first player's
static leaves as JSONL to a new file, including root, canonical action path,
board codes, since_capture, ply, clock, raw and score; tracing affects speed.

`nnue convert` accepts only the frozen prototype's v3 H512/D32 network and
creates a v6 network with zero clock rows. The runtime
accepts v6 with any positive hidden width divisible by 32, a 32-unit dense head,
one or four output buckets and learned clock rows. Dimensions must fit the file
and integer arithmetic. Validation inputs
are JSONL objects with board (81 codes), since_capture, ply and clock (50-200);
optional raw is checked within 1e-6. Every child accumulator is checked against
a full refresh. Diagnose requires site clock=200 and additionally searches each position with a fresh
64 MiB table at the requested node budget, with a 60 s safety ceiling.


For CPU profiling, build a separate diagnostic binary with
cargo build --release -j 4 -p cli --features nnue/profile --target-dir target/nnue_profile.
Its one-thread nnue diagnose output adds profile counters for accumulator
updates, the dense head (including its residual readout), linear readout,
move generation, ordering and TT operations. Each category samples one call
in 64 and reports calls, samples and sample_ns. These are instrumented timing
estimates; use an ordinary build for throughput and strength measurements.
Normal builds contain no profiling probes.

Profiling also reports profile_timer_ns, calibrated with empty sampled scopes.
Subtract this estimate from each sample mean before estimating category time;
the remaining instrumentation and sampling error are not removed by that correction.
