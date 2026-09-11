# cli

The `bot` binary. It is the only place that knows every model crate, so the
match runner stays model-agnostic.

```
bot eval  --candidate sq:runs/x/export.onnx [--reference sq:weights/sq_g128.onnx] [--pairs 32 | --sprt pairs.jsonl] [--resume] [--sims 32 | --move-ms 100] [--reference-move-ms 100 | --reference-sims 32] [--threads 4] [--player-threads 1] [--seed 0] [--opening-plies 8] [--records games.jsonl] [--stream]
bot play  --first sq:a.onnx --second sq:b.onnx [--sims 32 | --move-ms 100] [--player-threads 1] [--seed 0] [--opening-plies 8] [--random-moves K --random-from P --random-to Q] [--random-seed S]
bot rpsi  --player sq:model.onnx [--move-ms 250] [--max-move-ms 250] [--sims 32] [--movetime 100 | --fixed-sims 80] [--lock-sims] [--threads 4] [--name intransitive_bot]
bot analyse --engine conv:model.onnx [--threads 1]   # or sq:<onnx>, nnue:<file.nnue>
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

Adding a model means adding one arm to `player_from_spec` and one to
`analyser_from_spec`. Results are printed as JSON.

`eval --sprt pairs.jsonl` runs the fixed C1 pentanomial test: pair-average
candidate scores 0, .25, .5, .75, 1; hypotheses .50/.52, alpha=beta=.05,
LLR bounds ±log(19), batches of 16, first check 128, cap 3,008 pairs. It
conflicts with `--pairs` and `--records`; the journal contains both game
records for every retired pair. At most 16 concurrent workers are supported.
The report's `sequential` object includes the protocol, its SHA256 digest,
five counts, LLR and `stop_reason`: accept, reject, inconclusive or invalid.
Any forfeit, censored game or numerical failure invalidates the test. Acceptance
rejects the null; it does not establish a gain of at least two score points.
Intervals at sequential stopping are descriptive only. Fixed-count eval keeps
its existing report and flat game-record format.

Native player specs infer the network file to hash. External RPSI seats require
`--candidate-network PATH` / `--reference-network PATH` for their actual weights,
and an explicit executable path as the first token of their command. Use direct
engine executables; wrapper scripts and undeclared auxiliary model files are
not covered by this identity contract. The coordinator binary, each engine
binary and network, specs, player threads, CPU affinity, settings and openings
are bound into the protocol. Repeat the same options with `--resume` to continue;
any identity change is rejected. A completed journal returns its existing verdict.

The journal has an exclusive writer lock. Both colours and the entire batch
finish before a stopping decision. Records retire in opening-id order and each
batch is synced. Resume retains complete newline-terminated pair records and
discards only a torn last line; complete malformed lines fail. Pairs not yet
journaled are replayed after a crash. Wall-clock timings are not deterministic.

Numerical/reference tests run with `cargo test -p match --test sprt_reference`.
Before live use, run the production-rule simulation with
`cargo test -p match --release --test sprt_calibration -- --ignored --nocapture`.
The frozen Python fixtures live at `match/tests/fixtures/sprt.json`; production
statistics and the maintained calibration are Rust only.

`bot --version` prints the crate version followed by the git short hash and
`-dirty` when the checkout had uncommitted changes at build time (build.rs);
built outside a git history, the version alone.

`eval --threads` sets concurrent pairs; `--player-threads` sets threads within
one player. NNUE reserves 20 ms under host-provided time budgets. The RPSI
seat option `--movetime N` replaces `go sims` with a self-imposed N ms budget;
NNUE uses it without a reserve; `--fixed-sims N` instead replaces `go sims`
with a fixed count of N simulations; `--lock-sims` plays exactly `--sims`
simulations on every move whatever the host gives (a fixed-strength site
bot), except that a clock under 800 ms still cuts a move to one simulation.
Otherwise, explicit host movetime and site clocks take
precedence and retain the reserve. Simulation play maps
n simulations to n * 2,500 nodes on one worker (120 s safety ceiling); use timed
eval for equal wall-time comparisons. Its records include nodes, depth, score
and canonical PV under `search`. `partial=true` marks a move found by a fully
searched root child in an unfinished iteration; `depth` remains the last completed
depth. The same fields appear in `nnue diagnose`.

The runtime accepts format 6 and format 8, hidden widths divisible by 32 in
32..1024, one output head and dense width 32. Header scales and exact payload
lengths are validated. Validation inputs
are JSONL objects with board (81 codes), since_capture, ply and clock (50-200);
optional context, ids (42 slots per half), raw and raw6 check the shared fixtures.
Raw tolerance is 1e-12; raw6 is used for version 6. Every child accumulator is checked against
a full refresh. Diagnose requires site clock=200 and additionally searches each position with a fresh
64 MiB table at the requested node budget, with a 60 s safety ceiling.
Its accumulator work counters and timing denominators are documented in
models/nnue/README.md; timings are instrumented worker spans.


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


For data collection, play --random-moves K --random-from P --random-to Q
injects up to K uniformly random safe moves at distinct plies sampled uniformly
without replacement from the inclusive interval [P,Q]. Plies are zero-based
indices into the game record's moves, including the opening; P must be at least
--opening-plies. K must fit the interval. With K > 0 both bounds are required.
--random-seed defaults to --seed but uses a separate RNG, leaving opening
generation unchanged. The default K=0 preserves ordinary play and record output.

Each injection is uniform over legal actions after excluding engine::tactics
loses_in_two moves. If none is safe, the player chooses normally; if the game
ends, later slots are unused. Skipped slots are not rescheduled. A random move
can itself win or draw. Both players observe it, but choose is bypassed and its
stats entry is null. GameRecord.random_plies lists only injections actually
played and is omitted when empty. These options affect play only, not eval.


When auditing paired records, identify the model from first/second and the
actual mover; the reference changes seat between the two games. A timed NNUE
move reports sims=0 because this field counts Gumbel simulations. Its search
work is in search.nodes; zero sims alone does not mean NNUE skipped search.

## Self-play generation

`bot selfplay` runs independent, one-thread NNUE games in one process. It
shares immutable weights, keeps each game's TT between moves, and clears
TT and ordering state before the next game. Default concurrency is 16.

```text
bot selfplay --player nnue:runs/nnue_candidates/ft_gpu150a.nnue?hash=64 --nodes 250000 --games 20000 --threads 16 --seed 20260912001 --opening-plies 8 --random-moves 2 --random-from 8 --random-to 40 --multipv 1 --multipv-margin 60 --max-plies 1024 --records runs/generation --shard-games 256
bot selfplay --records runs/generation --verify
```

`--nodes` includes quiescence; there is no search SMP. The search depth limit
is 64. Random count defaults to zero; distinct injection slots use the
inclusive, zero-based window. Opening moves and injections use the existing
uniform safe-legal sampler. If every move loses immediately, the slot is
marked skipped and normal search plays. `--multipv` currently accepts only
1; the nonnegative margin (default 60) is recorded but unused.

Pin the parent before starting the generator. For Windows PowerShell, the
following sets this shell and its children to logical CPUs 16-31 and below
normal priority; run the command above in that shell:

```powershell
$generationProcess = [Diagnostics.Process]::GetCurrentProcess()
$generationProcess.PriorityClass = 'BelowNormal'
$generationProcess.ProcessorAffinity = [IntPtr]4294901760
```

Python's `psutil.Process().cpu_affinity(list(range(16,32)))` before
`subprocess.run([...])` is equivalent for affinity. Linux can use
`taskset -c 16-31 nice -n 10 bot selfplay ...`. The manifest records the
actual allowed logical CPUs, effective settings, model/binary/rule SHA256,
seed derivation and TT policy. Rules hashes normalize source line endings.

Each gzip JSONL line extends `GameRecord`; see match/README.md for the schema.
Shards have fixed game-ID ranges, zero gzip timestamps and atomic publication.
The atomically replaced manifest lists their sizes, hashes and counts. A
single writer lock prevents concurrent writers to the same directory.
Progress goes to stderr; the final stdout JSON reports generated games,
eligible roots, total nodes, summed search time and generation-wall rates.
Eligible means a completed label at ply >=16, before deduplication.

Resume by repeating **all** original options and adding `--resume`. Model,
binary, rules, effective settings (including target games, worker count and
CPU mask) must match exactly. Each game has its own derived seed. Per-move
journals preserve flushed decisions; completed games are stored durably
before joining a shard. Ctrl-C/termination preserves completed labels and
censors unfinished games with null outcomes. After an abrupt process exit,
resume recovers complete journal lines and ignores a torn final event.
Pending games wait for their full shard range; they are not silently lost.
Published shards and shard reconstruction from persisted records are
byte-identical. A fresh rerun reproduces game decisions, not measured
elapsed times. Per-move flushes protect against process exit; the last
un-synced events are not guaranteed across power loss.

`--verify` needs no model: it checks every published shard hash and replays
every game through the engine, including frames, root board fingerprints,
legal actions, random guards, engine proofs and terminal/censored results.
It does not re-search labels or audit unpublished active journals.

The summary also reports `worker_game_seconds`,
`worker_outside_search_seconds` and `publication_seconds`. These are summed
wall durations, not CPU cycles: worker game time includes scheduling and
journalling; publication runs on the coordinator and can overlap workers.
