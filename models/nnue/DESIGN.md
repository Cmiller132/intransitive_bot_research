# NNUE design

## 1. Purpose and ownership

The NNUE is the site bot's evaluator: a sparse, incrementally updated integer network read by an
alpha-beta search on the CPU. The workspace crate `nnue` holds the evaluator, the search and
`NnuePlayer: match::Player`; the Python package `nnue` (models/nnue/py) holds the features, the
model, the quantisation-aware trainer, the exporter, the dataset importers, the labellers and the
experiment runner. Rules and exact tactics come only from `engine`; the crate keeps no second
implementation of moves, apply or terminal checks, and the Python side reaches the rules through the
`engine` wheel. `bot` builds the player from `nnue:<path.nnue>`; the arena seat and the site binary
change only by the user's decision.

Two kinds of judgement, never mixed. A correctness or performance gate decides whether a change
enters the code: parity fixtures (feature ids and integer accumulators equal exactly between the
NumPy oracle and the Rust kernels, the scalar and SIMD kernels bit-identical, the Python and Rust
f64 evaluations within 1e-12; incremental updates equal a refresh over recorded trajectories),
identical fixed-node search signatures where the change claims to leave decisions unchanged, and a
throughput floor measured from outside on a frozen binary. A strength decision decides whether a
change is accepted or a network promoted: paired games through `bot eval` only, under a
preregistered manifest, either a fixed count of pairs with the bootstrap interval or the sequential
test of `bot eval --sprt`; the verdict is the runner's, from the recorded games. A candidate is in
one of four states, and the documents say which: proposed, implemented, correctness-verified,
strength-accepted.

The deployed network is identified by its file hash and its sidecar (run, epoch, objective, config,
datasets), never by a name; an unjudged candidate is not the incumbent. README owns the commands,
the experiment manifests and verdicts own recipes and results, claude_notes.md is the session log;
DESIGN states the contracts and the bounded findings that shaped them.

## 2. Evaluator and formats

The evaluator uses two canonical perspectives: the side to move and its opponent. The opponent frame
reflects squares across the anti-diagonal and swaps piece colours. Both halves share one feature
table and bias. Engine owns board codes, legal transitions and attack predicates.

RPSNNUE1 is little endian and accepts versions 6 and 8 only. The 36-byte header contains the
eight-byte magic, version, feature count F, hidden width H, QA=255, QB=64, f32 evaluation scale 600
and head count 1. H is a multiple of 32 in 32..1024. Version 6 requires F=1,004; version 8 requires
F=13,640. The payload is bias H i16, feature-major weights F*H i16, mover/opponent readout 2H i16,
readout bias i32, dense width u32=32, dense weights 32*2H i8, dense biases 32 i32 and residual
readout 32 i16. The total length is exactly 236 + H*(2F+70) bytes. Unsupported constants, truncation
and trailing bytes are errors.

For an occupied square s with perspective-relative code t in 1..6, the piece-square index is
81*(t-1)+s. Format 6 uses that row directly. Format 8 adds 486*c, where
c=min(r,2)+3*min(p,2)+9*min(sci,2), using the perspective's opponent rock/paper/scissors counts.
There are 27 contexts. Each perspective computes its own context; counts are not combined across
colours. Captures 4->3 and 3->2 preserve the capped component; 2->1 and 1->0 change it.

Attacked-piece rows are unconditioned, at F-518 + 81*(t-1)+s, active exactly when
engine::tactics::is_attacked holds for that occupied square. Two clock rows are active: one at F-32
for since_capture and one at F-16 for max(capture_clock-since_capture,0). Each uses the greatest
lower bound in [0,1,2,3,4,6,8,12,16,24,32,48,64,96,128,160]. Each perspective has 42 feature slots:
up to 20 pieces, up to 20 attacked pieces, two clocks. Unused slots carry sentinel F and contribute
zero; the sentinel is not stored in the served file.

Each accumulator is bias plus its active rows in i32, without saturation before the activation
clamp. Concatenate mover and opponent lanes, and let x=clamp(acc,0,255). The main readout is
sum(x*x*output)/(255*255*64) + output_bias/64, with an i64 sum. Dense input is nearest-integer
x*x/255; an exact half tie is impossible. Dense activations are
clamp(round_ties_even((dot(input,dense_weights)+dense_bias)/64),0,255). Add
sum(dense_activation*residual_output)/(255*64) to obtain raw. The engine score is
clamp(round_ties_even(600*raw),-28000,28000), below the mate-score band. Scalar, AVX2 and supported
AVX-512/VNNI paths implement the same arithmetic. Accumulator and dense scratch buffers are reused.

Conversion from format 6 to 8 repeats the 486 piece rows in every context and copies the bias,
attacked/clock rows and entire head unchanged. Only version/feature-count header fields and the
repeated table region differ. Both files therefore evaluate the same state identically. Training
represents a contextual row as shared_base[i]+residual[c,i]; conversion initializes the residual to
zero. Quantisation and range constraints apply to the served sum, which is flattened at export.
Factorisation does not add a second inference representation or change the file layout.

## 3. Accumulator and search

Exact material counts and clocks travel independently of deferred accumulator vectors. A pending
child records its before/after FeatureStates and the board/move needed for materialisation. A move
swaps the material sides and decrements the captured type on the new mover's side. A pending chain
obtains its parent state from the parent's pending after-state; it does not read an unresolved
accumulator's old state.

On evaluation, each half resolves independently. Its source is the parent's opposite half. In format
8, a changed capped context refreshes only that half from the bias and every active row family on
the child's board, without resolving its ancestors. An unchanged half resolves its source and
applies piece, capture, attacked-piece and clock-row differences. Format 6 uses this incremental
path for both halves. Ready bits prevent duplicate resolution; the child's FeatureState is committed
when both halves are ready. Replacing a pending sibling or finishing a search discards unresolved
work. Root refresh and refreshed child halves write into allocated buffers rather than allocating a
vector per node.

Search uses iterative deepening, aspiration windows, principal-variation search, ordered moves,
late-move reductions and tactical quiescence. Rule outcomes and exact goal/evasion predicates come
from engine. Terminal move outcomes take precedence over a child's static evaluation. The quiescence
budget and MAX_PLY frontier bound tactical continuation; the evaluator does not supply separate race
or rule logic. A node count includes quiescence nodes; qnodes is a subset, not additional work to
add to nodes.

The retained mechanisms are two killers per ply, quiet-move history with bounded updates,
history-aware reductions, reverse futility and late quiet futility at non-PV depth at most two with
tactical safety checks, partial-root selection, stalemate-first quiescence, and lazy SMP over a
shared table. Quiet futility requires an already searched move; it cannot prune every legal move.
Iterative deepening stops starting another iteration after 72% of the time budget has elapsed. This
build has no continuation history, capture history, null move, one-ply extensions, margin-based lazy
evaluation or shortened quiescence horizon.

The transposition key includes the canonical board and capture-clock state. Bound entries
distinguish exact/lower/upper scores and retain a legal ordering move; static evaluation is
separately valid. Mate-distance scores are normalised when stored/retrieved. Interrupted ancestors
and frontier-only results do not establish reusable search bounds. The single-thread path uses its
local table; multiple search workers share a table with coherent, nonblocking entry access. A
contended shared entry is a miss.

NnuePlayer refreshes the root and searches under its supplied budget. A simulation budget means at
most 2,500 nodes per simulation on one search thread, with a 120-second safety limit. A timed
external host deadline reserves 20 ms; a seat's own --movetime budget uses the full time through
choose_self_timed. Result depth denotes the last completed iteration; partial identifies an improved
playing choice from an unfinished iteration. Completed-root labels use the last completed iteration,
independently of that partial choice. Newgame clears search state. Analyser searches start with
cleared TT state.

Diagnose exposes exact-count transitions, capped-context changes, full and half refreshes,
incremental updates, row additions/removals and discarded pending work. materializations counts
halves, including two per full root refresh. The accounting identities are materializations =
2*full_refreshes + half_refreshes + incremental_updates and 2*prepared_edges + 2*full_refreshes =
materializations + pending_halves_discarded after pending cleanup. A partly resolved node can
contribute both materialised and discarded halves; pending_nodes_discarded is not a count of wholly
unevaluated nodes. Refresh/update timers are enabled only for diagnostic accounting and include
their measurement overhead. Performance acceptance uses the same frozen unprofiled binary, identical
roots/budgets and external elapsed/RSS observation. Correctness and speed do not establish playing
strength.

## 4. Data and trainer

### 4.1 Datasets

A dataset is one directory under runs/nnue_data/<set>/ of columnar NumPy arrays, memory-mapped when
read, plus provenance.json. Every row is one position in the mover's frame with an exact clock
context and one label: board u8 (N, 81) with codes 0 empty, 1-3 own rock, paper, scissors, 4-6
enemy; since_capture u16; ply u16; capture_clock u16 (the game's clock); target f32 in [-1, 1] from
the mover's view; weight f32; kind u8; outcome i8 with outcome_ok; source u8; game u32 (episode
within the set); orbit u64 (the board's symmetry-orbit hash); split u8. Kinds: 0 a played root
labelled with its lambda return, 1 the child of a visited candidate labelled minus the candidate's
completed Q, 2 an exact proof (1, -1 or 0), 3 a teacher search value, 4 a raw network value, 5 a
prototype replay row with averaged counters. Splits 0 train, 1 validation, 2 test, assigned by
episode first, with no symmetry orbit across splits. Sources 1 conv windows, 2 and 3 the prototype's
replay and searched arrays, 4 student positions, 5 human games, 6 the searched roots of `bot
selfplay`.

provenance.json records the producer and its inputs (the window hashes, the teacher engine, binary
and network hashes, simulations, rules, clock, sampling seed and selection); its hash is the
dataset's identity everywhere else (the trainer's inputs, the runner's pinned identities, the id
cache). A dataset is immutable once written; that is the contract, not a check: arrays changed under
an unchanged provenance file go undetected, a labeller that skips an existing output has not
compared its content, and the runner refuses only a provenance file whose hash moved. The optional
cache ids8.npy holds the format-8 feature ids of every row (int16, (N, 2, 42)) so a batch is a
gather rather than an encode; ids8.json binds it to the encoder's signature (the ids of 64 fixed
boards), the row count and the provenance hash, and a cache that is not this dataset's under the
current encoder is refused. `python -m nnue.data encode <set>` writes it.

### 4.2 Importers

`python -m nnue.importer conv --run <dir> --out <set> [--iterations a-b] [--children 16]` reads the
replay windows of a conv run (schema 2; plain or zstd; the bytes hashed, the shapes validated, any
other schema refused). The played root gets its lambda return (kind 0, weight 0.25); every visited
candidate's nonterminal child, applied through the engine, gets minus that candidate's completed Q
(kind 1, weight 1); outcomes stay on played rows. Episodes are recovered from the environment order
across contiguous windows; exact (board, since_capture, clock) contexts are merged with the weight
capped at 4. `nnue.importer human --file <export> --out <set>` gives every played root of a finished
site game its outcome (kind 0, weight 0.25, outcome recorded), merged the same way. `nnue.importer
selfplay --records <files> --out <set> [--min-ply]` imports the searched roots of `bot selfplay`
records with the values their searches completed (source 6). The two label domains stay apart:
producer-rule rows as recorded, site-rule rows (clock 200, no shaping) from fresh searches; a recipe
names what it mixes and at what share.

### 4.3 Labelling

`python -m nnue.label --input <set> --out <set> --engine <spec> [--sims N] [--workers W] [--rows N]
[--seed S] [--select <set>]` gives every row an exact proof from the engine (a win at once is 1,
every move losing in two is -1; kind 2) or the root value of `bot analyse` under the named engine
and `sims` (kind 3), with the chosen line's Q kept beside the root value in the provenance
statistics. The output keeps the input's boards, counters, games, orbits and splits and records the
site clock, under which the teacher searched. `--rows N` labels a seeded sample; `--select
<shallow>` labels instead the N rows whose input label differs most from their label in a shallow
pass over a sample of the same input (the largest disagreements between the two labels, which do not
say which side misjudged the position; the seeded sample is the control). Selection refuses a
shallow set that is not a subset of the input, duplicates on either side, non-finite gaps and too
few rows.

Roots are searched one `bot analyse` process per chunk, the answers matched by request id. A root
whose search fails, or whose process died before answering, is searched once more under the same
budget, never replaced by another root; roots still failing after the retry are dropped. Provenance
records the engine (spec, binary hash, network hash), the input set's provenance hash, the sample
(rows, seed), the selection set, `retries`, `errors`, and `cost`: the request slots submitted (a
slot behind a crashing row never starts), the processes, their wall seconds, the nodes the answers
report (a lower bound on the search done) and the attempts whose node count is unknown. A
preregistered step names rows requested and rows usable (the count after the retry it accepts), and
the verdict reports the final counts.

`python -m nnue.gpu_label --input <set> --out <set> --ckpt <checkpoint> [--teacher conv|sq] [--sims
128] [--batch 4096]` labels the same rows in bulk with conv's (or sq's) batched Gumbel search from a
training checkpoint when the GPU is free: the exact proofs of the next set are computed on the CPU
while the GPU searches the current one, several input/output pairs share one loaded teacher, the
teacher's repetition rule is recorded. It is not in the test suite. GPU use is the user's decision.

### 4.4 Model

One network, whose integer evaluation, rounding, clamping and file layout are section 2's: feature
tables of int16 rows shared by both perspectives (the mover, and the anti-diagonal reflection with
colours swapped), H hidden (a multiple of 32 in 32..1024; 512 in the lineage), squared clipped ReLU,
one int16 readout, one 32-wide int8 dense residual; value tanh(raw), search score round(600 raw).
Version 6 has 1,004 rows (486 piece-square, 486 attacked-piece, 16 elapsed-clock and 16
remaining-clock rows over the bounds 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160
plies); version 8 has 13,640 (the piece-square rows repeated in 27 opponent-material contexts c =
min(r, 2) + 3 min(p, 2) + 9 min(s, 2), then the same attacked and clock rows). The encoder and the
trainer fill 42 slots per perspective (20 pieces, 20 attacked pieces, two clock rows); an unused
slot carries the sentinel id F, which contributes nothing and is not a row of the served file.

Training keeps float parameters that the exporter rounds: the piece, attacked and clock tables
clamped to [-8, 8], the readout and residual to their int16 or int8 ranges, and constrained after
every step. A format 8 model is factorised (Stockfish's feature factorisation): W[c, i] = base[i] +
delta[c, i] with the residuals zero at the start and the clamp applied to the sum by moving the
residual, so a format 6 `--init` becomes the shared factor (`widen_contexts`, byte-identical to
`nnue.export convert`, which repeats the piece rows into every context and copies every other byte)
and a format 8 run starts from the incumbent's evaluation. `widen_hidden` pads a narrower network to
a wider one that evaluates identically at first: every new column gets its own small seeded feature
rows, the initial bias and zero readout and dense columns (the earlier zero-row padding left the new
columns identical and undifferentiated, item 33). The PyTorch fake-quantised forward agrees with the
file's integer evaluation within 2e-6 raw; the NumPy oracle over the file bytes and the Rust kernels
agree exactly on feature ids and integer accumulators and within 1e-12 on the f64 value, the scalar
and SIMD kernels bit-identically (section 2).

### 4.5 Training

`python -m nnue.train --run <name> --data <set>:<share> [...] [--<field> value]`; every field of
`Config` is a flag and runs/<name>/config.json records the values used: hidden 512, batch 2048,
steps_per_epoch 500, epochs 20, lr 3e-4, weight_decay 1e-5, warmup_steps 200, lr_floor 0.15,
qat_start_epoch 0, raw_weight 0.02, symmetry_weight 0.2, version 8, threads 4, device cpu, seed 0,
val_rows 8192, grad_clip 5, patience 6, init, resume, stop_epoch. Batches are drawn from the named
datasets by fixed shares with replacement (the shares normalised, each count rounded and at least
one row, the residual of the batch going to the last dataset), gathering only the columns a batch
needs when every set has an id cache. Every batch is augmented by a random one of the six board
symmetries and paired with a second view; the loss is the tanh mean squared error against the target
plus raw_weight times a bounded logit term, plus symmetry_weight times the squared difference of the
two views' values. Quantisation-aware training (fake quantisation of every table in the forward) is
on from qat_start_epoch; AdamW with linear warmup and a cosine schedule to lr_floor of the rate;
gradients clipped; a non-finite loss or gradient stops the run.

Validation after every epoch, per dataset and under all six symmetries at qat: error statistics
overall and by stratum (pieces at most 4, 5-12, over 12; since_capture over 100), the symmetry
spread, and the selection stratum, every row whose kind is not 0 (a hard game outcome is -1, 0 or 1
and a lambda return may lie between; either is one draw from the value, whose error floor would
swamp the rest). The selection objective is the share-weighted validation MSE over the selection
strata, every kind-0 row excluded wherever the stratum exists and a dataset whose validation rows
are all kind 0 contributing its overall MSE; `best.pt` and `best.nnue` (with a sidecar: run, epoch,
objective, config, datasets, hash) follow it; patience stops a run after that many epochs without
improvement; stop_epoch pauses after N epochs with the schedule unchanged. `latest.pt` is written
atomically after every epoch with the weights, the optimizer, the torch and augmentation RNG states,
the sampler state, the epoch, the step, the best objective, the config and the run's inputs (the
data parts, every dataset's provenance hash, the init file's hash). `--resume` restores all of it
and refuses other settings, another init or other datasets; log.csv is truncated to the checkpoint's
epochs before the run continues, so the checkpoint, never the log, says how far a run got. `--init`
takes a .nnue file (widened to the run's hidden and version as above) or a .pt checkpoint's weights.

## 5. Measurement and experiment runner

### 5.1 C1 paired measurement contract

bot eval is the strength authority. Each opening produces two games with candidate/reference seats
swapped. Budgets follow the player, not its colour. The independent sampling unit is the complete
opening pair. Candidate pair scores are 0, 0.25, 0.5, 0.75 or 1; the five counts form the
pentanomial sample. Equal display names are permitted: pair order and frozen artifact provenance
identify the roles.

Sequential evaluation compares expected pair score s0=.50 with s1=.52 using constrained pentanomial
maximum likelihood, alpha=beta=.05 and log-likelihood boundaries +/-log(19). The likelihood applies
a 0.001 count to empty cells. Results retire in opening order in batches of 16 pairs, with the first
decision at 128 and a cap of 3,008. A boundary crossing accepts/rejects; reaching the cap without
crossing is inconclusive. Forfeit, PlyCap or Interrupted invalidates the test. Both colours and the
current batch finish before a stop; a resumed incomplete batch is completed under the same rule. A
malformed protocol or complete journal record is an error.

The journal starts with protocol and protocol_digest, then numbered pair records with the two
GameRecords. The protocol binds settings, generated opening sequence, coordinator/build/network
identities, player thread count and logical process affinity. Resume requires equality, reconstructs
the stopping state and ignores/truncates only an unterminated final record. A terminal complete
journal reconstructs a report without constructing players. Ordinary --resume can continue a running
journal; it is not a general read-only inspection command.

The report's sequential object contains protocol, protocol_digest, counts, llr, stop_reason, error
and intervals_descriptive_only directly. Stop names are running/accept/reject/inconclusive/invalid.
End names are Goal/Elimination/Stalemate/CaptureClock/Forfeit/PlyCap/Interrupted. Winner 0/1 means
the game's first/second player; null means a draw or censorship as distinguished by End. For valid
games the first four endings apply. Complete pair counts exclude invalid pairs.

Fixed-count comparisons use paired uncertainty under their preregistered rule. Sequential-stop
bootstrap intervals are descriptive and cannot supply a fixed-count confirmatory lower-bound
decision. Calibration uses simulated pair distributions and an independent likelihood/trajectory
reference; scripted journal fixtures test serialization, pair accounting and terminal replay, not
model strength. Search changes use identical frozen weights in both external build seats; network
comparisons use the same frozen search implementation. Exact commands, seeds, hashes and decision
rules belong to the experiment manifest, with outcomes in its verdict.

The runner audits C1's report against its bound protocol and game records before using its stopping
result in the experiment verdict. A fixed-count match preregisters either lower_bound_above (the
paired bootstrap interval's lower endpoint strictly exceeds the threshold) or score_at_least (the
observed score is at least the threshold).

### 5.2 Experiment runner

`python -m nnue.experiment pin <experiment.json>`, `run <experiment.json> [--only train|match]` and
`verdict <experiment.json> [--audit]` run one preregistered experiment from one manifest (schema 2)
in its own directory under runs/nnue_gauntlets/<name>/: the frozen `bot`, the shared `network`, the
CPU `affinity`, `identities`, `arms`, `matches`, `rules`. `pin` hashes every fixed input (the bot,
the network, every `.exe` and `.nnue` side, every arm's init, every dataset's provenance file) into
`identities` before anything runs; `run` and `verdict` refuse a file that differs. The manifest is
written before the first game and not amended after a result is seen; a change of protocol is a new
manifest.

Arms train through `nnue.train` with the arm's config, init and data; `stops` are epoch counts at
which training pauses, the checkpoint is retained as epoch<N>.pt (its own epoch count checked) and
resumed exactly; endpoints (`latest@N`, `latest`, `best`) are exported from those checkpoints. An
existing endpoint is reused only when its sidecar hashes the file and carries the epoch its source
names and, for an arm the runner trains, the arm's config, data shares, dataset hashes and init hash
(an arm that names an existing run binds to the file, epoch and config); there is no fallback for an
unbound endpoint: a missing file, a missing, unreadable or malformed sidecar or one that does not
match is a broken endpoint, and a match that used one is invalid. CUDA exposure is experiment-wide:
the runner hides the GPU from every process it starts unless any arm names cuda, in which case
device 0 is exposed to all of them, and an environment that already sets CUDA_VISIBLE_DEVICES (to -1
included) overrides either; paired arms that are compared declare the same device (the trainer alone
hides CUDA by default).

A match names its candidate and reference as `<run>:<endpoint>`, a `.nnue` path or a `.exe` path,
the clock (`move_ms`), the seed, the opening plies (8), the concurrent pairs (8) and the player
threads (1). Network sides play under the frozen `bot`; `.exe` sides are search builds seated
through `rpsi:<exe> rpsi --player nnue:<network> --threads T` on both sides, so the process overhead
is equal and the weights are the experiment's shared network; a seat path with whitespace is
refused. A fixed-count match plays `pairs` seeded openings with the colours swapped and records
every game; a sequential match (`sprt`) runs C1's test with a journal that is resumed when it exists
and stopped by its own rule, streamed so the runner records each invocation's timing, finished,
failed or interrupted (launch time; pairs before and committed after, null when the journal could
not be read; process, startup, play, active and finish seconds kept apart as diagnostics; new pairs
per active second; per-batch occupancy and the idle share over complete batches; the aggregate is
committed pairs over summed active seconds and over summed process seconds, never paused time, and
is null beside the known count and the number of unknown invocations when any count is unknown),
written before the report is cached.

A played match is bound to its manifest entry, the affinity, the hashes of the files it used and the
hashes of its own report and journal or records (`<tag>.match.json`); `run` plays a match whose
report is not bound again (resuming a sequential journal) and `verdict` judges such a report
invalid; neither keeps an unbound report as a result. The verdict audits every match's recorded
games against its report: for a sequential test the journal header's protocol and digest must equal
the report's, the protocol's settings (seed, clock, opening plies, workers, player threads) must
match the manifest and its provenance hashes (coordinator, each side's binary and network) the
pinned identities; every pair must hold two games of one opening with the seats swapped and endings
among Goal, Elimination, Stalemate and CaptureClock (C1 censors Forfeit, PlyCap and Interrupted);
pairs, complete pairs, the pentanomial counts and the candidate's wins, draws and losses must equal
the report's; forfeits or an error on a completed test invalidate it; and, only for a journal that
exists, is bound and passed that audit, C1 itself replays it (`bot eval --sprt <journal> --resume`,
which plays nothing once the test has stopped) and its report must agree, so the verdict can never
start or continue a test. A fixed-count match is reconciled the same way from its records. A
malformed journal or record is a structured problem of its match, never an exception, and a journal
that cannot be read gives an unknown pair count in the timing, never a fabricated one. The verdict
records, per match, the numbers, the descriptive counts (endings, the first mover's score, distinct
openings, mean plies), the problems and validity; per rule (`lower_bound_above` and `score_at_least`
for fixed-count matches, `sprt_accept` for sequential ones) whether it is complete and met; a
sequential test's bootstrap interval is descriptive only.

Every `bot eval` runs inside a Windows job object (created suspended, assigned, then resumed, so
every seat inherits it) that is terminated when the invocation ends, however it ends: an
interruption or a reader error kills the tree, a coordinator that exits abruptly leaves no seat
behind, one that closes its output without exiting is killed after a grace period, and the job's
member list names any survivor. Every cleanup step checks its result; a step that cannot be proved
(a failed termination, query or handle close, a reader still holding the pipe) is recorded and keeps
the reservation like a survivor; that invocation ends with an error, its result discarded and its
timing kept, and nothing is launched again while the failure or a surviving member stands. One heavy
phase at a time: the runner takes the compute lock (runs/nnue_plan/compute_lock, created atomically,
never taken from a live owner) for the whole run, its final verdict included, at below-normal
priority on the declared CPUs, and releases it only when no child of its own, no job member and no
unproved cleanup remains; otherwise it keeps the lock and `lock release experiment <pid>` frees it
once they are accounted for. The standalone `verdict` takes the lock the same way for C1's replays;
`--audit` judges without launching anything and needs no lock. A manual holder (`lock acquire
<owner> <phase> [<mask>]`, `lock release <owner> <pid>`) does the same by hand.

## 6. Numbered measurement history (short, bounded, with artifacts)

Numbers are the old item numbers where source comments or README cite them; each entry is one
finding with its recipe, control, budget, size, estimate and artifact. Nothing here is a standing
prohibition. Entries 30-33 carry the numbers of the earlier numbered record; they were not
re-derived from their artifacts for this rewrite, and an entry whose report cannot be named cites
that record.

4. Migration control (2026-09-09): the prototype's release network converted to format 6 with zero
   clock rows gave identical raw scores on 1,024 stored positions and reproduced the prototype's
   fixed-node decisions on 100. Artifact: runs/nnue_migration/ (control.nnue, clock_validate.jsonl).
7. Clock rows and output buckets, bounded historical findings, each arm against a common opponent
   and not head-to-head: the data-only retrain ctrl_data scored .8375 (130/8/22) against
   nnue_migration/control and the retrain without clock rows ctrl_noclock .825 (121/22/17) against
   that same network, 80 pairs each at 100 ms with one player thread but under different search
   builds (clock_a.exe for the first, partial_root.exe for the second), so this is not a controlled
   isolation of the clock rows; the bucketed head ft_buckets scored .506 (69/24/67) against
   ctrl_data (80 pairs, partial_root.exe). The clock rows stay, the buckets were removed (one head).
   Artifacts: runs/nnue_gauntlets/ctrl_data_vs_control/accept.json,
   ctrl_noclock_vs_control/accept.json, ft_buckets_vs_ctrl_data/accept.json.
13. Search stages, frozen weights, one at a time at 64-160 games: partial-root selection 59 %; table
    and hot-loop efficiency, history, futility 48-52 % (below resolution); the bundle (efficiency,
    history-aware reductions, both futility rules) 57.4 % (54.5-60.2) over 1,000 games at 50 ms and
    55.7 % (51.8-59.5) over 500 at 100 ms against partial-root alone, retained; one-ply extensions
    and null move 47-48 %, not retained; stalemate-first quiescence +14 % nodes/s, 53.6 %
    (50.8-56.4) over 500 games. Artifacts: runs/nnue_gauntlets/astra_partial_root, astra_efficiency,
    astra_history, astra_history_lmr, astra_quiet_futility, astra_reverse_futility,
    astra_extensions, astra_null_move, astra_exact_tactics.
16. Resolution: in the measured samples, 160 games gave paired intervals about +-8 % wide, so search
    changes worth 1-3 % were bundled and confirmed over 1,000 games; from 2026-09-12 C1's sequential
    test screens single search patches, and a fixed-count confirmation of the accepted bundle at 100
    ms stays in the plan.
17. Against the SQ teacher weights@66 (weights/sq_g128.onnx, f720705d...0f58071) at the site budget
    (250 ms per move, sq searching every move, audited): ft_self7 .580 [.485, .670] (51/14/35) over
    50 pairs, and ft_self11 .650 [.570, .730] (59/12/29) over 50 pairs with four player threads; the
    2026-09-09 attempt was invalid (the sq seat stopped searching under the deadline). The old
    reports' bootstrap intervals are margin intervals; the score endpoints are (margin + 1)/2.
    Artifacts: runs/nnue_final/ft_self7_vs_sq_250ms_fixed.json and .games.jsonl;
    runs/nnue_turn9/teacher.report, teacher.command and teacher.audit .json with
    teacher.games.jsonl; runs/nnue_turn9/previous_fixed_audit.json (the per-seat SQ-health audit).
30. Self-rescoring loop (the incumbent's own search at 40 x 2,500 nodes labels its games): a
    100k-row round 59.7 % at node budget and 57.2 % at 50 ms over 500 games against its parent; the
    quiet-best filter 50.7 % (46.8-54.5), 160-simulation labels 48.8 % (44.9-52.5), 24 epochs 50.8
    %, a 50 % self share 44.8 %, no windows 44.7 %, 195k rows 55.0 % against 100k rows 55.1 %;
    rounds 14-16 from ft_self11 48.6-49.0 % (no resolved gain under that protocol); two random
    opening moves in half the families 53.3 % (49.6-57.1); the outcome mix 52.4 % and 48.8 % (null);
    each comparison's opponent is in its directory name or the record. Evidence: the historical
    numbered record (item 30 of the previous DESIGN) and runs/nnue_gauntlets/ft_self5_500 to
    ft_self18_1000 (one report and one games file per round), quietbest_vs_shallow_500,
    ft_lambda15_vs_self15, ft_lambda16_500_timed, ft_lr2e4_500, ft_lr5e5_500, ft_nowindows_500,
    ft_latewindows_500, ft_self5_share50_500, ft_self5_long_500, ft_random15_500_timed.
31. Relabelling existing sets with the stronger conv@150 teacher, three arms each against a common
    opponent (ft_gpu_labels or ctrl_data, as the directory names say), not each other: the
    continuation on conv@150 human labels 63.8 % (61.3-66.3) over 1,000 games at 50 ms against its
    parent, the control on conv@40 labels 65.4 % (62.9-67.9), the student rounds relabelled 61.6 %
    (59.0-64.2): no resolved gain from relabelling under that protocol; new windows stay in the
    mixture. Artifacts: runs/nnue_gauntlets/ft_round2_vs_ft_gpu_labels,
    ft_round3_vs_ft_gpu_labels_sims8, ft_round3_vs_ft_gpu_labels_timed, ft_gpu_all_vs_ctrl_data,
    ft_gpu_labels_vs_ctrl_data, ft_human_vs_ctrl_data, nnue_ft_all_vs_ft_gpu_labels_timed.
32. GPU passes: 20 epochs of 1,000 steps at batch 8,192 on the broad mixture took the plateaued
    ft_self11 to 63.8 % over 1,000 games at 50 ms and 62.1 % (58.6-65.6) at 100 ms (deployed as
    ft_gpu150a); a second continuation chained on it 52.5 % (49.9-55.0). Artifacts:
    runs/nnue_gauntlets/ft_gpu150a_50ms, ft_gpu150b_50ms, ft_gpu150d_50ms, ft_gpu40c_50ms,
    ft_gpu150a_sims32. Whether 60 epochs beat 20 was A1, judged in item 39.
33. Width 768 padded from 512 with identical zero columns scored 43.7 % (39.7-47.5) over 500 games
    at 50 ms against the 512 it started from; the padding defect (identical columns, identical
    gradients) is fixed by seeded columns and the width question is open (A3). Artifact:
    runs/nnue_gauntlets/ft_self5_w768_500_timed.
34. Search speed and threads (Astra, turns 15-16): a 35.24 % median paired speed-up on one thread
    (each benchmark pass over the same 100 fixed-node roots; the ratio of the median aggregate
    throughputs is 29.7 %, 1,062,227 to 1,377,801 nodes/s), all 100 results identical; the stages
    (znver4, staged move generation, deferred accumulators, prefetch, the AVX-512 readout, cheaper
    attack status) were each measured against their recorded comparator, several against the same
    baseline, and their gains do not add to the total. With frozen weights 58.8 % (56.3-61.3) at 50
    ms over 1,000 games and 57.5 % (54.0-60.9) at 100 ms over 500; four threads against one at 100
    ms .670 (.640-.700) over 250 pairs; lazy evaluation with a margin and a shorter quiescence
    horizon failed their gauntlets, as did the adaptive stop (.498 [.472, .5245] over 500 pairs at
    50 ms; runs/nnue_turn14/time50.command, report and audit .json), whereas the seat-owned
    time-budget fix was accepted (.562 [.531, .595] over 250 pairs at 100 ms;
    runs/nnue_turn16/budget100.command, report and audit .json) beside the retained 72 %
    iteration-admission threshold. C1's calibration: .043 is the largest wrong-boundary rate
    observed among the calibrated endpoint cases, not a bound proved by the 130k trials. Artifacts:
    runs/nnue_turn15/final_comparison.json, provenance.json, gauntlet50.audit.json and
    gauntlet100.audit.json with their command and report files;
    runs/nnue_turn16/scaling4_100.audit.json and the lazy50, lazy100, q2_50 and q2_100 command,
    audit and report files, accepted_parity.json and provenance.json binding the retained code;
    runs/nnue_turn20/c1_calibration/experiment.json and verdict.json.
35. Race rows (format 7): 390 ns per query; a 28-30 % throughput loss with eager evaluation, about
    91 % of the baseline throughput deferred; ft_race16 52.5 % (48.6-56.3) at 8 simulations and 42.7
    % (38.8-46.6) at 50 ms against ft_self11; on 60k self-labelled rows the error grouped by race
    bucket is at most 0.046 and a per-bucket correction removes 0.8 % of the residual variance.
    Removed with format 8; a race term returns only through the C3 proof. Artifacts:
    runs/nnue_gauntlets/ft_race16_500_movems50, ft_race16_500_sims8.
36. Measured null or negative, bounded, not standing prohibitions: clock-row removal, output
    buckets, the quiet-best filter, human outcome labels alone, doubling the windows alone, null
    move, one-ply extensions, exact tactics beyond the current rule, relabelling with a stronger
    teacher (31), a second chained continuation (32).
37. Controlled gen3 data comparison (one seed): 53.3 % at 50 ms and 50.3 % at 100 ms against the
    control; no resolved gain under that protocol, one seed only. Artifact:
    runs/nnue_gauntlets/gen3_controlled/.
38. Format 8: the contract (runs/nnue_plan/format8_contract.md), gates 1-5 passed (parity on 512
    positions and 2,996 trajectory states under both H32 nets, ids and accumulators exact and the
    f64 value within 1e-12, scalar/AVX2/VNNI bit-identical, mb_b and mb_b8 fixed-node signatures
    identical on 64 roots); the factorised step costs 13 % more than format 6 on the CPU trainer;
    over 2 M gen3 rows the full context holds 40.6 % of the perspectives and seven contexts fall
    under 0.5 %. Gate 6 (throughput and RSS from outside, at least 95 % of format 6 in both modes)
    is preregistered in runs/nnue_gauntlets/format8/experiment.json and pending.
39. A1, training duration (2026-09-13; runs/nnue_gauntlets/long_training, verdict.json a01c89a2...,
    the runner's original kept as verdict.runner.json): from mb_b on the new70 mixture, two seeds.
    Epoch 60 against epoch 20 of the same stretched schedule .541 [.500, .581] and .443 [.395, .493]
    (seed 2 at 50 and 100 ms), .503 [.464, .540] and .522 [.480, .563] (seed 3); the stretched
    60-epoch run against the conventional 20-epoch run .494 [.454, .535] and .517 [.472, .560] (seed
    2), .510 [.470, .550] and .542 [.497, .585] (seed 3): the four duration rules (paired lower
    bound above .5 at both budgets) not met. Both 60-epoch arms against mb_b .568 [.529, .608] and
    .552 [.510, .593] (seed 2), .554 [.511, .596] and .568 [.527, .612] (seed 3): the two promotion
    rules met; the conventional 20-epoch seed-3 run against mb_b .564 [.524, .605] at 50 ms and .525
    [.482, .568] at 100 ms (explanatory). 200 pairs at 50 ms and 150 at 100 ms per match, 5,600
    games, all valid under the audit and Astra's independent audit. Failure of the superiority rules
    is not evidence of equivalence, and the comparisons do not isolate the mixture as the cause of
    the gains over mb_b. Provenance: the endpoint and match bindings were written after the fact
    (the checkpoints retain the data shares and the init name, not dataset or init hashes).
40. C2, the quiescence transposition table (2026-09-13; runs/nnue_gauntlets/c2_qtt; f528f22 against
    a46a1ed on the shared mb_b at 50 ms, one player thread, C1's sequential test): rejected at 192
    pairs, LLR -3.107 against the bound -2.944, pentanomial counts [35, 43, 66, 32, 16], 106/123/155
    over 384 games, descriptive score interval about [.394, .478]; audited by the runner and
    replayed by C1 itself; the mechanism was removed from master before the gate 6 freeze (8b60c32).
    Timing: .351 pairs per active second with eight concurrent pairs, idle share .236 over twelve
    complete batches.

## 7. Open hypotheses (the agreed programme, runs/nnue_plan/step_plan_final.md)

The state of each is written next to it: proposed, implemented, correctness-verified or
strength-accepted for code; preregistered, running or judged for an experiment.

- A1, duration: 60 epochs against 20 from the same init on the same mixture, two seeds and a
  20-epoch control; runs/nnue_gauntlets/long_training, judged (item 39): no demonstrated benefit from extending
  the stretched schedule from epoch 20 to 60, and the recipe rule not met for either seed; both
  60-epoch arms qualify for the mb_a gate, runs/nnue_gauntlets/a1_promotion (score at least .575 at
  100 ms over 250 pairs per arm under the frozen C1 build; preregistered, running; deployment is the
  user's decision).
- C2, search patches by sequential test: the quiescence transposition table candidate (f528f22) was
  rejected (item 40); the engine uses the baseline quiescence without a transposition table. Further
  search patches are screened the same way, one at a time under C1, and an accepted bundle is
  confirmed at 100 ms over a fixed count.
- D, labels: selected roots (the largest disagreement between the source label and a shallow one;
  the deeper pilot labels are obtained afterwards) against a seeded random sample, the same
  requested roots and node budget per arm at 1 M nodes with the actual cost recorded;
  runs/nnue_gauntlets/d_pilot preregistered, unrun.
- B, format 8: the Rust side correctness-verified; gate 6 (throughput and RSS from outside, at least
  95 % of format 6 in both modes) preregistered, unrun; then the training arm against the format 6
  control on identical data, schedule and seed, a seed-2 pilot first and, only on its result, a
  separately preregistered seed-3 confirmation.
- A3, H1024 with the corrected widening (distinct small seeded new columns, zero outgoing columns,
  integer parity at the start, channel differentiation verified), after A1; A2, the high-lr restart, open now that A1 is judged; C3, a race term only through a measured race error that stays with deeper
  labels; A4, from scratch at H1024 over 200+ epochs, on the GPU only and only after shorter matched
  controls justify it. All proposed.
- Exit condition, met by nothing above: at least 65 % against frozen mb_a at 100 ms over 500 games,
  confirmed at 250 ms with four threads at 60 % or more over 500 games (two concurrent games), plus
  a confirmation at the seat's 200k fixed nodes, paired intervals reported at each budget;
  deployment is the user's decision.
