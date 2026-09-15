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
the experiment manifests and verdicts own recipes and results;
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
for fixed-count matches, `sprt_accept` and `sprt_reject` for sequential ones; the latter, with the
incumbent in the candidate seat, is the non-regression gate: the newcomer is not two points worse) whether it
is complete and met; a
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
    runs/nnue_gauntlets/ft_self5_w768_500_timed. Measured with the preflight checker
    (`nnue.widen_check`, 2026-09-13/14): with zero outgoing columns the new channels are served as
    zero under quantisation-aware training and receive no task gradient, from QAT at epoch 0
    (a3_width, 256 updates) and after one float epoch of 1,000 updates at lr up to 1e-4
    (a3_width_warmin: no new readout reached the 1/64 grid); seeding the new readout and dense
    columns with +-1/64 deviates from the parent by up to .37 raw on the 512 fixture positions (the
    dense path amplifies the seeds; a3_width_seeded); seeding the readout columns only
    (`--widen_outgoing 0.015625`, dense columns zero) deviates by at most .087 raw (mean .023) and
    every new channel carries a task gradient at the first quantised step (a3_width_readout
    preflight 494c97b1...), the widening under test in section 7. Built by Claude 2026-09-14 on
    the user's instruction; Astra's co-signature pending.
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
    under 0.5 %. Gate 6 (2026-09-13; runs/nnue_gauntlets/format8/gate6/attempt2, verdict
    23d49ee6...; the frozen post-C2 build e4e0d5ff..., mb_b against its conversion mb_b8, 100 fixed
    roots, one thread, five cycles, measured from outside): format 8 over format 6 throughput .9903
    at fixed nodes (1,431,545 against 1,445,625 nodes/s) and .9635 at 50 ms per move (1,453,672
    against 1,508,722), both at least the preregistered .95; loaded RSS 70.3 against 57.9 MiB; all
    fixed-node signatures equal. A first attempt (attempt1_invalid) was invalidated by its own input
    check when the runner module changed under it. A throughput result, not a strength result.
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
41. D, deep labels on selected roots (2026-09-13; runs/nnue_gauntlets/d_pilot; pinned manifest
    5cfbbeef..., verdict.json 8efd8548..., the runner's original kept as verdict.runner.json): from
    mb_b on the A1 mixture plus 5 % of 20,000 fresh 1 M-node labels (the frozen build e4e0d5ff... on
    mb_b), the 20,000 roots of a 200,000-root pool whose 2,500-node label differed most from the
    source label (gap mean .197 against the pool's .039) against a seeded random 20,000 of the same
    pool (overlap 1,988 positions); 20 epochs, seed 2. Selected against random .470 [.428, .513] at
    50 ms and .508 [.463, .553] at 100 ms: the selection rule not met. Against mb_b: selected .574
    [.533, .615] and .540 [.497, .583], random .534 [.494, .574] and .535 [.492, .578]: neither
    promotion rule met. 200 pairs at 50 ms and 150 at 100 ms per match, 2,100 games, all valid.
    Cost: 39.0 G nodes and 11.8 process-hours for the two deep arms, 0.5 G nodes for the screen.
    Failure of the rules is not evidence that selection has no value at a larger share or count, and
    no arm isolates whether 1 M-node labels beat the 100k-node source labels.
42. B, format 8 against format 6 (2026-09-13; runs/nnue_gauntlets/format8_train; pinned manifest
    ae185de3..., verdict.json 3c9709df..., the runner's original kept as verdict.runner.json): from
    mb_b's format 6 bytes, the format 8 arm factorised through widen_contexts (residuals zero)
    against the format 6 continuation on the A1 mixture, 20 epochs, seed 2, CPU. Format 8 against
    format 6 .583 [.544, .621] at 50 ms and .580 [.538, .622] at 100 ms: the format rule met.
    Against mb_b: format 8 .586 [.545, .628] and .547 [.503, .588], format 6 .578 [.538, .616] and
    .545 [.502, .590]: both promotion rules met. 200 pairs at 50 ms and 150 at 100 ms per match,
    2,100 games, all valid. At the retained epoch-20 endpoints, objective on the same validation
    rows .0197 against .0209; epoch-mean training value loss .0104 against .0154. Mean wall time
    over all 20 epochs 326 against 293 seconds per epoch (the last epoch 329 against 290). One seed;
    the seed-3 confirmation, preregistered before this result, must independently meet format_gain
    and promotion_format8 at both budgets to confirm training-format adoption.
43. C2, capture history (2026-09-13; runs/nnue_gauntlets/c2_capture_history; pinned manifest
    e5f72d57..., verdict.json 36ef7a24..., the runner's original kept as verdict.runner.json):
    dd15c18 (a worker-local table by absolute side, attacker type, canonical destination and victim
    type, scoring captures in place of the quiet table they shared by action index; updated at
    completed main-search cutoffs with the quiet table's gravity, bonus and halving), frozen as
    494b4b6 = d10f448 plus the patch (runs/nnue_bins/c2_search_capture_history.exe d00ab30f...)
    against the gate 6 build e4e0d5ff... on the shared mb_b at 50 ms, one player thread, C1's
    sequential test: accepted at 1,760 pairs, LLR 3.050 against the bound 2.944, pentanomial counts
    [170, 324, 715, 319, 232], 1368/903/1249 over 3,520 games, descriptive score interval [.504,
    .530]; reconciled by the runner, whose verdict included C1's journal replay, and independently
    audited without replay by Claude and Astra; the first patch of the bundle, merged into master
    2026-09-14 (95ae9ee). Timing: .352 pairs per active second with eight concurrent pairs, idle share
    .284.

44. B, format 8 against format 6, seed 3 (2026-09-14; runs/nnue_gauntlets/format8_train_s3; pinned
    manifest bd34ff3f..., verdict.json 565d51e7..., the runner's original kept as verdict.runner.json):
    the confirmation of item 42 preregistered before its result, from mb_b's format 6 bytes on the A1
    mixture, 20 epochs, seed 3, CPU (the format 8 arm interrupted at epoch 19 on 2026-09-13 and resumed
    exactly). Format 8 against format 6 .556 [.518, .596] at 50 ms and .552 [.510, .593] at 100 ms:
    the format rule met. Against mb_b: format 8 .629 [.593, .666] and .623 [.582, .665], format 6
    .566 [.528, .605] and .543 [.495, .592]: promotion_format8 met, promotion_control6 not met (the
    100 ms lower bound .495). 200 pairs at 50 ms and 150 at 100 ms per match, 2,100 games, all valid.
    Objective at epoch 20 .0196 against .0210; epoch-mean training value loss .0104 against .0153.
    Two seeds meet format_gain and promotion_format8 independently: format 8 is adopted as the
    training format. Audited by Claude alone (Astra unavailable under a Codex usage limit until
    2026-09-19); co-signature pending.

45. B2, format 8 on the qualified lineage (2026-09-14; runs/nnue_gauntlets/b2_format8_lineage; pinned
    manifest 9304f4f9..., verdict.json 5e6272f7..., the runner's original kept as verdict.runner.json):
    from long60_s2:epoch60's format 6 bytes, the format 8 arm factorised through widen_contexts against
    the format 6 continuation on the A1 mixture, 20 epochs, seed 2, both arms on the GPU (30 and 38 s per
    epoch). Format 8 against format 6 .564 [.522, .606] at 50 ms and .540 [.495, .583] at 100 ms: the
    format rule not met (the 100 ms lower bound .495). Against the parent: format 8 .564 [.521, .605] and
    .595 [.550, .640] (parent_gain_format8 met), format 6 .526 [.485, .568] and .528 [.483, .573]
    (explanatory). 200 pairs at 50 ms and 150 at 100 ms per match, 2,100 games, all valid. Objective at
    epoch 20 .0192 against .0202; epoch-mean training value loss .0098 against .0141. The format 8
    continuation is the strongest network measured against the qualified parent; whether the format or
    the continuation carries the 100 ms gain is left to the seed-3 confirmation (b2_format8_lineage_s3,
    its activation amended before its matches). The width-parent selection is not authorised by this
    verdict; a3_width_readout, trained on that parent under the user's instruction, is judged as its
    own matched comparison. Audited by Claude alone; co-signature pending.

46. Exit check of B2's network against mb_a (2026-09-14; runs/nnue_gauntlets/exit_check_b2; pinned
    manifest 646e7f43..., verdict.json 01c9050a..., the runner's original kept as verdict.runner.json):
    b2_format8_s2:epoch20 (ba8e0b93...) against the deployed mb_a under the frozen gate 6 build on both
    seats, one thread, 100 ms, 250 pairs: 247 wins, 136 draws, 117 losses, .630 [.598, .662], 500 games
    valid, 250 distinct openings, first mover .514. arena_candidate (.575) met; exit_100ms (.65) not met.
    Against the same reference long60_s2:epoch60 scored .599 [.566, .632] (the promotion test of item
    39's lineage): the format 8 continuation is 3.1 points stronger at 100 ms, the interval's lower bound
    at the old point estimate; the network alone does not reach the exit clause. The deployment-candidate
    test deploy_candidate_b2_ch (this network under the accepted capture-history build against mb_a under
    the gate 6 build, per-side networks in the runner) follows; deployment is the user's decision (a
    format 8 network needs a format 8 capable arena binary). Audited by Claude alone; co-signature pending.

47. Deployment candidate: the accepted search build with B2's network against the deployed pair
    (2026-09-14; runs/nnue_gauntlets/deploy_candidate_b2_ch; pinned manifest 19e979aa..., verdict.json
    214d3413..., the runner's original kept as verdict.runner.json): c2_search_capture_history.exe
    (d00ab30f..., item 43) playing b2_format8_s2:epoch20 (ba8e0b93...) against the frozen gate 6 build
    (e4e0d5ff...) playing mb_a (2e275215...), rpsi seats on both sides with one thread each (per-side
    networks in the runner, 4e04d59 and ee4e32b), 100 ms, 250 pairs: 273 wins, 115 draws, 112 losses,
    .661 [.625, .695], 500 games valid, mean 251 plies. arena_candidate (.575) and exit_100ms (.65)
    both met: the first clause of the exit condition (section 7) is met for the first time, by the
    pair; the network alone scored .630 (item 46) and the patch's sequential estimate was .517 at
    50 ms (item 43). The remaining clauses (250 ms with four threads at .60 over 500 games; the seat's
    200k fixed nodes) are preregistered next; deployment is the user's decision. Audited by Claude
    alone; co-signature pending.

48. A3, H1024 by readout-seeded widening against the H512 continuation (2026-09-14;
    runs/nnue_gauntlets/a3_width_readout; pinned manifest ea3da948..., verdict.json 0e4fce77..., the
    runner's original kept as verdict.runner.json): both arms from b2_format8_s2:epoch20 on the A1
    mixture, 20 epochs at lr 1e-4, one float epoch then QAT, seed 2, on the GPU, the wide arm widened
    with `--widen_outgoing 0.015625` (preflight passed, item 33). H1024 against H512 .408 [.369, .446]
    at 50 ms and .423 [.380, .465] at 100 ms: width_gain not met, the wide network clearly weaker on
    the clock. Against the parent: H1024 .429 [.391, .468] and .410 [.367, .453] (promotion_h1024 not
    met); the H512 continuation .528 [.491, .564] and .472 [.430, .513] (promotion_control512 not met).
    200 pairs at 50 ms and 150 at 100 ms per match, 2,100 games, all valid. Objective at epoch 20:
    H1024 .0194, H512 .0194, the parent .0192; training value loss .0080 and .0085 against the parent's
    .0098. Reading: the lineage has exhausted its mixture; twenty more epochs at either width lower the
    training loss and raise the held-out objective, and the wider network's slower evaluation is not
    repaid. Width is closed at this data size; exit_check_a3 is not run. Audited by Claude alone;
    co-signature pending.

49. Exit condition, second clause: the deployment pair at 250 ms with four threads (2026-09-14;
    runs/nnue_gauntlets/exit_250ms_b2_ch; pinned manifest 0c3e0eed..., verdict.json 8d1f5a0e..., the
    runner's original 1cf1e1d0... kept as verdict.runner.json): c2_search_capture_history.exe
    (d00ab30f...) playing b2_format8_s2:epoch20 (ba8e0b93...) against the frozen gate 6 build
    (e4e0d5ff...) playing mb_a (2e275215...), rpsi seats with four player threads each, two concurrent
    games, 250 ms, 250 pairs: 227 wins, 183 draws, 90 losses, .637 [.607, .667], 500 games valid, 250
    distinct openings, first mover .533, mean 288 plies (Goal 316, CaptureClock 183, Elimination 1).
    exit_250ms_4t (.60) met: the second clause of the exit condition (section 7) is met by the same pair
    as the first; the third (the seat's 200k fixed nodes, exit_200k_b2_ch) follows in the queue. The
    draw share rises from 23 % at 100 ms to 37 % here and the games run longer. Audited mechanically
    by the loop (runs/nnue_plan/autoloop.py audit: runner and re-judgement agree); Claude's reading;
    co-signature pending.

50. Exit condition, third clause: the deployment pair at the seat's 200k fixed nodes (2026-09-14;
    runs/nnue_gauntlets/exit_200k_b2_ch; pinned manifest ba282170..., verdict.json 91e7303c..., the
    runner's original 0b33e986... kept as verdict.runner.json): c2_search_capture_history.exe
    (d00ab30f...) playing b2_format8_s2:epoch20 (ba8e0b93...) against the frozen gate 6 build
    (e4e0d5ff...) playing mb_a (2e275215...), rpsi seats with one player thread each, eight concurrent
    games, 80 simulations (200k nodes) per move, 250 pairs: 234 wins, 145 draws, 121 losses, .613
    [.580, .647], 500 games valid, 250 distinct openings, first mover .563, mean 259 plies (Goal 353,
    CaptureClock 145, Elimination 2). exit_80sims_1t (.60) met: the third and last clause of the exit
    condition (section 7) is met by the same pair as the first two. The draw share is 29 % here, between
    the 23 % at 100 ms and the 37 % at 250 ms with four threads. Judged 21:53 by the queue's runner and
    audited mechanically by the loop (runner and re-judgement agree); Claude's reading; co-signature
    pending.

51. B2's seed-3 confirmation: format 8 factorised from long60_s2:epoch60 against its format 6
    continuation (2026-09-14; runs/nnue_gauntlets/b2_format8_lineage_s3; pinned manifest 54230500...,
    verdict.json 34926105..., the runner's original 33c62d2e... kept as verdict.runner.json): both
    arms from long60_s2:epoch60 on the lineage's 12-set mixture, 20 epochs at lr 1e-4,
    quantisation-aware from the first epoch, seed 3, on the GPU (b2_format8_s3:epoch20 04d05114...,
    b2_control6_s3:epoch20 fd052a18...); 200 pairs at 50 ms and 150 at 100 ms per match, one thread,
    eight concurrent games, the parent matches beside, 2,100 games, all valid. Format 8 against format
    6: .546 [.507, .586] at 50 ms and .527 [.482, .573] at 100 ms, so format_gain (lower bound above
    .5 at both budgets) is not met at 100 ms, as in seed 2 (item 45, lower bound .495). Against the
    parent: format 8 .569 [.527, .609] and .567 [.522, .610], parent_gain_format8 met at both budgets;
    format 6 .511 [.472, .550] and .523 [.482, .565]. Reading: the second seed repeats the first. The
    format 8 continuation beats its parent by five to seven points at both budgets and the format 6
    continuation does not, while the format-against-format difference (2.7 to 4.6 points) is smaller
    because the control also learns from its twenty epochs; the adoption of format 8 (item 44) stands
    and the exit clauses were met by a format 8 network (items 47, 49, 50). Judged 23:08 by the
    queue's runner and audited mechanically by the loop (runner and re-judgement agree); Claude's
    reading; co-signature pending.

52. C2, late move pruning by sequential test (2026-09-14; runs/nnue_gauntlets/c2_lmp; pinned manifest
    53af4c88..., verdict.json 93a8ef5a..., the runner's original 4709d4ec... kept as
    verdict.runner.json): runs/nnue_bins/c2_search_lmp.exe (3b4f6032..., revision 158a661 = the
    capture-history base 494b4b6 + 9a1e452) against the accepted capture-history build
    c2_search_capture_history.exe (d00ab30f...), rpsi seats with one thread each, eight concurrent
    games, both playing mb_b (a3180912...), 50 ms, the C1 sequential test (.50 against .52): rejected
    at 128 pairs with 66 wins, 41 draws, 149 losses, .338 [.289, .389], LLR -3.96, pentanomial counts
    [41, 25, 44, 12, 6], 256 games valid, 128 distinct openings, mean 242 plies. accept_lmp not met,
    and not as a null: the patch is sixteen points weaker than its base at 50 ms, so the moves it
    prunes late in the list are often the best ones under this move ordering. The bundle stays at
    capture history; internal iterative reduction is next on that base, and a re-tuned or
    history-guarded pruning would be a new frozen build with its own preregistration. The improvement
    loop's search build is unchanged. Judged 23:14 by the queue's runner and audited mechanically by
    the loop (agree); Claude's reading; co-signature pending.

53. A4, from scratch at H1024 over 200 epochs (2026-09-14 and 15;
    runs/nnue_gauntlets/a4_scratch_h1024; pinned manifest 3e1f6e28..., verdict.json 51e56cd1..., the
    runner's original 4bee0d8c... kept as verdict.runner.json): one arm a4_h1024_scratch_s2 without an
    init on the lineage's 12-set mixture, 200 epochs of 1,000 steps at batch 8192, lr 3e-4 with 1,000
    warm-up steps and a floor of .15, twenty float epochs then quantisation-aware, seed 2, trained on
    the GPU 2026-09-14 under the user's grant; endpoint epoch200 (19212038...) played, the best
    validation checkpoint at epoch 101 (objective .0255, c1ca294f...) recorded beside it. Against
    long60_s2:epoch60: .251 [.216, .286] at 50 ms (200 pairs) and .302 [.255, .350] at 100 ms (150
    pairs); against mb_a .339 [.304, .374] at 100 ms (250 pairs); 1,200 games valid, one thread, eight
    concurrent games, mean 214 to 228 plies. parent_gain and arena_bar not met, by twenty to
    twenty-five points. Reading: two hundred passes from scratch over the mixture the lineage's
    continuations train on for twenty leave a wider network far behind the parent, and the validation
    objective bottomed at epoch 101 (.0255 against about .019 for the lineage's networks on this
    mixture, item 48), so the second hundred passes overfit. The corpus does not carry the lineage's
    strength; that strength was accumulated over its self-play generations, which is what the
    improvement loop continues. From scratch is closed at this corpus size: the loop's scratch arm
    stays off (its gate, a4_vs_long60_100 at .48 or better, reads .302) and LOOP_PLAN section E is not
    taken; revisit only when the pool is several generations deeper. Judged 2026-09-15 00:01 by the
    queue's runner and audited mechanically by the loop at 00:02 (agree); Claude's reading;
    co-signature pending.

## 7. Open hypotheses (the agreed programme, runs/nnue_plan/step_plan_final.md)

The state of each is written next to it: proposed, implemented, correctness-verified or
strength-accepted for code; preregistered, running or judged for an experiment.

- A1, duration: 60 epochs against 20 from the same init on the same mixture, two seeds and a
  20-epoch control; runs/nnue_gauntlets/long_training, judged (item 39): no demonstrated benefit
  from extending the stretched schedule from epoch 20 to 60, and the recipe rule not met for either
  seed; both 60-epoch arms qualify for the mb_a gate, runs/nnue_gauntlets/a1_promotion (score at
  least .575 at 100 ms over 250 pairs per arm under the frozen C1 build; preregistered, running;
  deployment is the user's decision).
- C2, search patches by sequential test: the quiescence transposition table rejected (item 40);
  capture history accepted (item 43), the first patch of the bundle, merged into master 2026-09-14 (95ae9ee);
  late move pruning and internal iterative reduction are screened next on that base, one at a time
  (LMP frozen 2026-09-14 by Claude as runs/nnue_bins/c2_search_lmp.exe 3b4f6032..., revision 158a661 =
  494b4b6 + 9a1e452; c2_lmp 53af4c88... judged 2026-09-14 23:14, item 52: rejected at 128 pairs, .338 at 50 ms,
  so the bundle stays at capture history and internal iterative reduction is next on that base); a fixed-count
  100 ms confirmation of the bundle after at most three accepted patches.
- D, labels: the pilot judged (item 41), the selection not supported at this size; a larger share or
  a label-depth arm would need its own preregistration.
- B, format 8: the seed-2 pilot met its rules (item 42); the seed-3 confirmation (format8_train_s3,
  pinned bd34ff3f...) was stopped 2026-09-13 with its format 8 arm at epoch 19 and resumed 2026-09-14
  09:45 and judged 12:48 (item 44): format_gain and promotion_format8 met at both budgets, so
  format 8 is adopted as the training format. B2
  (runs/nnue_gauntlets/b2_format8_lineage: format 8 factorised from long60_s2:epoch60 against its
  format 6 continuation) was pinned 2026-09-14 (9304f4f9..., both arms on the GPU under the user's grant,
  the seed-3 guard overridden by the user's instruction), its arms trained (10 and 13 minutes), its
  matches queued; its own seed-3 confirmation b2_format8_lineage_s3 (da64d98c...) and exit_check_b2
  (the network against mb_a at 100 ms, .575 and .65 rules) are preregistered. B2 judged 14:06 (item 45):
  parent_gain_format8 met, format_gain not met at 100 ms by the lower bound .495; exit_check_b2 running;
  the seed-3 matches queued (guard amended before they play). b2x_duration40 (d636972c..., the format 8
  continuation over 40 epochs against B2's 20-epoch endpoint) pinned 14:05, its arm trained on the GPU
  (1,218 s), matches queued. deploy_candidate_b2_ch (19e979aa..., the accepted capture-history build
  playing b2_format8_s2:epoch20 against the gate 6 build playing mb_a at 100 ms over 250 pairs, .575 and
  .65 rules; per-side networks in the runner, 4e04d59) pinned 14:18 and queued after exit_check_b2.
  exit_check_b2 judged 14:33 (item 46): .630 [.598, .662] against mb_a, the arena bar met, the exit clause
  not; deploy_candidate_b2_ch judged 15:00 (item 47): .661 [.625, .695], both rules met.
  b2_format8_lineage_s3 judged 23:08 (item 51): parent_gain_format8 met at both budgets, format_gain again
  not met at 100 ms (lower bound .482); format 8 stands. b2x_duration40's matches were dropped from the queue
  2026-09-14 22:55 on the user's word (the arm is trained; re-queue any time).
- A3, H1024 with the corrected widening: the manifest runs/nnue_gauntlets/a3_width (format 6 from
  long60_s2:epoch60) failed its preflight 2026-09-13
  (runs/nnue_gauntlets/a3_width/preflight/report.json 75068e4b...): structure and integer parity
  exact, but under quantisation-aware training the zero readout columns are served as zero and no
  new channel received a task gradient in 256 updates. The agreed remedy (one float epoch before QAT, checker e059935, merged
  2026-09-14) failed its preflight too (a3_width_warmin: no new readout reached the 1/64 grid in
  1,000 float updates); readout-seeded widening passed (item 33): a3_width_readout (format 8 H1024
  against H512 from B2's format 8 parent, both with `--widen_outgoing 0.015625` and one float epoch,
  pinned ea3da948..., arms trained on the GPU 2026-09-14, judged 16:15 (item 48): width_gain,
  promotion_h1024 and promotion_control512 all not met; exit_check_a3, preregistered, is not run).
  A2, the high-lr restart: a2_restart_lr3e4 (ba1fb0e3...) pinned 14:20, one arm from b2_format8_s2:epoch20
  at lr 3e-4 against A3's H512 continuation arm at 1e-4 as the matched control (same init, mixture, seed,
  QAT start and length), judged at 50 and 100 ms plus the parent at 100 ms; the arm trained on the GPU, its
  matches dropped from the queue 2026-09-14 22:55 on the user's word (re-queue any time). C3, a race term only through a measured race error that stays with
  deeper labels; A4, from scratch at H1024 over 200 epochs: a4_scratch_h1024 pinned 3e1f6e28..., trained on the GPU
  2026-09-14 (the user's grant; the GPU's long job while the CPU played the other matches), judged 2026-09-15
  00:01 (item 53): .251 and .302 against long60_s2:epoch60 at 50 and 100 ms, .339 against mb_a at 100 ms,
  neither rule met; from scratch is closed at this corpus size and the loop's scratch arm stays off. C3
  proposed.
- Arena: nnue_2 = long60_s2:epoch60 (ed00ddf9..., the network that met the arena bar in the
  promotion test of item 39's lineage) deployed 2026-09-13 19:53 on the user's decision as a second
  NNUE seat at a fixed 200k nodes per move beside nnue_1 = mb_a; the accepted capture-history patch
  is not in the shared arena binary. nnue_3 = the deployment candidate of item 47 (b2_format8_s2:epoch20
  under the capture-history search, a seat-local Linux build of master e7251b5, c2efc2e4...) deployed
  2026-09-14 17:31 on the user's decision as a third seat at the same fixed 200k nodes, one thread;
  nnue_1 and nnue_2 unchanged as references. The Henhen site bot Vlad_NNUE runs the same pair and
  binary since 2026-09-14 23:46 UTC (six threads; README). Its per-move cap was raised from 1000 to
  3000 ms on 2026-09-15 01:00 UTC on the user's decision: under the site's 60 s + 1 s clock the seat's
  budget formula (match/src/rpsi.rs `move_budget_ms`: 0.8 x increment + clock/30 - 150 ms, floor
  `--move-ms` 1000) targets about 2.6 s early in a game, and the old cap made every move exactly
  1000 ms and never spent the bank; the floor is unchanged, so the seat spends more only while its
  clock allows it. Unmeasured on the clock (no clock model in `bot eval`); the site ladder is the
  reading.
- Exit condition, its first clause met by the deployment candidate of item 47 (the capture-history
  build with b2_format8_s2:epoch20, .661 at 100 ms) and its second by the same pair (item 49:
  exit_250ms_b2_ch 0c3e0eed..., .637 [.607, .667] at 250 ms with four threads, judged 20:55) and the
  third by the same pair (item 50: exit_200k_b2_ch ba282170..., .613 [.580, .647] at the seat's 200k
  fixed nodes under the runner's fixed-simulation match mode, judged 21:53): all three clauses are met.
  The clauses were: at least
  65 % against frozen mb_a at 100 ms over 500 games,
  confirmed at 250 ms with four threads at 60 % or more over 500 games (two concurrent games), plus
  a confirmation at the seat's 200k fixed nodes, paired intervals reported at each budget;
  deployment is the user's decision.
- The improvement loop (runs/nnue_plan/autoloop.py, rewritten 2026-09-14 after the review of its
  skeleton and redesigned 2026-09-15 as a fixed-node ladder; runs/nnue_loop/ holds its state, log,
  digest, proposals and alerts; runs/nnue_plan/LOOP_PLAN_2026-09-14.md the reasoning): self-play from
  the accepted network on both CCDs at idle priority; when the pool's untested rows reach the
  increment (2.7 M rows, one 12,800-game batch, doubled after null rounds up to 4x) four mixture continuations of the accepted
  network, a control every third round, an optional from-scratch arm and the mean-weight soup of the
  continuations are trained on the GPU; then, under the lock, at fixed nodes with 16 games at once
  and both seats through the accepted search build: a 200-opening screen of every variant at 20k
  nodes (the held-out MSE recorded beside it, never decisive; the best at least .49 or the round is
  null), the C1 sequential test at 40k nodes (gain), the same test at 240k nodes with the accepted
  network in the candidate seat (hold; rule sprt_reject = not two points worse), both met promote;
  then a deployment ladder against the deployed pair (sequential accept at 240k, the 250 ms four-thread
  site-like test at .5, an anchor reading); both deploy rules met write a proposal. The two budgets
  are six apart as fishtest's STC and LTC. Since 2026-09-15 08:20 (the user's decisions of 08:00 and
  08:20) a promotion is deployed to the arena at once as the next nnue_N seat (runs/nnue_plan/deploy_arena.py:
  the seat-local capture-history build of nnue_3, one thread, 200k fixed nodes; refused after a search change
  until the seat binary is rebuilt for Linux; the site bot is not touched) and the arena's own rating confirms
  it: a settled rating 30 Elo above the previous strongest nnue seat = the deployed pair, a rating below the
  previous seat beyond both half-widths = an alert. The redesign's deployment ladder (deploy_ltc, deploy_site,
  anchor_ltc: 8-12 h of lane b per promotion at the effect sizes seen) is off behind DEPLOY_LADDER. Round 1
  (2026-09-15, on gen5's 2.72 M rows) promoted the soup of its four continuations: screen .530 [.491, .569],
  gain .527 [.507, .547] accepted in 736 pairs at 40k nodes, hold .513 [.488, .538] at 240k (the incumbent
  could not show +2 in 384 pairs); deployed as nnue_4 at 08:13, its ladder abandoned 33 min in. Started 2026-09-14 21:00, redesigned code
  from 2026-09-15 00:50. The queue was trimmed 22:55 on the user's word (gen5_round never preregistered:
  round 1 tests gen5's data directly against the accepted network); the loop took the lane 01:14 when the
  queue drained; a Windows update rebooted the machine 01:33 and the loop was restarted 04:20 with its
  generators resumed from their records.
  On the user's list of 2026-09-15 09:05 (claude_notes 09:25): the hold test is off (HOLD_LTC; a gain_stc
  accept promotes at once and the arena's regression alert is the guard: about 90 minutes of lane b saved
  per promoted round), the conv distillation sets conv_13_25_2m and conv_26_40_2m leave every mixture from
  round 3 (DROPPED_SETS, the other sets keeping their proportions; the lineage's own 100k-node labels are
  the teacher now), control rounds add a control seed replica and the mean of the two controls
  (control_soup, screened for attribution of the averaging), a from-scratch H1024 arm trains again once the
  lineage's fresh rows reach 40 M (WIDTH_RETEST_ROWS; reported only, item 53's question at a larger corpus),
  the trainer exports an exponential moving average of the weights on request (Config.ema, commit 4cccc20;
  the `ema` variant at .999 screened beside pool), and the generator records the first K principal-variation
  positions of every search label (`--pv-labels`, schema 2, commit f88ed35; the search unchanged, checked
  move for move against the accepted binary) for the importer's `--pv-rows` line rows (kind TEACHER, source 7,
  weight .5, roots first among duplicates): the pv_ab_01 A/B on identical games decides whether the loop's
  imports use them. Generation measured 2,050-2,240 games/h per full 16-thread lane at 100k nodes, about
  4,300 games/h and 0.95 M rows/h with both lanes.
  On the user's word of 2026-09-15 11:00 (claude_notes 11:05), without their A/Bs: the importer turns the
  three recorded line positions of every root into rows (PV_ROWS 3, weight .5) from the next pool import,
  and generation samples a near-best alternative from the next job (the c4 build runs/nnue_bins/c4_multipv.exe,
  commit 4bae8b8: `--multipv 2 --multipv-margin 30 --multipv-prob 50 --multipv-nodes 25`, schema 3; at
  multipv 1 its records equal c3's move for move; a dry run played the alternative at 28.5 % of eligible
  plies for +24 % nodes and +18 % search time per root). The pool is measured in searched-root rows
  (root_rows: the round trigger, the width-retest count and the mixture shares ignore line rows, so PV rows
  add positions per game without moving the cadence or the recipe's shares). pv_ab_01 and the multipv A/B
  are not run: the screen, the gain test and the arena judge the combined change round by round, and a null
  streak is the signal to revisit either. The same morning the label-budget test nodes_ab_01
  (make_nodes_ab_manifest.py) took both lanes: three generations at the same node total from the r01 soup
  (3,200 games at 100k nodes, 12,800 at 25k, 32,000 at 10k), each trained as a pool-recipe continuation with
  identical mixtures except the fresh set; the cheapest arm that beats the 100k arm under the sequential
  test at 40k becomes NODES for new jobs.
  Run as 25k against 50k (the user 11:20: 100k "safely too high"; the 10k batch stopped at 9,984 games, unused;
  the 25k batch capped at 6,400 games and the 50k batch 3,200, about 37 G nodes each; the loop meanwhile switched
  to 25k with multipv and PV labels at 11:12 and had both lanes back by 12:02). Verdict 13:42 (nodes_ab_01_train
  pinned 12:15, nodes_ab_01 pinned 12:37, both arms from the r01 soup under the pool recipe): 25k arm .504 [.464,
  .545] and 50k arm .515 [.475, .555] against the r01 soup at 20k over 400 games each; 25k against 50k at 40k
  .497 [.479, .515] over 1,920 games, the sequential test undecided at its cap. The preregistered tie-break named
  the reference arm's 50k; the user chose 25k (13:50) for the rows and the round cadence, the arena to decide
  whether the deeper label is needed: NODES stays 25k, ROUND_ROWS 6 M root rows. The arena the same hour: nnue_5
  settled at 252.9 +-30.8 after 616 games, delta -2.9 against nnue_4 (not confirmed, no regression), the second
  promotion in a row whose 40k-node gain test passed and whose arena reading sits under the +30 bar.
  The PV rows' own control (2026-09-15 15:30, the user): every pool or parent set with line rows gets a
  roots-only twin (`<set>_roots`: the same records imported without --pv-rows, so its rows are exactly the set's
  root rows and its share is the set's), and one arm per round, pool_roots, trains the pool mixture on the twins.
  It is screened and can take the gate like any arm; pool against pool_roots at the screen and on the held-out
  MSE is a running A/B of the PV rows on identical games (about 11 GPU minutes and 4 lane-b minutes per round,
  one 8-minute import per batch). The arm is skipped in a round whose twins are not all imported. Multipv
  sampling has no such control: it changes the games, so its test is a separate generation.
  The screen in stages with common openings (2026-09-15 16:00, the user, after Stockfish's easy_train.py, which
  tests every saved epoch at 25,000 nodes from one book and adds games only to the networks whose Elo plus 1.5
  error bars still reaches the leader): stages of 100, 100 and 200 openings; every net plays stage 1 from the
  same openings (one seed per stage, so the comparison between arms is paired on the openings; before, each arm
  drew its own 200); after each stage the wins, draws and losses are pooled and only the contenders go on, a
  contender being a candidate whose pooled score plus 1.5 standard errors reaches both the leader's pooled score
  and .49 (control and width arms play stage 1 only); the choice is by pooled score. The same lane-b budget as
  the flat 200-opening screen (about 4,000 games), spent on the contenders: the finalists end with 800 games
  instead of 400 and the losers with 200. First used by round 3's screen (loop_r03_screen, stage 1 pinned 16:02).
  Two guards added 17:15 after round 3's stage 2 re-admitted a net dropped at stage 1 (its 200-game error bar
  was wider): a net dropped at a stage stays dropped, and the last stage takes at most four contenders by pooled
  score (round 3's stage 3 had run eight of ten: 3,200 games where four would have been 1,600).
  Round 3's result (the first round on 25k labels, multipv games and PV rows): every arm above .49 at the screen;
  ema chosen at .5475 over 800 games (pool_s3 .545, human2x .536; the base pool arm .4925 over 400 against its
  seed replica's .545, seed noise again); the gain test accepted at .536 [.511, .563] over 432 pairs at 40k; ema
  promoted and deployed as nnue_6 at 17:46. A loop bug found at the promotion: finish_round zeroed the untested
  row count after every non-failed round, losing the sets imported while the round ran (loop_08 and loop_09,
  4.97 M root rows); fixed to recount the pool sets outside the round, the state corrected under a stop.
