# conv: convolution trunk with interleaved attention

Six pairs of a 3x3 convolution block and a full-board attention block over the
81 squares at width 160, 6,031,492 parameters against sq_g128's 4.16M. The
five 32-dimensional attention heads use fused scaled-dot-product attention;
opset-17 ONNX export automatically selects the equivalent explicit path. What
the rules can compute exactly is an input or an exact search value rather than
something the network learns: 46 observation planes carry obstacle-aware
reachability, material by type and the capture clock, and the search resolves
short forced wins and losses itself. The heads are a from-to policy, a
categorical from-to Q, a distributional state value, the win/draw/loss
outcome head, the regret ranking and value head of the search control, and
the training-only tactics, occupancy, plies-to-end, material and
opponent-reply heads. DESIGN.md is the specification; every numbered item
there is approved before it is built.

[RGSC.md](RGSC.md) contains the RGSC diagnosis, evidence limits and proposed
experiments. It distinguishes validated fixes from experimental training changes.

V, WDL and regret share a spatial state encoder that projects each square to
32 features, flattens all 81 squares and produces a 256-dimensional state
vector. Plies-to-end and final-material prediction continue to use the pooled
mean of the head features. The earlier conv_g128 run remains the 112-width
model.

Four things separate this line from `sq`:

- **46 planes** (item 1) instead of 25, including goal-path and arrival
  distances per piece type through the squares that type may enter, and both
  the plies since the last capture and the plies left before the clock draws.
- **Exact tactics** (item 14) shared with the play-time search: wins at once
  and losses in two at every expanded node, wins in three at the root.
  A winning move is worth +1 and is the move played; a move that loses in two
  is worth -1, is skipped by selection and by the root ordering while another
  move exists, and takes no policy-target mass; a node whose every move loses
  is worth -1. The same three labels train the tactics head.
- **A per-game capture clock** (item 19) drawn uniformly from 50 to 200 plies
  without a capture, counted down by plane 15; a clock ending pays
  `-0.05 x sign(material lead)` to the mover, and a game truncated at 1000
  plies keeps its own search value as its label.
- **Direct distillation** (item 26): `conv.distil` defaults to the conv_g128
  checkpoint-150 EMA and trains the student on every position toward its full
  policy, categorical Q, categorical V and WDL distributions, plus the current
  exact tactics labels, without search. The earlier scalar-Q `sq` teacher path
  remains selectable. Self-play starts from the distilled student's EMA.

## Files

- `src/` (Rust crate `conv`): `encode` for the 46 planes, `OrtNet` (the
  `search::Evaluator` over an exported graph and its sidecar) and `ConvPlayer`,
  which implements `match::Player` on top of `search::Gumbel` with this line's
  play settings; `ConvPlayer::load(onnx_path, threads)` takes the default
  settings, `with_settings` a `PlaySettings`.
- `py/conv/`: `planes.py` (board constants, the reference plane encoder and
  `tactics_reference`, the brute-force tactics oracle), `kernels.py` (the
  Triton rules: apply, derive, the three tactics checks), `env.py` (the
  batched environment with the per-board clock), `model.py` (the network),
  `search.py` (the batched Gumbel search), `replay.py` (rollout windows and
  labels), `control.py` (the regret-guided search control: the prioritised
  regret buffer self-play restarts from, DESIGN.md item 33), `train.py`
  (self-play loop and losses), `export.py` (ONNX), `diagnose_control.py`
  (offline CPU diagnostics of RGSC ranking and replay feedback), `control_audit.py`
  (independent label reconstruction and repeated-opening analysis), `probe_control.py`
  (frozen-checkpoint candidate comparisons using independent continuations),
  `probe_parity.py` (evaluator, search and saved-pool comparisons against production inference),
  `fit_control.py` (frozen control-head feature extraction and reconstruction checks),
  `config.py` (every tunable), `paths.py`.

## Self-play search

`search.GumbelSearch` keeps one tree per board in a static arena. A node
stores only its legal moves, in 80 slots (`search.SLOTS`: ten pieces with
eight king steps each) that name the action of each move, so the arena costs
about an eighth of an all-actions layout and the descent kernel runs over
128 lanes. The actor is a callable `evaluate(board, since, ply, clock) ->
(logits, q, value, draw, legal, count[, rank, regret])`: the student supplies
its state value V as the leaf value (`distil.py` trains that head on the
teacher's V distribution, or policy-weighted Q when the teacher is sq), the
WDL draw probability of every state for self-play contempt (`q` selects the
older per-action Q-mass proxy), and, with the regret head, the search
control's ranking score and predicted regret of every node it expands. Each search returns a `search.Root`: the root's moves in their slots, the
policy target and the packed exact labels over those slots (`pack_tactics`,
`unpack_tactics`, `spread` to lay them out over the 648 actions), the root
value, the move to play and the root candidates with their completed Q. The
replay windows keep the slot layout; the learner permutes the moves with the
augmentation and spreads the target and labels afresh for every batch.

The search control's signed label is the suffix mean of calibration excess
`2Q(Q - z)`, with `z` in the value learner's terminal-return domain. Until the
rank head beats uniform selection for two consecutive windows, each fresh
game contributes its largest measured excess. Afterwards Gumbel-max samples
from its ranking distribution over played rows and tree nodes. Buffer priority
uses clamped excess through one observation and a variance-corrected squared
mean residual from two observations onward.

Default training collects 192 x 1024 rows per window, with full searches on
half of steps, and keeps up to 16 windows. The 5e-4 base learning rate is
held through iteration 40, cosine-decayed to 1e-4 at iteration 120 and then
held there; checkpointed rollback factors multiply this schedule. Final
material is supervised for win and capture-clock endings, independently of
the decisive-only plies-to-end mask.

## Python interface

`pip install -e ".[dev]"` from `py/` installs the package with pytest, ruff and
maturin; the `engine` wheel comes from `cargo xtask wheel`. On Windows,
`kernels.py` points Triton's C compiler (`CC`) at the tcc it bundles when `CC`
is unset. Commands, from the workspace root:

```
python -m conv.distil --run <name> [--distil.teacher conv|sq] [--distil.steps N] [--distil.ckpt PATH]
python -m conv.train  --run <name> --init runs/<name>/distil.pt [--resume runs/<name>/latest.pt] [--iters N]
python -m conv.export --ckpt runs/<name>/latest.pt --out runs/<name>/export.onnx [--weights ema|model]
python -m conv.diagnose_control --run <name> [--start N] [--stop N] [--audit] [--out PATH]
python -m conv.probe_control --ckpt runs/<name>/ckpt_NNNNNN.pt --device cpu|cuda --estimand state_policy|action_mode|root_context --out PATH [--games 2] [--pairs 2] [--anchors 1] [--max-root-attempts 128] [--seed 0] [--resume]
python -m conv.probe_parity --study execution --ckpt runs/<name>/ckpt_NNNNNN.pt --pilot PATH --device cuda --out PATH [--resume]
python -m conv.probe_parity --study batch_shape --ckpt runs/<name>/ckpt_NNNNNN.pt --prior PATH --device cuda --out PATH [--resume]
python -m conv.fit_control --ckpt runs/<name>/ckpt_NNNNNN.pt --parity PATH --device cuda --out PATH
```

A run is written to `<workspace root>/runs/<name>/` whatever the working
directory (`paths.py`). Relative `--distil.ckpt` paths are workspace-relative;
`--init` and `--resume` are resolved from the working directory. Distillation
writes `distil.pt` and `distil.csv` there; training then starts from
`distil.pt` with `--init`.

Every config field is a flag, for example `--search.sims 64` or
`--learn.batch 512`; `runs/<name>/config.json` records the values used.

A run directory holds `config.json`, `latest.pt`, `ckpt_NNNNNN.pt` every
`checkpoint_every` iterations, `log.csv` with one row per iteration (losses,
the acting network, conversion metrics, timings) and the replay windows.

A checkpoint holds the trained weights, their EMA, the optimizer moments and
the config. `--init` starts a fresh run from a checkpoint's weights;
`--resume` continues one with its optimizer state and iteration count.

Resume restores network, play and rule settings from the checkpoint; supply
the original learning, search and control flags explicitly. It restores replay
windows and the RGSC buffer but starts fresh actor games and random streams;
unfinished old outcome/regret tails remain masked. The `conv_g192p10` run's
launcher, `runs/conv_g192p10/scripts/launch_train.ps1 -Resume <checkpoint>`
(kept with the run, untracked), keeps its original flags, rejects overlapping Python GPU workloads and CPU-test
environment settings, and archives existing stdout/stderr before launching.

`diagnose_control` reads current-format saved windows through the last logged
iteration (or `--stop`) and writes `rgsc_diagnostics.json` in the run directory.
It uses one CPU torch thread and performs no model inference. The report
reconstructs the logged pooled rank lift, measures effective sample size and
per-game played-row lift, and supplies game-bootstrap intervals and offline
temperature sensitivity at 1, 2 and 4. It also compares replay outcomes with
the preceding immutable checkpoint when available, counting missing feedback
and separating raw-label error from clipping. It does not change training.
The `within_actor_game` diagnostic fixes each actor window's selection mass
to its share of a game's rows, isolating ranking within one actor version.
Games already underway at `--start` are excluded; check `log_lift_matches`
before comparing a reconstructed iteration with its log. These are diagnostics
of the saved actor's played rows, not an evaluation of strength or of the
unrecorded alternatives in search trees. Intervals are conditional on these
games; common actors and repeated openings can introduce further dependence.
For checkpoint comparisons, also check `lost_keys_match` and
`tree_error_matches`: an unlabelled final row does not distinguish an unfinished
game from a ply-cap ending, so final-row cap evictions cannot be reconstructed.

`--audit` independently reconstructs the signed and paper squared-error suffix
targets from saved Q and terminal returns, checks labels and outcome perspectives,
and compares selection against both targets. Repeated-opening checks split games
into disjoint prefix and confirmation groups within one actor window; the stricter
grouping also fixes the opening action and full/cheap search mode. Pair products
remain signed; selection compares the top quarter separately within each actor.
Intervals cluster exact openings across actor/context groups;
shared search modes and adaptive selection still prevent an IID interpretation.
The report includes exact synthetic counterexamples for variance, clipping,
teacher leakage, actor drift, outcome-dependent trajectory length, and vanishing
corrective gradients in the ranking objective.

The audit also counts immediate wins missing from the saved candidate array,
separately for full and cheap search, and distinguishes admission failures from
retained wins that were not played. Counts refer to dependent played rows, not
independent games. `first_win_shortening` measures delay in reconstructed games.
For its label sensitivity, the first immediate win ends the saved trajectory,
its selected Q becomes exactly one, and earlier logged Q values stay fixed.
`first_win_prefix_original` and `first_win_prefix_corrected` compare the same
retained positions using their renormalized saved ranks. Games without an
immediate win retain their original labels. This is an offline sensitivity
calculation, not a rollout of the corrected search or a strength estimate.

`probe_control` performs inference and search but never trains or uses the live
restart buffer. It requires an immutable checkpoint, an explicit device and a
new output path, or `--resume` with an existing artifact and identical protocol
and settings. Its bounded-return protocol requires a finite `clock_penalty`
within [-1,1]; unsupported settings fail before creating progress or initializing
CUDA inference. It atomically saves each completed episode and root-search
attempt; an interrupted in-flight unit is rerun. Use one process per artifact.
CPU mode enables the Triton interpreter; full-size checkpoint rollouts can be
very slow. Run CUDA probes only when the training GPU is free.
The EMA network runs eagerly, with BF16 autocast on CUDA; search still uses
CUDA graphs. Parity with the compiled production actor requires separate
validation. The report records eager execution and the actual backend flags.
The default two-game pilot measures cost and checks collection; it is too small
to establish which selection method is better.

Each discovery game records rank sampling, rank argmax, predicted-regret argmax,
and uniform selection over played and eligible tree-node occurrences. Each
discovery episode also retains the full candidate pool in a compressed,
checksummed payload: unique states and ordered occurrences with source,
step/node location, rank and predicted regret. `candidate_batches` decodes the
original offer batches; `replay_candidate_pool` verifies the recorded selections
and probabilities without inference. The game summaries retain the four
selections; the full payload lives in the corresponding discovery episode.
Resuming validates the pool before reusing completed work. These alternatives
allow a new head to choose from the original pool, but its choices still need
fresh held-out continuation measurements. Independent
continuation pairs estimate the explicitly selected opening quantity:

- `state_policy`: squared mean residual over stochastic root searches and
  continuations; errors of opposite sign between actions can cancel.
- `action_mode`: mean squared residual bias conditional on the on-policy
  opening action and full/cheap mode. Another independent root search is
  rejection-matched to that action/mode before independent continuation.
- `root_context`: squared residual bias conditional on a reproduced complete
  root search, with independent future randomness. It retains variation
  between root-search contexts that the head may be unable to remove.

A separate base continuation supplies uniform future anchors, each labelled by
new independent pairs. Every episode has separate search, search-mode and environment streams;
candidate recording uses independent random streams and does not affect play.
The artifact records state/episode identities, checkpoint and source hashes,
execution precision, root-tree fingerprints, candidate probabilities, pair
provenance, attempt costs, coverage, censoring, and game-bootstrap intervals.
Ply-capped discoveries produce no confirmations. Other incomplete comparisons are
excluded and counted; this can bias estimates. `--max-root-attempts` bounds
coarse-context rejection work; exhausted attempts are missing measurements,
not zero-utility observations. Each comparison also reports a
`finite_sample_completion_range`: unknown signed pair products retain support
[-4,4], and averages retain every planned game, pair and anchor. This range
bounds completion of the sampled grid; it is not a confidence interval and
does not account for population or Monte Carlo uncertainty. Each anchor starts
with a fresh tree and
repetition history. These quantities describe error under that frozen restart
protocol, not learning gain or playing strength.

`probe_parity --study execution` compares the pilot's eager BF16/TF32-off evaluator, eager BF16
with TF32 enabled, and production `compile_net` with BF16/TF32 enabled. It runs
on the Windows CUDA runtime and requires a matching immutable EMA checkpoint
and completed two-game protocol-3 pilot. The supported actor uses 1,024 boards,
128/16 full-search simulations/candidates, 16/4 cheap search, a fixed capture
clock and active control/WDL heads. Checkpoint, configuration and source identity
are checked before inference. Run only while the GPU is free.

The command freezes a 16-state panel and a heterogeneous 1,024-state batch,
compares evaluator outputs and matched search contexts, repeats same-path roots,
checks two-ply mode switches, and rescores complete saved candidate pools while
preserving occurrence multiplicity and selector RNGs. Same-shape comparisons
use explicit modes and search seeds; equal master seeds do not couple ordinary
actor trajectories or different batch shapes. Search uses CUDA graphs in all
variants. Output includes numerical/action/selection differences, source hashes
and separate warm-up and measured costs; it does not declare global parity or
RGSC efficacy. Use a new output path, or `--resume` for the identical request.
Completed units are saved atomically; interrupted units rerun and compilation
warms up again in a new process. A named mutex prevents duplicate parity workers.

The `batch_shape` study takes a completed execution-parity artifact through
`--prior` instead of `--pilot`. It regenerates and saves original-shape Gumbels
from recorded CUDA seeds, then injects identical tensors into compiled scalar
and batched fresh roots. Its fixed 12-unit stress plan repeats the original full
batch and scalar rows 770/811, and a constructed cheap batch whose row 13 uses
the original scalar near-tie noise. Both shapes repeat each selected case.
Noise references, native-noise checks and final active-survivor margins are
recorded. The constructed cheap batch is not a replay of the old cheap batch.
This selected stress comparison preserves the prior artifact and tests numerical
shape sensitivity and repeatability; it does not estimate population accuracy
or long-trajectory equivalence.

`fit_control` extracts the actual 256-dimensional control-head inputs from the
full pools in a completed execution-parity artifact. It requires the matching
immutable EMA checkpoint and CUDA runtime, preserves original batch-1,024 chunk
ordering and repeat-last padding, and retains occurrence and discovery-family
mappings. BF16 features are stored as their exact 16-bit representation. The
compiled path rejects graph breaks and requires exact agreement with saved C
scores, unhooked versus hooked outputs, and the original head reconstructed
from captured features. A failure preserves its evidence for inspection. Use a
new output path while the GPU is free. The artifact contains development
features and model predictions; it does not contain utility labels or a fitted
replacement head.

Export writes `<out>` (input `planes`; outputs `logits`, `q`, `value`,
`plies_to_end`, `draw`, and `wdl` when the default WDL head is enabled) and
`<out>.json` (name, checkpoint hash, plane and atom counts, the prior
temperatures alpha and beta, plies-to-end class centres), then checks ONNX
Runtime against torch. The Rust side reads both files.

## Tests

The tests check the kernels and the planes against the `engine` binding on
random positions, the search semantics with a tiny network stand-in (including
the tactics and the policy target they shape), one collection and learner step
in both self-play phases, and export parity with torch. From `py/`:

```
TRITON_INTERPRET=1 python -m pytest -q
```

`TRITON_INTERPRET=1` runs the Triton kernels on the CPU; `tests/conftest.py`
sets it itself when CUDA is unavailable. `cargo xtask check` runs the same
tests on the CPU, `cargo xtask check --gpu` on the GPU.
