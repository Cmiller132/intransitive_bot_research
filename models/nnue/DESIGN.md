# nnue design

Status: proposed 2026-09-09 by Claude and Astra, built the same day; every
item below is implemented and carries its measurement where one exists.
Items are numbered; a change to any item is a design change and needs
approval. The process that produced the deployed network and the work that
remains are the last two sections.

The idea: a sparse, incrementally updated integer network as the static
evaluator of an alpha-beta search on the CPU. The network is trained on
searched positions from the self-play models (any model that writes replay
windows) and on positions from its own games labelled by such a teacher, and
selected by games, never by validation loss. The prototype this continues
(the read-only workspace D:/Research/NNUE) scored 1-6-25 against
sq_g128g@60 at 750 ms; its integer arithmetic and search are migrated
unchanged, everything around them is rebuilt.

## Structure

1. Package: the workspace crate `nnue` (evaluator, alpha-beta search,
   `NnuePlayer: match::Player`) and the Python package `nnue` (features,
   network, quantisation-aware training, export, dataset import, student
   collection, teacher labelling, training). Rules and exact tactics come
   only from `engine`; the crate keeps no second implementation of moves,
   apply or terminal checks. `nnue:<path.nnue>` builds the player in the
   `bot` CLI. No native Python extension: features are NumPy over the same
   integer cell coding, and rule checks go through the existing `engine`
   wheel.
2. Player: `choose` refreshes the root accumulator and searches under
   `Clock::Time(d)` with a 20 ms overhead, or under `Clock::Sims(n)` for a
   fixed budget of n x 2,500 nodes on one thread: the arena and `bot eval`
   pair players at equal simulations, and 2,500 nodes is what one sq_g128
   simulation costs in CPU time on the site's processor (32 simulations,
   about 100 ms, against roughly 800k NNUE nodes per second per thread);
   the constant is declared here so the comparison is honest, and item 3 is
   the measurement at equal wall time. `new_game` clears the table and the
   move heuristics. `MoveInfo` carries nodes, depth, score and the principal
   variation in the fields that fit and leaves the search-only fields empty.
3. Timed eval: `bot eval --move-ms N [--player-threads T]` plays the same
   paired games under `Clock::Time` per move with T search threads per
   player; the report gains complete-pair counts and a seeded opening-pair
   bootstrap interval. The simulation mode stays the default for sq and conv.
   Project.md's definition of the eval gains this timed variant (a workspace
   change outside models/, listed here for approval).
4. Migration control: the prototype's release network (format 3,
   972 features, 512 hidden, 32 dense) is converted once to the new format
   with zero clock rows and must give identical raw scores on 1,024 stored
   positions; the new player must reproduce the prototype binary's fixed-node
   decisions on 100 stored positions and run no more than 5 % slower at
   equal nodes. This is a correctness check, not a strength claim.

## Evaluator (file format RPSNNUE1 version 6)

5. Features, 1,004: the prototype's 486 piece-square rows and 486
   attacked-piece rows (a piece attacked by an adjacent enemy that beats
   it), plus 16 elapsed-clock rows and 16 remaining-clock rows, one-hot over
   the lower bounds 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
   160 plies, shared by both perspectives. Two clock rows are active per
   perspective; they change only when a bucket boundary is crossed or a
   capture resets the clock. Clock limits 50-200 are supported; a position
   without a clock context is rejected rather than defaulted.
6. Transformer and head, migrated: two perspectives (the mover, and the
   anti-diagonal reflection with colours swapped) sharing H int16 rows
   (scale 255, H from the header, any multiple of 32; 512 by default, 768
   costs 12 % more per node) into int32 accumulators; squared clipped ReLU; an int16
   linear readout (scale 64); a 32-wide int8 dense residual with int32 bias
   and int16 output; value tanh(raw); search score round(600 raw). Scalar
   integer arithmetic with ties-to-even is the reference; AVX2 and VNNI
   kernels are equivalents tested against it.
7. Output buckets by total pieces on the board, 2-4, 5-8, 9-12 and 13-20:
   a bucket-specific linear readout, residual output and bias, with the
   transformer and the dense transform shared. Built as an ablation beside
   item 5; kept if it passes the gauntlet. Measured: neither the clock rows
   (82.5 % without them against the same control's 83.8 %) nor the buckets
   (50.6 %) changed strength; the format keeps both, trained or zeroed.
8. Deferred, each a separate design change when the data supports it: full
   per-bucket residual heads, a deeper residual (1024 to 32 to 16 to 1),
   width 768 or 1024, richer local features.
9. Loss: the control is the prototype's tanh mean squared error plus a
   0.02-weight bounded logit term; the ablation is soft cross-entropy with
   logits 2 raw against (target + 1) / 2. An outcome term is off by default
   and ablated at 10 % weight on verified played outcomes only. Sixfold
   symmetry augmentation with a paired consistency loss of 0.2 in every
   batch; no runtime ensemble. Quantisation-aware training throughout when
   fine-tuning, after a float warm-up from scratch. Row filters: `--quiet`
   drops rows where the mover can capture (41 % of the data), Stockfish's
   quiet-position rule; the selection objective for `best.nnue` is the
   validation loss over teacher and proof rows only, so a change in the
   outcome-labelled share cannot move the selection.
10. Exactness: the NumPy oracle over the file bytes, the Rust scalar kernel
    and the SIMD kernels agree exactly (measured: zero error on 1,024
    positions and every child); the PyTorch fake-quantised forward, which
    accumulates in float32, agrees with them within 2e-6 raw (measured
    1.9e-6 at worst); incremental updates equal a refresh over 1,000-move
    trajectories with captures and clock-bucket boundaries.

## Search

11. Kept from the prototype: iterative deepening, principal variation search,
    aspiration windows, a transposition table keyed on the board, the clock
    counter and the clock limit, killers, history, late move reductions,
    quiescence over captures and goal approaches with goal-threat evasions,
    and lazy SMP over a shared table (four threads at the site). Iteration
    stops when 72 % of the budget is spent.
12. Exact tactics from `engine::tactics`: wins at once and loses in two in
    every move loop, wins in three at the root; mate scores are separate
    from evaluation scores and a win precedes a clock draw; quiescence never
    stands pat through a proved immediate loss.
13. Improvements, in this order, each kept only if it passes the gauntlet of
    item 16 with the same weights: (a) table and hot-loop efficiency (reused
    buffers, cached static evaluation, Zobrist keys); (b) continuation and
    capture history, then history-aware reductions; (c) a goal-threat and
    single-evasion extension of one ply, at most two per path;
    (d) reverse futility, then late quiet futility at non-PV depth 2 or
    less with margins fitted on the evaluation scale, never pruning every
    legal move; (e) time management and SMP tuning at 250 ms. Measured:
    one at a time, only partial-root selection passed (59 %) and (a), (b)
    and (d) scored 48-52 % on 64-160 games, below the stages' resolution;
    as a bundle (a, history-aware reductions, both futility rules) over
    1,000 games at 50 ms and 500 at 100 ms they scored 57.4 % (54.5-60.2)
    and 55.7 % (51.8-59.5) against partial-root alone and are retained;
    (c) and null move (item 14) scored 47-48 % and are not in the bundle.
    Speed: quiescence proves stalemate (one legal move suffices) before
    generating the move list, +14 % nodes per second, 53.6 % (50.8-56.4)
    over 500 games; the sampled profile behind it (`nnue/profile` feature,
    one call in 64 timed, in `bot nnue diagnose`) puts the accumulator
    update at 24 %, the dense head at 19 %, move generation at 12 %,
    ordering and the table at 5 % each on H512; batching the accumulator
    adds and subs into AVX2 tiles gave no gain (49.5 %, reverted). (e) measured: four threads
    at 250 ms score 61 % against one thread, doubled time 65 %; the
    difference (-3.75 points, -11.25 to +4.00) is not a shortfall, so the
    lazy SMP stays; `bot eval --reference-move-ms` gives the yardstick.
14. Null-move pruning last and verified: non-PV nodes only, mobile
    positions with enough pieces, far from the clock and from either goal;
    no consecutive nulls; the null move is `engine::flip`, never a legal
    action, and its leaves are never training data.
15. The clock enters through item 5, not through score decay. A bounded
    adjustment of the static score near the clock may be ablated later with
    its own gauntlet; exact mate and draw scores are never scaled.

## Measurement

16. Gauntlet: candidate against the incumbent NNUE with frozen artifacts,
    64 MiB table each, site rules, no repetition draw, wins 1, draws 0.5,
    frozen eight-ply random openings deduplicated by symmetry orbit and
    played with both colours, through `bot eval --move-ms`, or under a
    fixed node budget (`--sims N`, N x 2,500 nodes) for evaluator-only
    changes, which is independent of machine load and runs a stage in
    minutes; a search change or a wider network is judged on the clock. Screen: 64
    games at 50 ms, one thread each. Accept: 160 new games at 100 ms, one
    thread each, at least 55 % with the bootstrap 95 % lower bound above
    50 %. Confirm, for a claim at the site budget: 128 new games at 250 ms
    with four threads each, one pair at a time; run for the candidate about
    to be deployed and for an accept below 65 %, not for every change, since
    a decisive accept at 100 ms has never been reversed at 250 ms here and
    the stage costs an hour of four threads. Seeds are declared before
    play; no early stopping and no best-of-seeds. Resolution: 160 games
    bound a change to about +-8 %, so search changes worth 1-3 % each
    (Stockfish's typical merge) are tested as a bundle over 1,000 games,
    and per-change gauntlets only reject what is clearly worse.
17. Final comparison: 16 games over 8 new paired openings at 250 ms for both
    sides with four threads each against sq_g128g@60 (`sq:` player on the
    deployed ONNX with the deployed search options; not the deployed
    binary's bytes). Recorded with hashes, options, platform, timings and
    failures. Eight pairs bound the gap; they are not an Elo estimate.
    Measured 2026-09-09 with 50 pairs: invalid as a strength claim, because
    the sq reference stopped searching after 31 games under the deadline
    (a retained call-cost bound in the shared Gumbel search; reported, not
    ours to fix); in the 20 healthy games sq had a median 33 simulations
    and the NNUE scored 42.5 %. Every NNUE loss was by goal, seen by the
    search 8-20 plies out while the static value stayed optimistic in
    some goal races, which points at late-game race positions as the next
    data target (item 35). Measured 2026-09-10 with `--reference-sims`
    (sq searching every move, audited): the round-7 network at 250 ms
    and four threads scores 63.5 % (55.5-71.0) against sq at 32
    simulations, the arena's budget, and 39.5 % (31.5-47.5) against sq at
    128. The arena seat plays at 32 x 2,500 nodes on one thread, so its
    rating understates the site-budget strength. On a shared clock with
    the repaired search (sq searching every move, audited per seat): the
    round-7 network 58.0 % (48.5-67.0) and the round-11 network 65.0 %
    (57.0-73.0) against sq at 250 ms and four threads each, 50 pairs.
    The retained search itself holds at that budget: 58.5 % (53.3-63.5)
    over partial-root alone with the same weights, 100 pairs.

## Data

18. Shards (`runs/nnue_data/<set>/`): columnar NumPy arrays, memory-mapped:
    board u8 (N, 81) in the canonical frame, since_capture u16, ply u16,
    capture_clock u16, target f32 in [-1, 1] from the mover's view,
    weight f32, kind u8 (played-root lambda return, candidate child minus
    completed Q, exact proof, teacher search value, raw network), outcome
    i8 with outcome_ok, source u8, game u32, orbit u64, split u8; plus
    provenance.json (producer, run, window hashes, teacher, simulations,
    rules, clock, shaping flags). Counters are exact, never averaged.
19. Importer `python -m nnue.importer conv --run <dir> --out <set>`: one
    adapter per window schema (conv schema 2 first, sq next),
    reading plain or zstd windows through one handle, hashing the bytes and
    validating shapes; unknown schemas fail. Labels: the played root gets
    its lambda return at weight 0.25; each visited candidate's nonterminal
    child, applied through the engine, gets minus its completed Q at weight
    1; outcomes only on played rows with outcome_ok. Episodes are
    reconstructed from the environment order across contiguous windows.
    Exact (board, since_capture, clock) contexts are deduplicated with the
    weight capped at 4. Splits 90/5/5 by episode and opening family, then
    six-symmetry board orbits are excluded across splits.
20. Two label domains kept apart: source-rule pretraining (the producer's
    clocks and shaping, as recorded) and site-rule labels (clock 200, no
    shaping) from fresh teacher searches; a training recipe names what it
    mixes and at what share.
21. Student collection `python -m nnue.collect`: per round 128 games over
    64 opening families through `bot`, the current NNUE against itself and
    against the previous one at 2k, 8k and 20k nodes, saving trajectories
    and the leaves the search evaluated; 10k states per round sampled as
    40 % played roots, 30 % evaluated leaves, 20 % legal alternatives
    around value drops, 10 % targeted sparse, long-clock and race
    positions.
22. Teacher labelling `python -m nnue.label` through `bot analyse` with a
    teacher profile (site rules, zero contempt, repetition penalty and
    moves-left, seeded): 256 simulations per root, 512 for hard cases, 10 %
    repeated at 1,024 to measure stability; exact proofs from the engine;
    the chosen action's Q and the root value recorded separately. Measured:
    sq_g128 at 128 simulations costs 150 ms per position per CPU thread.
    `python -m nnue.gpu_label --teacher conv|sq` labels the same rows with
    the batched Gumbel search of a training checkpoint when the GPU is free
    (about 250 rows per second at 128 simulations, batch 4,096); the
    teacher is chosen by arena rating. The two teachers agree (correlation
    0.981, mean absolute difference 0.066 on a student round), and teacher
    labels are the strongest lever measured: a fine-tune on 150k labelled
    human positions scored 70 % against the data-only control.
23. Training `python -m nnue.train --run <name>`: streaming shards, an
    explicit batch mixture (at least half broad window data), a feature-id
    cache per set (`python -m nnue.data encode <set>`, int16 (N, 2, 42))
    so a batch is a gather rather than an encode (the CPU trainer's
    bottleneck: 69 of 80 ms per step), the accumulation as a sparse CSR
    product on the CPU (PyTorch's embedding bag was 64 % of the remaining
    step; the product gives the same sums at a fifth of the cost, 24-29k
    rows per second on 4-8 threads), `--device cuda` for 230k-340k rows
    per second when the GPU is free (measured: continuing the fine-tuned
    lineage for six passes over every source scored 61 % at node budget and
    62 % on the clock over its parent, while nine passes from scratch
    scored 46 % despite a lower validation loss, so passes on the proven
    lineage count and validation loss alone does not select), AdamW with a
    cosine schedule, exact resume (weights, optimizer, RNG, sampler cursor),
    a thread flag, exports `<run>/best.nnue` with a JSON sidecar (hashes,
    schema, metrics); output under `runs/<name>/`.
24. Experiment order: migration control (item 4); a data-only 512-wide
    control on the conv windows plus the prototype's data; clock features
    (item 5); buckets (item 7); the search items of 13 with frozen weights;
    one student round (items 21-22); the final comparison (item 17).
    Winners are combined and retested.
25. Resources: below-normal priority throughout; the thread count follows
    the machine's headroom measured against the live GPU run's process
    (it needs little CPU: 30 % of 32 logical cores were busy with 12
    threads of ours), so labelling, the bottleneck, runs on eight workers
    when the load allows; CUDA hidden from every Python process; the live
    run's process and directory untouched except for reading its windows,
    configuration and exports.

## Process (how the deployed network was made, and how to make the next)

26. Lineage. Every accepted network is a fine-tune of the previous one;
    the chain is migration control (the prototype's weights converted)
    -> data-only control on the conv windows and the prototype's data
    -> teacher labels on human positions -> student rounds -> a many-pass
    continuation on every source. From-scratch runs never passed. Keep the
    chain: initialise from the incumbent (`--init`), never from random.
27. Sources, in order of measured value: (a) positions labelled by a
    stronger teacher's search (human games, the student's own games; the
    largest gains); (b) the producer's replay windows with its own search
    values (volume; the late windows are already close to the strongest
    teacher, correlation 0.95); (c) played outcomes (no measurable gain on
    their own; kept at weight 0.25 in the human import). Teacher labels
    come from `nnue.gpu_label` when the GPU is free (250 rows/s) and from
    `nnue.label` on the CPU otherwise (1-2 rows/s per thread with sq; far
    faster with the NNUE's own search, item 30).
28. One iteration: `nnue.collect` (the incumbent's games against its
    parent under node budgets, roots, leaves, alternatives and targeted
    positions) -> label -> `nnue.data encode` -> `nnue.train --init
    <incumbent>` on a mixture with about half window data -> gauntlet at
    `--sims 8` (evaluator-only, minutes) -> timed accept with the
    deployed search binary -> arena seat (`arena/upload.sh ... --replace`,
    then restart `arena`). Every command and seed is in the run's
    provenance or sidecar; the gauntlet verdict records the binary.
29. Search changes are tested separately from evaluator changes, with
    frozen weights on both seats, and bundled when each is small
    (item 13). A bundle needs 1,000 games at 50 ms and 500 at 100 ms to
    resolve a few per cent.

## Future work (not built; in the order Claude and Astra would take it)

30. Self-rescoring (built 2026-09-09 evening): `bot analyse --engine
    nnue:<file>` exposes the static evaluation as the network head (child
    static values as Q, no policy) and a node-budget search, so
    `nnue.label --engine nnue:<file> --sims 40` labels the student's
    positions with the incumbent's own search at 100k nodes (120 ms per
    row per worker) and rounds no longer wait for the teacher. Stockfish's
    data is made this way. Measured: a 10k-row round was unresolved
    (52-54 %); a 100k-row round (1,280 games, 26 minutes of labelling on
    eight workers) scored 59.7 % over 500 games at node budget and 57.2 %
    over 500 games at 50 ms against its parent, a full acceptance step
    with no teacher. The loop is now: collect 640 families, label with the
    incumbent at 40 simulations, fine-tune 12 epochs at 30 % share,
    resolve over 500 games. `nnue.label --quiet-best` is Stockfish's
    generation filter (drop rows whose best move is a capture); measured
    on the same 30k rows it dropped 18 % and scored 50.7 % (46.8-54.5)
    against the unfiltered labels, so the loop stays unfiltered. Label depth: the same 30k rows at 160
    simulations against 40 scored 48.8 % (44.9-52.5), so the loop keeps
    40 and spends the time on rows. Recipe checks on the same data: 24 epochs
    scored 50.8 % against 12 (no change), a 50 % self share scored 44.8 %
    against 30 % (worse; the window and teacher rows anchor the network),
    learning rates 2e-4 and 5e-5 scored 51-53 % against 1e-4, the late
    windows alone 50 %, and no windows at all 44.7 %, so the recipe
    stands and the windows stay in every mixture. Volume: a round of
    195k rows scored 55.0 % against its parent, a round of 100k rows from
    the same parent 55.1 %, so rows per round do not raise the gain.
    Depth: two node-budget promotions of 55 % each summed to 53.7 % at
    50 ms, and 160-simulation labels did not transfer better than
    40-simulation ones on the clock (48.8 %), so the loop's acceptance
    runs on the clock (500 games at 50 ms) and the site-budget gap is
    measured against sq at fixed simulations (item 17). Rounds vary: 6 scored 51 % and 7, from the same
    parent with a new seed, 56 %, so one flat round is not saturation.
31. Retrain when the teacher is clearly stronger: import the new windows
    (`nnue.importer conv --run runs/conv_g128 ...`), relabel the human
    and student sets with the new checkpoint through `nnue.gpu_label`,
    continue the lineage (item 26) for tens of passes on the GPU
    (`--device cuda`, 20-60 epochs of 1,000 steps at batch 8,192 take
    15-45 minutes), then the gauntlets of item 28. Expect the evaluator to
    track the teacher's mid-game judgement, which is where the losses
    against sq were.
32. Passes: the many-pass continuation was still improving at its best
    epoch (18 of 20); 60-150 epochs on the GPU are the next evaluator test.
33. Width 768 only after the passes are in place: it costs 12 % per node
    and its from-scratch run lost; a 768-wide continuation needs the
    lineage widened rather than random: `NNUE.widen_hidden` pads a 512-wide
    network to 768 (new rows zero, new bias 0.15, head columns zero) so it
    evaluates identically at first, and `--init <512.nnue> --hidden 768`
    applies it; judged on the clock, never at node budget. Measured: the
    padded 768 fine-tuned on round 5's data scored 43.7 % (39.7-47.5)
    over 500 games at 50 ms against the 512 it started from; the extra
    rows do not earn their cost in one fine-tune, so width stays 512 until
    a many-pass GPU continuation can try it.
34. Search: multithreading worth at four threads against doubled time, and
    the evaluator's cost per node (profile and SIMD work), both under
    measurement by Astra; then time management at 250 ms; sequential
    testing (SPRT) in `bot eval` if a paired-outcome likelihood is added
    cleanly.
35. The race blind spot: of 78 goal losses against sq, 59 were runner or
    tempo races and 19 capture or escort sequences; the static value was
    optimistic twelve plies out in 13, nine of them races. Built
    2026-09-10 (Astra: engine query and the Rust side, Claude: the Python
    side) as format 7: per side, the bucketed goal distance (1-8, none) of
    the best runner that passes a piece-type and tempo interception filter
    (`engine::race::race_buckets`, a geometric hint that ignores clocks,
    moving barriers and wins elsewhere), 18 rows after the clock rows, two
    active per perspective (own side, other side; the pair is queried once
    on the actual mover's board and swapped for the other perspective, so
    the tempo is the real one). Python encodes 44 slots for both formats
    and a format 6 model masks the race ids; `widen_features` pads the
    lineage with zero rows, and the widened ft_self11 searches identically
    on the Rust side. Measured cost at H512, one thread: 390 ns per query,
    about 28-30 % of search throughput (1.06 M to 0.74 M nodes/s on the
    benchmark positions), so on the clock the feature must be worth about
    seven points of score before it breaks even (a doubling of time is
    worth 15 %). Under test: ft_race16, the incumbent fine-tuned with the
    rows on round 16's mixture, against ft_self11 on the same binary at 50
    ms and at 8 simulations (the value net of the cost). If the value is
    there but the cost eats it, the next step is a cheaper query (skip it
    when no piece moved closer to a goal, or cache it by position). Also:
    teacher disagreement rounds at 512 simulations on the rows where
    student and teacher differ most.
36. Not worth repeating (measured null or negative): clock rows, output
    buckets, the quiet-position filter, human outcome labels alone,
    doubling the windows alone, null move, one-ply extensions, exact
    tactics beyond the current rule.
