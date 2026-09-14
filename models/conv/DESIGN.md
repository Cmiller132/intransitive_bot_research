# conv design

Status 2026-09-13: conv_g192p10, the ten-pair run of this design, is the
current line; its restart-off arm completed iteration 160 (RGSC.md). The
earlier runs conv_g128 (width 112) and conv_g160 are stopped.

Status: items 1-30 approved on 2026-09-08 and built; items 13, 18 and 26
amended and approved the same day. Items 31-33 and the 2026-09-10 amendments
are also approved. The current network has 6,031,492 parameters. The earlier
run conv_g128 remains the 112-width model (restarted 2026-09-09 with the
amendments to items 5, 6 and 25). Items are numbered;
a change to any item is a design change and needs approval. "sq item N" refers to
models/sq/DESIGN.md. The reference sq_g128 is the baseline to beat at equal
simulations: this design is about 6.03M against 4.16M.

The idea: local 3x3 convolution blocks alternate with full-board attention
blocks; everything the engine can compute exactly (obstacle-aware
reachability, material by type, the capture clock, short forced wins and
losses) is an input or an exact search value rather than something the
network must learn; the heads are a policy, a light action-value head, a
distributional state value, and training-only heads for tactics, occupancy,
plies to end and final material.

## Input

1. 46 planes over the 81 squares in the mover's frame, all in [0, 1]:
   - 0-5: piece one-hots (own rock, paper, scissors; enemy rock, paper,
     scissors).
   - 6-8: enemy attack maps by type: an adjacent enemy piece of that type
     could move onto the square.
   - 9-11: own attack maps by type, the same for own pieces.
   - 12: the home and goal squares a1 and i9. 13: constant one.
   - 14: plies since the last capture / 200. 15: plies without a capture
     left before the clock draws / 200 (the game's clock minus plane 14's
     count). The ply count is not an input; no rule depends on it.
   - 16-17: Chebyshev distance to the goal / 8 and to home / 8.
   - 18-23: goal-path distance per type. For own type T, the fewest king
     steps from the square to i9 through squares a T piece may enter (empty
     or holding an enemy piece T beats); enemy pieces of the other two types
     and own pieces are walls. Capped at 16, scaled by 1/16, unreachable = 1.
     Planes 21-23 are the same for the three enemy types toward a1.
   - 24-29: arrival maps per type. For own type T, the fewest king steps for
     any own T piece to reach the square through the same passable set
     (multi-source), capped and scaled as above; 27-29 for enemy types.
   - 30-31: moves available from the square for the own piece on it / 8 and
     for the enemy piece on it / 8.
   - 32-45: global values broadcast to every square: 32-34 own piece counts
     per type / 4, 35-37 enemy counts per type / 4, 38-39 own and enemy total
     mobility / 64, 40-42 race distance per own type (the least goal-path
     distance over own pieces of that type / 16, 1 when none), 43-45 the
     same for the enemy types.
   The GPU kernel and the Rust encoder both produce these planes and are
   tested against the engine binding, as for sq.

## Network

2. Width 112 throughout. Stem: 3x3 convolution from 46 planes to 112, plus a
   learned position embedding per square.
   Amended 2026-09-10: width 160, five attention heads (32 dimensions each)
   and feed-forward width 640. The six pairs, 64-dimensional policy, Q,
   tactics and reply from-to heads, and Smolgen on attention blocks 1 and 4
   are unchanged. Width 160 gives the trunk and its dense convolutions more
   capacity while the 32-dimensional attention heads unlock fused kernels;
   conv_g128 remains the documented 112-width run.
3. Trunk: six pairs of blocks, each pair a convolution block followed by an
   attention block. The residual stream is (batch, 81, 112), viewed as a
   9x9 grid inside the convolution blocks. Pre-norm with LayerNorm over the
   width everywhere; GELU. Residual stream in float32; the blocks' matmuls run
   in bf16 under autocast (amended 2026-09-09: a bf16 stream rounds away every
   block's contribution once its magnitude grows, which turned a drift in the
   first convolution block into a collapse in run conv_g128).
4. Convolution block: LayerNorm, 3x3 convolution 112 to 112, GELU, 3x3
   convolution 112 to 112, added to the stream. Zero padding; the position
   embedding and plane 12 tell the block where the edges and corners are.
5. Attention block: LayerNorm, 4 heads of 28, output projection, added to
   the stream; LayerNorm, feed-forward 112 to 448 to 112, added. No
   depthwise mix (the convolution blocks do the local mixing). Amended
   2026-09-09: QK-norm (queries and keys RMS-normalised per head with a
   learned scale per dimension) and the logits and softmax in float32. In
   run conv_g128 the scaled dot products of the fifth attention block reached
   56 after distillation and 396 by the third self-play iteration, where bf16
   logits are quantised to steps of 2; the softmax saturated, the compiled
   and eager backward passes disagreed by a factor of five, and the run
   collapsed twice at the same iteration.
   Amended 2026-09-10: use scaled-dot-product attention by default, with q, k
   (after QK-norm and learned scaling), v and the additive bias cast to one
   dtype. CUDA bf16 autocast therefore uses the memory-efficient bf16 kernel,
   while CPU uses float32. Head dimension 28 blocked the fused kernels; at 32,
   QK-norm bounds `|q.k| / sqrt(d)` so bf16 logits lose little precision. The
   explicit fp32 attention remains selectable and is used for opset-17 ONNX
   export, avoiding reliance on SDPA float-mask lowering at that opset.
6. Attention bias per attention block: a learned 81x81 table per head (sq
   item 4) plus a table over relation and relative offset, per head 7
   relations (beats, beaten by, same type, both own, both enemy,
   occupied-empty, empty-occupied) x 17x17 offsets, zero-initialised; plus
   Smolgen (sq item 4: compress 8, hidden 64, latent 16) on the second and
   fifth attention blocks.
7. Head features: LayerNorm then ReLU on the trunk output, shared by every
   head. No extra block in front of any head.
8. Policy head: sq item 5 (from-to attention, `q[from] . k[to] / 8 +
   from_term + dir_bias`, masked to legal moves).
9. Action-value head: the from-to categorical head of sq item 6 (51 atoms in
   [-1, 1], Q the expectation) on the shared head features.
10. State-value head: the mean of the head features over the squares
    concatenated with the features of a1 and i9 (336), one hidden layer of
    256, 51 atoms in [-1, 1]; V is the expectation.
    Amended 2026-09-10: a shared spatial state encoder applies a 160-to-32
    linear projection and GELU to every square, flattens the resulting 81x32
    features, then applies a 2592-to-256 linear layer and ReLU. Its 256-vector
    feeds V directly (256 to 51 atoms), WDL (256 to 3) and regret (256 to 128
    to 2); plies to end and final material stay on the pooled mean. The old
    mean plus a1 and i9 discarded threats and material arrangement away from
    the goal squares. Lc0's value head uses the same per-square projection,
    flatten and dense-256 construction. The direct WDL projection is enough
    because the shared encoder already supplies the nonlinear compression.
    `Net.state_head` selects it ("spatial"); "pooled" rebuilds the older
    heads, and a checkpoint whose config lacks the field (every conv_g128
    checkpoint) loads as "pooled", so the distillation teacher of item 26
    and the arena export of old checkpoints keep working.
11. Tactics head, training only: a from-to head with three logits per
    action: wins at once, loses at once to a reply, wins in three plies.
12. Auxiliary heads, training only: occupancy 16 plies ahead (7 classes per
    square), plies to end (16 classes) and final material (2 values), as in
    sq item 7. The reply and danger heads are dropped: the reply is the next
    position's policy, and danger is covered by the attack-map inputs and
    the tactics head.

## Self-play search

13. Batched Gumbel search: sq item 8, with the state value V of item 10 as
    the value of an expanded node before any visit (the same value the
    play-time search uses, item 27); the sq teacher of item 26, which has no
    state value, supplies its policy-weighted Q as in sq's own search.
14. Exact tactics, replacing sq item 9, shared with the play-time search
    (item 25). The engine gains a tactics module with three exact checks per
    legal move, each tested against brute force: wins at once (goal entry,
    last capture, stalemating the opponent); loses in two (the reply has a
    win at once); wins in three (every reply leaves a win at once). The GPU
    kernels mirror the first two at every expanded node and all three at
    the root. Completed Q is +1 for a move that wins at once or in three and
    -1 for a move that loses in two; a node whose every move loses in two is
    terminal with value -1. Selection inside the tree and the root ordering
    skip a move that loses in two whenever a move that does not lose exists
    (so a hopeless-looking non-losing move still beats a certain loss); a
    winning move, when present, is the played move. Immediate wins precede
    longer exact wins at candidate admission, every halving stage and final
    move choice. Legal, non-excluded and active-survivor masks remain binding.
15. Root noise: sq item 10 unchanged.
16. Tree reuse: sq item 11 unchanged.
17. Played move, policy target and root value: sq item 12 unchanged; a move
    that loses in two receives no target mass through its completed Q.
    Amended 2026-09-09 (contempt 0.1 in self-play, as at play, item 29):
    for the move choice and the policy target every root move's completed Q
    loses 0.1 x its draw mass (its Q distribution under the play-time draw
    kernel of width 0.15); the Q and value labels stay the zero-sum
    completed Q, so contempt shapes behaviour, not the value function.
    Draws inside the tree keep their zero-sum values. Amended 2026-09-09
    (conv_g128 from iteration 24): the charge is 0.05, half of the play-time
    contempt; self-play elsewhere (AlphaZero, Lc0, KataGo) uses none, and the
    Q-mass proxy also charges quiet moves in merely equal positions, so the
    charge stays small until item 31's head replaces the proxy.
    Amended 2026-09-10: `Search.contempt_source` defaults to `wdl`, so the
    0.05 charge uses the WDL head's draw probability. At the root an expanded
    move uses its child state's draw probability and an unexpanded move uses
    the root state's; draw probability is unchanged by the mover-frame swap.
    `q` remains available and keeps the old per-action Q-mass proxy. On 7000
    fresh rows from windows 140-151 with checkpoint-150 EMA, WDL P(draw) had
    ECE 0.013 and Brier 0.134 (constant predictor 0.234), while the played
    move's Q-mass proxy had ECE 0.117 and Brier 0.176, predicted 0.70 against
    a 0.63 draw rate, and its 0.9-1.0 bin drew only 80 %. The calibrated state
    head is therefore the better contempt signal. Zero-sum tree values and
    labels are unchanged; the export still emits both per-action `draw` and
    `wdl`, and play-time contempt in item 29 is outside this amendment. A
    node the rules settle stores an exact draw probability instead of the
    head's: 1 on a clock or repetition draw, 0 with a winning move, no move
    or every move losing.
18. Playout-cap randomisation (amended 2026-09-09, conv_g128 from iteration
    41; every step was a full search before): half of the steps, drawn per
    step, use the full search (128 simulations, 16 candidates) and yield a
    policy target; the other half a 16-simulation, 4-candidate search that
    yields value labels only, as sq item 13 with 0.5 in place of 0.25:
    self-play is four fifths of an iteration, and KataGo measured the
    randomisation at 87 Elo and 1.37x the time to strength; the larger full
    share keeps the value bootstraps, which come from the root search, on
    the full search half of the time.
    Amended 2026-09-10: the full-search share stays 0.5 and item 25 raises
    a window to 192 steps (run conv_g160 from iteration 3; iterations 0-2
    ran at 0.25). A 0.25 share was tried first for diversity: at 192 steps it
    gives 49k policy-labelled rows per iteration against the old 66k, and
    with 1368 learner batches each row is trained on 5.3 times instead of 8,
    so the policy head saw half the labelled samples per iteration of
    conv_g128 while three quarters of the lambda-return bootstraps came from
    16-simulation roots (KataGo's cheap search is 100 of 600 visits; ours is
    16 of 128). Doubling the steps at 0.25 buys only 16-simulation value rows
    and halves each policy row's training again. At 0.5 and 192 steps: 98k
    policy rows, 197k value rows, 5.3 passes per row, half the bootstraps
    from full searches, 13.8k simulation batches (about 12 minutes per
    iteration).
19. Training rules: the capture clock is drawn per game uniformly from 50 to
    200 plies without a capture and is what plane 15 counts down; clock
    endings pay -0.05 x sign(material lead) to the mover, where the lead
    respects the cycle (amended 2026-09-09: a piece is safe when the enemy has
    no piece of the type that beats it; the side with more safe pieces leads,
    and with equal safe counts the side with more pieces); twofold repetition
    over game history plus search path is a draw inside the tree; games are
    truncated at 1000 plies, and a truncated step's value label is its own
    search value, not zero.

## Losses and optimisation

20. Policy cross-entropy to the search target on full-search steps (sq
    item 15).
21. Categorical Q loss: sq item 16 (played action toward the lambda return,
    lambda 0.8825, HL-Gauss sigma 0.75 atom spacings; visited root
    candidates toward their completed Q), plus the same loss toward +1 on
    actions that win at once or in three and toward -1 on actions that lose
    in two, weight 1, on every step.
22. Categorical V loss toward the lambda return, HL-Gauss, weight 1.
23. Tactics head: binary cross-entropy on the three exact labels, weight
    0.25, on every step.
24. Auxiliary weights: occupancy 0.5, plies to end 0.25, material 0.25;
    plies to end and material only on positions whose game ended decisively
    (sq item 17).
    Amended 2026-09-10: final material is supervised when a game ends by a
    win or by the capture clock, using the material on the ending board in
    each row's mover frame; a game cut by the ply cap remains unlabelled.
    `Window.material_ok` carries this mask independently, while plies to end
    remains gated only by a decisive ending. About 80 % of games end by the
    clock, so the old shared gate discarded about 80 % of the available
    final-material supervision.
25. Replay, batch, optimiser, gradient clip, augmentation: sq item 18
    unchanged. EMA weights: sq item 19 unchanged. Replay amended 2026-09-09
    to KataGo's growing window: every window (128 x 1024 positions) is kept
    on disk zstd-compressed (about 24 MB each); the learner samples
    uniformly from the newest W windows, where W starts at 4 and grows as
    4 + (n^0.75 - 4^0.75) / (0.75 x 4^-0.25) x 0.4 in the windows produced
    so far (8 at 16, 16 at 50), capped at 16 (32 until conv_g128 iteration
    38: the value and Q targets are the collecting net's bootstrapped
    returns and are never refreshed, and against the iteration-36 net the
    stored returns of windows 16 to 23 iterations old disagree by a fifth
    more than fresh ones, so the window stops before that age; KataGo's
    outcome targets do not age); each iteration runs 1368 batches of 768
    whatever the window, so a position is trained on about eight times over
    its life as before. Amended 2026-09-09 after
    conv_g128 diverged at iteration 3 (the root cause was the attention logit
    growth of item 5; the collapsed weights also showed the first
    convolution block's output fifty-fold larger, a symptom): the dense 3x3
    convolution weights train at one third of the learning rate (Adam moves
    every weight by about the learning rate, so a layer's pre-activations
    drift in proportion to its fan-in, 1008 for the convolutions against 112
    for the attention layers; sq has only depthwise convolutions); linear
    warm-up over the first 500 steps from a fresh optimizer; a batch whose
    gradient norm is not finite or above 1000 is skipped; an iteration whose
    policy loss exceeds the last good iteration's by half, or whose mean
    gradient norm is above 1000, is rolled back to the last good weights and
    optimizer state and the learning rate halved.
    Amended 2026-09-10: each window is 192 x 1024 positions; replay remains
    at 1368 batches of 768. The base 5e-4 learning rate is held through
    iteration 40, cosine-decays to 1e-4 at iteration 120, then stays at 1e-4.
    Warm-up and the one-third convolution group multiplier apply on top. A
    rollback now halves a separate persistent factor stored in the
    checkpoint, so the configured base and its iteration schedule do not
    mutate. conv_g128 trained for 160 iterations at a constant 5e-4 and sat
    flat at about +25 Elo over sq for its last 40; AlphaZero-style engines
    decay the learning rate rather than hold it indefinitely.
26. Start by direct distillation from the reference, without search: sq_g128
    plays games with its raw policy on the training environment (its own 25
    planes through `sq.kernels.derive`), and on every position the student
    trains toward the reference's policy (cross-entropy), its per-action Q
    (categorical, HL-Gauss) and its policy-weighted Q as the state value,
    plus the exact tactics labels; positions stream through once. Self-play
    then starts from the distilled weights with the student's EMA as the
    actor from iteration 0.
    Amended 2026-09-10: `Distil.teacher` defaults to `conv` and its checkpoint
    to `runs/conv_g128/ckpt_000150.pt`; `sq` keeps the path above as an
    alternative. A conv teacher is built under its checkpoint's own
    width-112 config and reads the same 46 derived planes as the student. The
    student matches the teacher's legal-masked policy and its full 51-atom Q
    and V distributions by cross-entropy, and its WDL probabilities by
    cross-entropy; reply and regret are not distilled, and exact tactics come
    from the current clock-aware kernels. The teacher chooses moves from its
    policy at the existing distillation temperature. It was trained with the
    clock-blind tactics kernels, so its Q near clock expiry contains the old
    +-1 labels on 0.005-0.03 % of rows, but the exact tactics labels produced
    during distillation are correct.

## Play

27. Rust: sq item 20 (ONNX Runtime on CPU, batch 8, position cache, 16
    candidates, tree reuse, 32 simulations under the eval clock, the time
    budget under the site clock); the leaf value is V from item 10. The
    export carries logits, q, value, plies_to_end and draw.
28. Prior and leaf policy: sq item 21 unchanged (alpha and beta from the
    sidecar).
29. Contempt, repetition penalty, moves-left utility: sq item 22 unchanged.
30. Exact tactics at play, in the shared search crate so that every model
    including the reference searches the same way: item 14 with the engine's
    checks, wins-at-once and loses-in-two at every expanded node, wins-in-
    three at the root. The wins-in-three check skips moves that a sound
    necessary condition rules out (no mover's piece beside the goal, more
    than one enemy piece, enemy mobility too high to be removed by one
    move), so it costs a few milliseconds only in positions near the goal.
    Amended 2026-09-10: the GPU tactics kernels (`conv/kernels.py`
    loses_in_two / wins_in_three) take each board's plies since the last
    capture and its capture clock, as the engine's tactics do through
    `apply`: a non-capture move on which the clock expires draws, so it is
    neither a loss in two nor a win in three, and a non-capture reply on
    which it expires draws and refutes a win in three (a clock of 0 never
    expires). Until then they ignored the clock, so with one or two clock
    plies left the training search gave moves a completed Q of exactly +-1
    (and the Q loss its +-1 target, item 21) where the environment then drew
    by the clock; the play-side Rust search was never affected. The brute
    force oracle `planes.tactics_reference` takes the same two arguments.

31. Win / draw / loss head (added 2026-09-09, `Net.wdl`, off by default until
    tested): three logits from the value head's features, trained by
    cross-entropy (weight 0.5) toward the final result of the row's game from
    the mover's view, +1 / 0 / -1. Results are back-filled when a game ends,
    into earlier windows when it began there (rows of a game cut by the ply
    cap stay unlabelled), and a window is rewritten on disk when it gains
    labels. The value head's targets are lambda returns, a scalar
    bootstrapped from search values eight plies out: a coin flip between a
    win and a loss and a certain draw both regress to 0 and look alike to
    it, and to the Q-mass draw proxy of items 17 and 29. The outcome head
    separates them, never goes stale, and is meant to replace the proxy as
    the draw probability contempt charges once its calibration is checked.
    With the head on, the export adds a `wdl` output of the three
    probabilities; a checkpoint without the head loads with the head fresh,
    which is how `--net.wdl true` on resume turns it on for a running run
    (conv_g128 from iteration 24). Windows written before the labels existed
    carry none and contribute nothing to its loss.
    Amended 2026-09-10: `Net.wdl` is on by default and the head is a direct
    256-to-3 projection from item 10's shared spatial state vector. The shared
    nonlinear encoder retains board arrangement before outcome calibration.
    Amended 2026-09-10: the calibration reported in item 17 passed, and
    self-play contempt now consumes this head's state draw probability by
    default.

32. Opponent-reply head (added 2026-09-09, `Net.reply`, conv_g128 from
    iteration 41): a from-to head like the policy's, trained by
    cross-entropy (weight 0.15) toward the move the opponent actually played
    next, carried into the mover's frame by the anti-diagonal mirror that
    separates consecutive plies (`planes.flip_actions`), on every row whose
    next row is the same game. KataGo's ablation of its opponent-move head
    is the largest of its auxiliary targets (74 Elo, 1.30x the time to
    strength). Training-only, not exported; a checkpoint without it loads
    with the head fresh; a window's last rows learn their replies when the
    next window arrives; windows written before the label existed derive it
    on load from the stored moves, since every game then started at ply 0.

33. Regret-guided search control (added 2026-09-09, conv_g128 from iteration
    41; RGSC, Tsai et al., ICLR 2026): self-play restarts from the positions
    the agent misjudged most. A row's regret is the mean, over the rest of
    its game, of the squared gap between the played move's completed Q and
    the final outcome from each mover's view (`Outcomes` labels it with the
    outcome; a game cut by the ply cap has none). `Net.regret` is a head on
    the value features with two outputs, a ranking score and a predicted
    regret, computed at every search node too: the score trains by the
    paper's listwise loss, `-log sum_s softmax(score)_s exp(regret_s)` over
    the labelled rows of a batch plus one zero-regret dummy member (weight
    0.25), the prediction by squared error (0.25). Each game's candidate is
    the highest-scored of its played rows (a replay's first row, the state
    it started from, aside) and of the best non-terminal search node of
    each of its searches; when the game ends it enters the prioritised
    regret buffer (`Control.capacity` 512 states keyed by position, capture
    clock, ply and game clock) with its measured regret, or the node's
    predicted one, when there is room or it beats the lowest entry, which
    it replaces; a state already present moves toward the new regret by
    `ema` 0.5. A finished board restarts from a buffer state with
    probability `restart` 0.5, drawn in proportion to regret to the power
    `1 / temperature` (10), keeping the stored clock, plies since a capture
    and ply count, else from the initial position. The buffer lives in the
    checkpoint. Each board's restart is drawn before every step from the
    copy of the buffer made at the window's start and taken when its game
    ends at that step, so a replay ends one or two windows after its draw,
    which shapes three departures from the paper's game-by-game buffer: the
    replayed state is read from the game's first row, which holds it
    verbatim, not from its index in a copy since reordered by evictions; no
    entry takes more than `share` (1/16) of a copy's draw mass, since the
    paper's resampling after every game, which demotes a state as soon as
    it is measured, has no counterpart within a window; and a state's
    replays are folded in at the next upload as one `ema` step toward their
    mean regret (a state evicted meanwhile competes as a candidate again).
    A replay cut by the ply cap has no outcome and taught nothing about its
    state, which leaves the buffer rather than keep a regret no game can
    revise. Other departures: the ranking loss's candidate set is the
    batch; a restart carries no repetition history from before its position
    (only repetitions after it count); a resumed run with an untrained
    regret head spends `Control.warmup` (5) iterations without restarts,
    filling the buffer from played states by measured regret, in place of
    the paper's separate warm-up games with frozen policy and value (a
    labelled window is one sixteenth of the batches, so two iterations
    would train the heads on about one pass over a window's rows). Fresh
    heads on a resume are copied into the EMA actor and get zero optimizer
    moments before the first rollback snapshot. Every row records whether
    its game started from the buffer (`Window.replayed`), and the log
    carries the value loss on fresh and on replayed rows apart
    (`value_fresh`, `value_replay`): replays are the positions the value
    head got wrong, so a loss over the mixed rows cannot say whether the
    head improved on the distribution the arena plays.
    Amended 2026-09-10: the regret head consumes item 10's shared 256-vector,
    then uses its existing 128-unit ReLU hidden layer and two outputs. Spatial
    threats and material arrangements are relevant to predicting where search
    was wrong, so it shares the same representation as V and WDL.
    Observed 2026-09-10 (conv_g128 iterations 152-157): the buffer collapsed
    onto the clock-blind tactics labels of the amendment to item 30. States
    one or two plies from a clock draw with a "loses in two" or "wins in
    three" move carried a played Q of exactly +-1 and drew, so their regret
    was exactly 1.0 and could not be learned away; 88 % of the 512 entries
    had at most 5 clock plies left, replays started at ply ~297 with 7
    pieces, lasted 2 plies and drew 95 %, and half of all games were such
    replays (8 % of rows). Remedy: the kernel fix, then a restart with the
    buffer emptied and the warm-up restored (runs/conv_g128/clock_fix/).
    Amended 2026-09-10 (conv_g160 from its restart at iteration ~21; evidence
    and both Astra reviews in runs/conv_g160/astra_rgsc/): the buffer had
    converged on descendants of one game (310 of 512 entries with one game
    clock at iteration 14, 167 of the last 260 arrivals search nodes entered on
    predicted regret). Two causes. First, a departure from the paper's
    Algorithm 1 (lines 31-35): a restarted game only updates its opening and
    never stores a candidate, whereas `finish` took candidates from a replay's
    later rows and its search tree, so a lineage seeded itself. Now a replay
    only observes its opening. Second, the regret (Q - z)^2 on +-1 outcomes
    is bias^2 + Var(z), and on the openings replayed most the variance
    dominated (0.61 against a squared bias of 0.07): the buffer ranked
    balanced positions the net valued correctly. Tested and rejected: a
    correction by the WDL head's variance (the raw head predicts ~0.7 for
    every such state) and search surprise, post-search value against raw V
    (search agrees with the raw net on exactly the misjudged openings). Adopted:
    every game played from an opening, the proposing game included, is an
    observation of its played Q and its outcome in the Q domain (the terminal
    row's return, parity-adjusted, so a clock ending counts its material
    penalty); from two observations on the opening's priority is the squared
    gap between its mean played Q and its mean outcome, plain running means,
    the paper's discrepancy between evaluation and outcome estimated over the
    repeated games (it equals the paper's rule for deterministic outcomes; on
    the recorded replays coin flips fall to 0.09 after six games while
    misjudged openings stay near 0.45). The paper regret is kept per entry,
    folded over a window's m replays as (1-a)^m R + (1-(1-a)^m) mean(r), the
    order-independent form of m sequential EMA steps, and clamped to [0, 4]. A
    replay of an opening evicted meanwhile is lost, not resurrected (`upload`
    re-inserted it before). `Control.temperature` 0.1 -> 0.5: the paper's
    0.1 was tuned for per-game resampling, while a whole 192-step window draws
    from one copy, so the 1/16 cap let the top entries take ~30 replays each;
    the cap stays as the batching safety valve; no per-clock or per-lineage
    cap (a clock is not a lineage). Logged: priority mean/max, the fraction of
    scored and of tree entries, the largest single-clock share, candidates and
    admissions by source, lost replays, the draw's effective sample size and
    top-16 mass, predicted minus first measured regret of tree entries, the
    replayed openings' mean priority, played-Q drift between replays, and the
    replayed-row fraction of the window. The rank and regret heads keep the
    trajectory regret labels and losses unchanged.
    Amended 2026-09-10 (evening): the measured label separated confidently
    wrong rows from flat rows, 1.07 against 0.55, but the ranking head put them
    at the 0.35-0.38 percentile while putting ply < 5 rows at the 0.74-0.84
    percentile. In this game the outcome variance term of E[(Q-z)^2] = bias^2
    + Var(z) dominates throughout the opening; 95 % of the buffer became
    opening tree nodes carrying the regret head's predicted mean, and replays
    became near-fresh games. Candidates are now the played row of highest
    measured regret, the paper's "without ranking network" ablation of
    appendix F.1, with no tree nodes. They are queued and inserted only after
    the window's replays are folded, matching Algorithm 1's update-before-store
    order and ending the 60-100 lost replays per iteration. The rank and regret
    heads continue training as auxiliary heads.
    Amended 2026-09-10 (night): the paper's squared-error label is bias squared
    plus outcome variance, so its premise that high regret is rare fails in
    this game. Regret is now calibration excess 2Q(Q - z), using the Q-domain
    terminal return; the two heads are unchanged. Equation 3's ranking
    distribution samples one candidate by Gumbel-max over played rows and tree
    nodes instead of taking the argmax, so an uninformative head falls back to
    uniform tree restarts. Priority is clamped excess until two observations,
    then the persistent squared residual rbar^2 - s^2/n. The paper's Figure-7
    readiness test gives the heads control after rank-weighted measured excess
    beats uniform selection for two consecutive windows. The transition resets
    the head and its optimizer moments, versions away old labels, and starts
    from an empty buffer.
