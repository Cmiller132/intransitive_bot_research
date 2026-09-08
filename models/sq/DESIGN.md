# sq design

The model as built, in numbered items. A change to any item is a design change
and needs approval.

## Input

1. 25 planes over 81 squares, mover's frame: 6 piece one-hots (own and enemy
   rock, paper, scissors), 3 enemy-threat planes by attacker type, home
   squares, plies since capture / 200, ply / 200, a constant plane, 6 race
   planes (own and enemy goal distances, race leaders, coordinate distances),
   6 mobility and material planes.

## Network

2. Stem: 3x3 convolution to width 128, plus a learned position embedding.
3. Trunk: 16 attention blocks, 4 heads of 32, GELU feed-forward of 512,
   pre-norm, a residual depthwise 3x3 mix at the start of each block.
4. Attention bias per block: a full learned 81x81 table per head, plus a
   rule-relation bias (beats, beaten by, same type, both own, both enemy,
   occupied-empty, empty-occupied, adjacent) with one weight per head and
   relation, plus a small Smolgen dynamic bias (compress 8, hidden 64,
   latent 16) on every third block.
5. Policy head: batch-normalised trunk features, then from-to attention,
   `q[from] . k[to] / 8 + from_term + dir_bias`, over the 648 actions, masked
   to legal moves.
6. Action-value head: one extra attention block, batch normalisation, then a
   from-to categorical head with 51 atoms in [-1, 1]; Q is the expectation.
7. Auxiliary heads, training only: occupancy 16 plies ahead, plies to end
   (16 classes), opponent reply, capture-danger squares, final material.

## Self-play search

8. Batched Gumbel search on 1024 boards: 16 root candidates from raw policy
   logits plus Gumbel noise, 128 simulations by sequential halving (16x2, 8x4,
   4x8, 2x16), interior deficit selection with sigma = (50 + max visits),
   completed Q = backed-up mean or the network Q when unvisited, leaf value
   = expected Q under the raw legal policy, sign-alternating undiscounted
   backups.
9. Exact tactics: a root or leaf whose mover has an immediate win (goal entry,
   last capture or stalemating the opponent) is valued +1 for that mover; such
   root moves are always candidates and, when present, the played move.
10. Root noise: Gumbel noise is scaled by 1.5 for the first 10 plies of a game
    and by 1 afterwards.
11. Tree reuse: after a move, up to 64 nodes of the played child's subtree,
    in creation order, seed the next search; their visits and values are
    kept.
12. Played move: the halving winner (subject to item 9). Policy target: softmax
    over legal moves of logits + sigma x completed Q. Root value: expected
    completed Q under the target.
13. Playout-cap randomisation: 25 % of steps use the full search, the rest a
    16-simulation, 4-candidate search that yields value labels only.
14. Training rules: capture clock 100 plies ramping to 50 over 50 iterations
    from a configured start iteration, then held; clock endings pay
    -0.05 x sign(material lead) to the mover; twofold repetition over game
    history plus search path is a draw inside the tree; games capped at 1000
    plies with value zero.

## Losses and optimisation

15. Policy cross-entropy to the search target on full-search steps.
16. Categorical Q loss on the played action toward the lambda return
    (lambda 0.8825, no discount) with HL-Gauss targets, sigma 0.75 atom
    spacings, plus the same loss on visited root candidates toward their
    completed Q, weight 1.
17. Auxiliary weights: occupancy 0.5, plies-to-end 0.25, reply 0.25, danger
    0.25, material 0.25. Plies-to-end and material are trained only on
    positions whose game ended decisively.
18. Replay: the 4 newest windows of 1024 x 128 positions on the host, each
    trained twice; batch 768; AdamW 5e-4 with weight decay 1e-4 on matrices;
    gradient clip 10; diagonal-flip and cyclic-type augmentation; bf16
    residual stream; replay persisted to disk.
19. Weights: an exponential moving average (0.999) of the trained weights
    acts in self-play and is what the export ships; checkpoints hold both
    with the optimizer state.

## Play

20. Rust: ONNX Runtime on CPU, batch 8 leaves per call, position cache;
    `search::Gumbel` with 16 candidates, 32 simulations, exact root checks,
    tree reuse.
21. Prior = softmax((q + beta log pi) / (alpha + beta)) over legal moves with
    alpha and beta from the export sidecar; leaf value = expected q under the
    prior.
22. Contempt 0.15: each action's q loses contempt x its draw mass (value
    distribution near zero) for the side to move at the root, and a rule draw
    is worth -0.15 to that side. A leaf that repeats a game or path position
    is shifted 0.1 against the root side. Moves-left utility of up to 0.04
    favours shorter wins and longer losses.
