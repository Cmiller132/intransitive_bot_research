# search

Generic Gumbel MCTS over the engine's rules. A model plugs in through the
`Evaluator` trait and gets a complete play-time search: batched leaf
evaluation, sequential halving at the root, completed Q, exact short tactics,
tree reuse between moves and a simulation or time budget. The tactics come
from `engine::tactics`: at every expanded node a move that wins at once has
completed Q +1 and a move after which the reply wins at once has -1 without a
visit and is never selected while a move that does not lose exists, a node
whose every move loses is worth -1, and at the root a move that wins in three
counts as a win; a winning root move is played without search. Models extend
it by implementing `Evaluator` differently or by wrapping `Gumbel` in their
player.

## Interface

- `Evaluator`: `evaluate(states, signs) -> Vec<Eval>` where `Eval` holds the
  legal actions, their log prior, their action values, the state value, and
  an optional plies-to-end estimate; `signs` says whether each state is the
  root side's turn (+1) or the opponent's (-1).
- `Gumbel::new(params)`, `choose(evaluator, state, history, budget, rng) ->
  (Action, Info)`, `reset()`.
- `Params`: rules, candidates, `c_visit`, `c_scale`, batch, sims, `max_sims`,
  reuse, moves-left cap, slope and threshold, contempt, repetition penalty,
  repetition draw.
- `Budget`: `Sims(n)` or `Deadline(instant)`.
- `Info`: simulations, nodes and evaluations used, root value, plies left,
  whether a winning move was played without search, visits and completed Q
  per legal root action, and the root actions in search order.
- `History`: occurrence count of each position played so far in the game, for
  repetition checks.

Deadline planning carries a smoothed per-simulation cost between moves and
measures a conservative per-leaf cost afresh on each search. Batch admission
reserves time for returning the move and applies a margin to measured costs.
If any wall time remains, the first batch admits at least one descent even
when the cost estimate or return reserve would otherwise exclude it; later
batches must fit the estimate. A slow call can overrun its deadline, but its
per-leaf bound cannot starve later moves or games. An expired deadline or an
exact winning move can still return zero simulations. Fixed simulation
budgets ignore timing estimates and run the requested number of descents.
