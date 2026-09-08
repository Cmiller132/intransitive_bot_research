# search

Generic Gumbel MCTS over the engine's rules. A model plugs in through the
`Evaluator` trait and gets a complete play-time search: batched leaf
evaluation, sequential halving at the root, completed Q, exact root tactics,
tree reuse between moves and a simulation or time budget. Models extend it by
implementing `Evaluator` differently or by wrapping `Gumbel` in their player.

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
  whether an immediate win was played, visits and completed Q per legal root
  action, and the root actions in search order.
- `History`: occurrence count of each position played so far in the game, for
  repetition checks.
