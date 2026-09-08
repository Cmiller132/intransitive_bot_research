# match

Runs games between two players and reports results. This is the only way
games are played outside a model's own GPU self-play: evaluation, arbitrary
matches, and the site adapter all go through it. It knows the engine and the
`Player` trait and nothing about networks.

## Interface

- `Player` (player.rs): `new_game()`, `choose(state, history, clock) -> Action`,
  `name()`. A player receives every position it must move in; it keeps its
  own tree or cache across calls. `Clock` is `Sims(n)` or `Time(duration)`.
- `play` (game.rs): one game between two players from an `Opening` (random
  legal plies) under `Rules`, returning the winner, the end reason and the
  move list as a `GameRecord`.
- `eval` (eval.rs): the one evaluation. Paired games: each opening is played
  twice with colours swapped, every move at a fixed simulation count (32),
  no clock, pairs spread over threads. Reports wins, draws, losses, the paired
  margin and its 95 % interval.
- `rpsi` (rpsi.rs): the site adapter. `Session` speaks the RPSI protocol on
  stdin and stdout to the site's bot client (`rpsi`, `isready`, `setoption`,
  `newgame`, `position fen`, `legalmoves`, `go`, `stop`, `quit`), `Frame`
  converts the site's frame (Blue's home corner, FEN with rank 1 first and
  uppercase Blue, `a1-b2` move tokens) to the engine's canonical frame and
  back, and `move_budget_ms` sets the thinking time under the site clock.

Game records are written as JSON lines when a path is given; nothing else is
persisted.
