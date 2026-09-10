# match

Runs games between two players and reports results. This is the only way
games are played outside a model's own GPU self-play: evaluation, arbitrary
matches, and the site adapter all go through it. It knows the engine and the
`Player` trait and nothing about networks.

## Interface

- `Player` (player.rs): `new_game()`, `choose(state, history, clock) -> Action`,
  `name()`. A player receives every position it must move in; it keeps its
  own tree or cache across calls. `Clock` is `Sims(n)` or `Time(duration)`.
  Defaulted methods: `observe(action)` is told every move (opening
  plies included) for players that track the game externally, `forfeited()`
  reports that the last `choose` produced no move, and `info()` returns the
  search behind the last move (`MoveInfo`: root value, the played move's Q
  and visit share, plies left, the top root moves) for players that expose it.
  `supports_clock` validates budget support, `set_leaves` configures an optional
  training leaf sink, and `search_details` adds model-specific search fields
  to recorded MoveStats without inventing policy visits for alpha-beta.
- `play` (game.rs): one game between two players from an `Opening` (random
  legal plies) under `Rules`, returning the winner, the end reason (goal,
  elimination, stalemate, capture clock or forfeit), the moves as absolute
  tokens (Blue's home a1) and each mover's `MoveStats` as a `GameRecord`.
  `play_observed` also reports every played move as it happens.
- `eval` (eval.rs): the one evaluation. Paired games: each opening is played
  twice with colours swapped, every move at a fixed simulation count (32)
  or `EvalConfig.move_ms` wall time, pairs spread over threads. An optional
  `reference_move_ms` overrides only the reference time; `reference_sims` instead
  sets its simulation count. Both require a timed candidate and are mutually
  exclusive; the runner swaps budgets with the players. Reports carry effective
  `reference_move_ms` and `reference_sims`, with the unused unit null. Reports wins,
  draws, losses, the paired margin and its 95 % interval, plus a seeded
  10,000-resample opening-pair bootstrap interval on the same [-1,1] scale.
  Complete pairs exclude forfeits, which remain counted in the WDL report.
  Timed reports carry `move_ms` and `sims=0`. With `stream` every game's start, moves (with
  stats) and end go to stdout as JSON lines while it plays.
- `rpsi` (rpsi.rs): the site adapter. `Session` speaks the RPSI protocol on
  stdin and stdout to the site's bot client (`rpsi`, `isready`, `setoption`,
  `newgame`, `position fen`, `legalmoves`, `go`, `stop`, `quit`), `Frame`
  converts the site's frame (Blue's home corner, FEN with rank 1 first and
  uppercase Blue, `a1-b2` move tokens) to the engine's canonical frame and
  back, and `move_budget_ms` sets the thinking time under the site clock.
  `go sims N` fixes the simulation count for one move. Explicit
  `go movetime N` passes N milliseconds unchanged to the player, including
  short eval budgets; only Fischer clocks use the site clock allocation.
  Players exposing `MoveInfo` also emit `info json {"info":...,"search":...}`
  before `bestmove`. `info` contains canonical action indices and `search`
  carries the player's optional search details, including NNUE nodes and time.
- `analysis` (analysis.rs): `Analyser`, what a model exposes for analysis
  (`heads`: network policy, action values, state value and its source,
  plies-to-end distribution, draw mass; `search`: a fresh model search), and
  `serve`, the JSON-lines protocol behind `bot analyse`: a request names a
  setup board (or the standard one), the side to move, the capture clock, a
  move line and a simulation count; the response carries the legal moves,
  the heads, the search lines with principal variations, or how the game
  ended. All boards and tokens are absolute.
- `client` (client.rs): the other end of RPSI. `RpsiPlayer::spawn(command)`
  starts an external engine on its stdin and stdout, completes the handshake,
  and plays it as a `Player`: every move request is the site's start FEN
  (Blue's home at i1) with the full move list, the legal moves, and
  `go sims N` or `go movetime ms`. An illegal, unparsable or late move
  (180 s), a crash or a failed handshake forfeits the game; the engine's
  stderr passes through. Any engine that plays on the site can take a seat.
  Optional `info json` telemetry is forwarded to game records and cleared
  before every move and new game; engines without it have no recorded stats.

Game records are written as JSON lines when a path is given; nothing else is
persisted.

## Random moves for data generation

RandomMoves::new(count, from, to, seed) samples distinct zero-based game plies
uniformly from the inclusive interval. play_with_random_moves uses the same
runner as play and requires the injection interval to start after the opening
(at its first subsequent ply or later). RandomMoves::plies() exposes the planned
slots. The chosen action is uniform among legal moves not marked loses_in_two
by engine::tactics; when every legal move loses immediately, normal play is used.
Early game end and unsafe slots can reduce the actual count; slots are never
rescheduled. Winning and drawing random actions are allowed.

GameRecord.random_plies records actual injected indices into moves; it is omitted
when empty. Injected moves have null stats, bypass choose, and are delivered to
both players through observe. All ordinary history, clock and end handling is
shared with normal play. A zero-count schedule preserves ordinary records.


When auditing paired records, identify the model from first/second and the
actual mover; the reference changes seat between the two games. A timed NNUE
move reports sims=0 because this field counts Gumbel simulations. Its search
work is in search.nodes; zero sims alone does not mean NNUE skipped search.
