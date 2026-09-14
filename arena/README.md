# arena

The rating runner and the site in one service: every model in the pool plays
rated games against the rest, each game is recorded with both players'
search statistics, and the same engines answer analysis requests. It serves
rankings, matchups, live and historical games, an opening explorer, a game
library with imports from the rps site, and the analysis board at
https://rps.blueshrimp.uk (through the Cloudflare tunnel) and on the LAN.

It runs on a Debian 12 container (8 threads, 6 GB) from `/opt/arena`: `bot` and `libonnxruntime.so.1` (the Linux build),
`backend/` (this package), `web/` (the built frontend), `venv/`, `players/`,
`arena.db`. Services: `arena` (uvicorn on the loopback, unit `arena.service`,
settings in `/etc/arena.env`), `caddy` (port 80, `Caddyfile`: static files,
`/api` and `/healthz` proxied) and `cloudflared` (the tunnel).

## Layout

- `backend/`: the Python package `arena`, the service and the rating runner
  (backend/README.md).
- `frontend/`: the Vite + React + TypeScript site (frontend/README.md);
  `dist/` is shipped to `/opt/arena/web`.
- `arena.service`, `Caddyfile`: the units installed on the container.
- `upload.sh`: adds a player from any machine:
  `arena/upload.sh <arena url> <name> <spec> [--replace] <files>...`
  (`ARENA_ADMIN_TOKEN` when not on the LAN).
- `watch.sh`: `arena/watch.sh <url> <run> <model>` exports and uploads every
  tenth checkpoint named `ckpt_NNNNNN.pt` (six digits) of a training run as
  `<run>_<n>` until stopped. Checkpoints with suffixes are ignored. Add
  `--once` for one scan from a scheduler; it returns nonzero if an export
  or upload fails. Failed exports are retried with their parity check.

`cargo xtask check` runs both packages' lints, tests and the frontend build.

## Players and games

A player is `players/<name>/player.json` (`{"name", "spec", "added"}`) plus
its files. The spec is what `bot` understands: `sq:<file>` or `conv:<file>`
for an ONNX export with its `.json` beside it, or `rpsi:<command>` for any
engine speaking the site's RPSI protocol, uploaded as a `.tar.gz`. Uploads
go to `POST /api/players`; a re-upload with `replace` discards the player's
games. Retired players keep their games but are neither scheduled nor shown.

Jobs are `bot eval` with 2 openings played from both sides at 32
simulations, one thread each, `ARENA_JOBS` at a time (4 by default), streamed move by move into the
database: a game is visible while it is played, and every move carries the
mover's search summary (root value, the played move's Q and visit share,
plies left, the top root moves), stored from Blue's point of view.

Ratings and pairings follow the KataGo training server's rating games
(`katago-server`: `BayesianRatingService`, `RatingNetworkPairerService`).
After each job, and whenever the players table changes (an upload, a
replacement, a retirement), the ratings are refitted: Bradley-Terry with draws half, four
virtual draws between every checkpoint and its parent (the previous
`<series>_<n>` of the same series; a player without one gets a very weak
prior of equality with everyone), minorisation-maximisation sweeps, and the
anchor (`ARENA_ANCHOR`, `sq_g128`) reset to 0. A player's uncertainty is its
own, `1 / sqrt(sum(games * p * (1 - p)))` over its games: how well it is
pinned to the pool, not how well the pool is pinned to the anchor, which the
anchor's own row shows. A history row is kept per player after each fit.

Each new job picks its candidate among the players that are neither settled
(half-width of 30 Elo or less after at least 8 games) nor over their budget
(`ARENA_MAX_GAMES`, 1000) by one of four modes: the possibly strongest (upper
confidence bound), the most uncertain, the fewest games, each at weight 1 as
on the server, and at weight 0.5 the fewest games against the anchor, which
then is the opponent. Otherwise the opponent is drawn from the players within
1200 Elo of the candidate's rating jittered by its uncertainty, weighted by
the variance of the predicted result, so new checkpoints mostly meet their
neighbours. Settled players only play as opponents. Three failed jobs in a
row mark a player broken until re-uploaded.

Imported games (henhen PGN, game id or series, meaf id, line or workshop,
FEN, JSON) carry no play-time stats; a review annotates them with any
analysable engine on request.

## Analysis

`POST /api/eval` resolves a position (a game ply, a FEN, or a board with the
side to move and capture clock), asks the engine's `bot analyse` process and
returns the legal moves, the value, the search lines with principal
variations and the heads: policy, search visit share, action values, draw
mass, the value panel (the conv line's state value head, or sq's
policy-weighted Q, labelled as such, the win, draw and loss probabilities of
a model exported with the outcome head, the root value after search and
plies left) and the plies-to-end distribution. Budgets: quick (network
only), standard (128 simulations), deep (800, admin), or an explicit
simulation count. Responses are cached per position, engine and budget.

## Access

Visitors browse everything and run quick and standard analysis and game
reviews without per-IP rate limits. Imports remain limited to 12 per minute;
the review queue is bounded by the analysis process count. Uploads, retiring,
deletions and deep budgets need the `X-Admin-Token` header
(`ARENA_ADMIN_TOKEN` in `/etc/arena.env`) or a LAN
address; the tunnel's visitors arrive with their own address through
`X-Forwarded-For`.

## Training side

`watch.sh` runs on the training machine beside a run and uploads every tenth
checkpoint; nothing is evaluated there.
