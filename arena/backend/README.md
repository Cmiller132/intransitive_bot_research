# arena/backend

The `arena` Python package: the FastAPI service behind the site and the
rating runner, one process (`uvicorn arena.api.app:app`) on the container's
loopback, in front of which Caddy serves the built frontend and proxies
`/api` and `/healthz`.

## Modules

- `settings.py`: `Settings`, from the `ARENA_*` environment (`/etc/arena.env`
  on the container): `ROOT` (runtime directory: `bot`, `players/`,
  `arena.db`), `ADMIN_TOKEN`, `PUBLIC`, `JOBS` (parallel `bot eval`
  processes, 4; 6 on the container), `TARGET` (settle half-width, 30 Elo),
  `MAX_GAMES` (a player's game budget, 1000), `SIMS` (32), `PAIRS` (openings
  per job, 2),
  `ANCHOR` (`sq_g128`), `ANALYSIS_PROCS` (2), `BOT` (the bot command when not
  `ROOT/bot`), `SCHEDULER` (off to serve without playing).
- `core/`: `frames` (absolute squares and the site's display frames), `game`
  (records and replay under the engine's rules through `bot`), `notation`,
  `pgn`, `meaf`, `symmetry`, `export`, `fetch` (the site's game and series
  downloads).
- `players.py`: the `players/<name>/player.json` pool, uploads and specs.
- `scheduler.py`: the job loop, `bot eval --stream` parsing into live games,
  the KataGo-style pairer, failure streaks; `ratings.py`: the Bayesian Elo
  fit and its history.
- `engines.py`: the pool of `bot analyse` processes and the analysis
  response; `analysis/`: reviews of imported games, the opening explorer,
  player insights.
- `events.py`: the pub/sub behind the `/api/live/events` stream; overflowing
  live queues emit `resync` so clients reload authoritative records.
  Review streams send the current job status on every connection.
- `db/`: `schema.sql` and `store.py`, sqlite in WAL mode.
- `api/`: the app, its dependencies (admin by token or LAN address) and one
  router per resource: players (`admin.py`), games, live, ratings, positions
  and eval, analysis, engines, export, jobs, meta.
- `fake_bot.py`: a stand-in for the binary that the tests run against.

## Running

```
python -m pip install -e ".[dev]"
python -m ruff check . && python -m ruff format --check . && python -m pytest -q
ARENA_ROOT=/opt/arena uvicorn arena.api.app:app --host 127.0.0.1 --port 8100
```

`cargo xtask check` runs the lints and tests. The container's unit is
`../arena.service`, its Caddy configuration `../Caddyfile`.
