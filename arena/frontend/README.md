# arena/frontend

Vite + React 18 + TypeScript (strict) single-page app for the arena: rankings,
games (library and live), matchups, the analysis workspace, the opening
explorer, the engine pool and local settings.

```
npm ci
npm run dev        # proxies /api and /healthz to VITE_API_PROXY (default http://127.0.0.1:8100)
npm run typecheck
npm run lint
npm test
npm run build      # dist/ is shipped to /opt/arena/web
```

## Conventions

- No UI kit and no dependency beyond `react`, `react-router-dom`, `zustand` and
  `uplot`. Styling is hand-written CSS over the tokens in `src/theme/tokens.css`.
- Squares are absolute everywhere (a1 = 0, i9 = 80). Only `src/rules/frames.ts`
  converts to a display frame; every board layer receives absolute data and
  calls `toFrame` itself.
- `src/api/types.ts` mirrors the backend contract; `src/api/client.ts` is the
  only place that talks to `/api`, and `src/api/live.ts` holds the live
  hooks (scheduler polling, the `/api/live/events` stream and watched games).
  `liveGames.ts` keeps game snapshots and streamed plies contiguous in both
  game views; gaps, reconnects, foregrounding and five-second polling
  resynchronise live records.
- Analysis and position exports send the original setup and its continuation,
  or a stored game ID and ply. A variation shares its displayed position.
- Pure logic lives beside its page and is unit-tested with vitest
  (`runGrouping`, `gameStats`, `matchupCells`, `analysisHeads`, `analysisLine`,
  `libraryState`, `rules/*`).
- Local preferences (theme, frame, notation, piece set, admin token, overlay
  labels) live in `src/state/store.ts` under `arena-*` localStorage keys.
