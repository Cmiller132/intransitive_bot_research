import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { useLiveGames } from '../api/live'
import type { Frame, GameRecord, LiveJob, Side } from '../api/types'
import { Board } from '../board/Board'
import type { PieceSetId } from '../board/pieces'
import { SideName } from '../components/SideName'
import { statesFor } from '../rules/replay'
import { formatMove, type Dialect } from '../rules/notation'
import { useArenaStore } from '../state/store'
import { signed } from './gameStats'
import {
    endingText,
    lastMover,
    liveCards,
    liveCounts,
    sideToMove,
    valueFor,
    type LiveCard,
    type LiveEnding,
} from './liveCards'

type Names = { blue: string; red: string }

const PLACEHOLDER: Names = { blue: 'Blue', red: 'Red' }

/** Running games as a grid of small boards, following /api/live/events. */
export function LiveView() {
    const { frame, dialect, pieceSet } = useArenaStore()
    const [jobs, setJobs] = useState<LiveJob[]>()
    const [names, setNames] = useState<Record<string, Names>>({})
    const [endings, setEndings] = useState<Record<string, LiveEnding>>({})
    const [now, setNow] = useState(() => Date.now())
    const [error, setError] = useState('')
    const refreshing = useRef(false)
    const refreshAgain = useRef(false)
    const mounted = useRef(false)

    const refresh = useCallback(function refresh() {
        if (!mounted.current) return
        if (refreshing.current) {
            refreshAgain.current = true
            return
        }
        refreshing.current = true
        api.live()
            .then((value) => {
                if (!mounted.current) return
                setJobs(value.jobs)
                setError('')
                // Games whose job has gone away are dropped, so a long watch stays small.
                const keep = new Set(value.jobs.flatMap((job) => job.games.map((game) => game.game_id)))
                const prune = <T,>(current: Record<string, T>): Record<string, T> => {
                    const stale = Object.keys(current).filter((id) => !keep.has(id))
                    if (!stale.length) return current
                    const next = { ...current }
                    for (const id of stale) delete next[id]
                    return next
                }
                setNames(prune)
                setEndings(prune)
            })
            .catch((cause: Error) => {
                if (mounted.current) setError(cause.message)
            })
            .finally(() => {
                refreshing.current = false
                if (refreshAgain.current) {
                    refreshAgain.current = false
                    refresh()
                }
            })
    }, [])

    useEffect(() => {
        mounted.current = true
        refresh()
        const timer = window.setInterval(refresh, 5_000)
        return () => {
            mounted.current = false
            window.clearInterval(timer)
        }
    }, [refresh])

    // Running games, plus the ones that ended while we watched: a finished card
    // still wants the record for its players and its final position.
    const shownIds = useMemo(() => {
        const live = (jobs || []).flatMap((job) =>
            job.games.filter((game) => game.live).map((game) => game.game_id),
        )
        return [...new Set([...live, ...Object.keys(endings)])]
    }, [jobs, endings])
    const { records: games, errors: gameErrors } = useLiveGames(
        shownIds,
        (event) => {
            if (event.type === 'move') {
                return
            }
            if (event.type === 'game_start') {
                const { blue, red } = event
                if (blue && red) setNames((current) => ({ ...current, [event.game_id]: { blue, red } }))
                refresh()
                return
            }
            if (event.type === 'game_end') {
                const ending: LiveEnding = {
                    winner: event.winner ?? null,
                    reason: event.reason || 'unknown',
                    at: Date.now(),
                }
                setEndings((current) => ({ ...current, [event.game_id]: ending }))
                setNow(Date.now())
                refresh()
                return
            }
            if (event.type !== 'fit') refresh()
        },
        refresh,
    )

    // A finished card expires on its own; the clock only runs while one is up.
    const anyFinished = Object.keys(endings).length > 0
    useEffect(() => {
        if (!anyFinished) return
        const timer = window.setInterval(() => setNow(Date.now()), 3_000)
        return () => window.clearInterval(timer)
    }, [anyFinished])

    const cards = useMemo(() => liveCards(jobs || [], endings, now), [jobs, endings, now])
    const counts = liveCounts(cards)
    const updateError = shownIds.some((id) => gameErrors[id])

    return (
        <section className="live-view">
            {error && (
                <p className="form-error" role="alert">
                    Live status unavailable: {error}
                </p>
            )}
            {updateError && (
                <p className="form-error" role="alert">
                    Some live boards could not refresh. Retrying…
                </p>
            )}
            {!jobs && !error && (
                <div className="loading" role="status">
                    <span className="cycle-loader" aria-hidden="true">
                        RSP
                    </span>{' '}
                    Asking the scheduler…
                </div>
            )}
            {jobs && !cards.length && (
                <div className="empty-state inline-empty">
                    <h2>Nothing is playing right now</h2>
                    <p>
                        The scheduler starts a job as soon as a player's rating interval is wider than the
                        target.
                    </p>
                </div>
            )}
            {cards.length > 0 && (
                <p className="live-summary mono" aria-live="polite">
                    <b>{counts.live} live</b> across {counts.jobs} job{counts.jobs === 1 ? '' : 's'}
                    {counts.finished ? ` · ${counts.finished} just finished` : ''}
                </p>
            )}
            <div className="live-grid">
                {cards.map((card) => (
                    <LiveCardView
                        key={card.game.game_id}
                        card={card}
                        loaded={games[card.game.game_id]}
                        names={names[card.game.game_id] || PLACEHOLDER}
                        frame={frame}
                        dialect={dialect}
                        pieceSet={pieceSet}
                    />
                ))}
            </div>
        </section>
    )
}

const LiveCardView = memo(function LiveCardView({
    card,
    loaded,
    names,
    frame,
    dialect,
    pieceSet,
}: {
    card: LiveCard
    loaded?: GameRecord
    names: Names
    frame: Frame
    dialect: Dialect
    pieceSet: PieceSetId
}) {
    const { game, job, ending } = card
    const moves = loaded?.moves
    const setup = loaded?.setup || null
    const first = loaded?.meta.start_to_move === 'red' ? 'red' : 'blue'
    const replay = useMemo(() => statesFor(moves || [], setup, first), [moves, setup, first])
    const position = loaded ? replay.states.at(-1) : undefined
    names = loaded ? { blue: loaded.players.blue.name, red: loaded.players.red.name } : names
    const plies = moves ? moves.length : game.ply_count
    const last = moves?.at(-1)
    const mover = lastMover(plies)
    const turn: Side | null = ending ? null : sideToMove(plies)
    const value = mover ? valueFor(last?.stats, mover) : null
    const notation =
        last && replay.states.length > 1
            ? formatMove(
                  last,
                  dialect,
                  replay.states[replay.states.length - 2].board,
                  last.capture,
                  false,
                  frame,
              )
            : '—'
    // The scheduler pairs a rating candidate with a chosen opponent and reports
    // them in that order, so `job.a` is this job's candidate.
    const tagFor = (name: string) => (name === job.a ? 'candidate' : name === job.b ? 'opponent' : undefined)

    return (
        <Link
            className={`live-card${ending ? ' is-final' : ''}`}
            to={`/game/${encodeURIComponent(game.game_id)}`}
            aria-label={`${names.blue} as Blue against ${names.red} as Red, ply ${plies}${
                ending ? `, ${endingText(ending, names.blue, names.red)}` : ''
            }`}
        >
            <div className="live-card-head mono">
                <span>
                    job {String(job.id)} · pair {game.pair} · game {game.game}
                </span>
                <span className={`live-badge${ending ? ' final' : ''}`}>{ending ? 'final' : 'live'}</span>
            </div>
            <div className="live-card-board" aria-hidden="true">
                {position ? (
                    <Board
                        board={position.board}
                        frame={frame}
                        pieceSet={pieceSet}
                        lastMove={last}
                        overlays={[]}
                        labels={false}
                        orientationLabel={`${frame} frame`}
                        homeTint
                    />
                ) : (
                    <div className="live-card-skeleton" />
                )}
            </div>
            <div className="live-card-players">
                <SideName side="blue" name={names.blue} turn={turn === 'blue'} tag={tagFor(names.blue)} />
                <SideName side="red" name={names.red} turn={turn === 'red'} tag={tagFor(names.red)} />
            </div>
            <div className="live-card-meta mono">
                {ending ? (
                    <span className="live-card-result">{endingText(ending, names.blue, names.red)}</span>
                ) : (
                    <>
                        <span>ply {plies}</span>
                        <span>{notation}</span>
                        <span
                            className={mover ? `side-${mover}` : undefined}
                            title={
                                mover
                                    ? `${names[mover]} scored this position ${
                                          value === null ? 'without a value' : signed(value)
                                      } for itself`
                                    : 'no search value yet'
                            }
                        >
                            {value === null ? '—' : signed(value)}
                            {last?.stats?.exact_win ? ' win' : ''}
                        </span>
                    </>
                )}
            </div>
        </Link>
    )
})
