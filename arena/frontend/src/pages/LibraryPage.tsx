import { useDeferredValue, useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import { useSchedulerStatus } from '../api/live'
import type { GameSummary } from '../api/types'
import { GameList } from '../components/GameList'
import { ImportForm } from '../components/ImportForm'
import { LiveView } from './LiveView'
import { LIBRARY_PAGE_SIZE, libraryPageCount, libraryRange } from './libraryState'

export function LibraryPage() {
    const [search, setSearch] = useSearchParams()
    const live = search.get('live') === '1'
    const player = search.get('player') || ''
    const [games, setGames] = useState<GameSummary[]>([])
    const [total, setTotal] = useState(0)
    const [q, setQ] = useState('')
    const deferredQuery = useDeferredValue(q)
    const [source, setSource] = useState('')
    const [result, setResult] = useState('')
    const [page, setPage] = useState(0)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')
    const scheduler = useSchedulerStatus()

    useEffect(() => {
        if (live) return
        let cancelled = false
        api.games({
            q: deferredQuery.trim(),
            source,
            result,
            player,
            limit: LIBRARY_PAGE_SIZE,
            offset: page * LIBRARY_PAGE_SIZE,
        })
            .then((response) => {
                if (cancelled) return
                setGames(response.items)
                setTotal(response.total)
                setError('')
                setLoading(false)
            })
            .catch((cause: Error) => {
                if (cancelled) return
                setGames([])
                setTotal(0)
                setError(cause.message)
                setLoading(false)
            })
        return () => {
            cancelled = true
        }
    }, [deferredQuery, source, result, player, page, live])

    const pages = libraryPageCount(total)
    const updateFilter = (setter: (value: string) => void, value: string) => {
        setLoading(true)
        setError('')
        setter(value)
        setPage(0)
    }
    const changePage = (next: number) => {
        setLoading(true)
        setError('')
        setPage(next)
    }
    const setLive = (value: boolean) => {
        const next = new URLSearchParams(search)
        if (value) next.set('live', '1')
        else next.delete('live')
        setSearch(next, { replace: true })
    }
    const clearPlayer = () => {
        setPage(0)
        setLoading(true)
        setError('')
        const next = new URLSearchParams(search)
        next.delete('player')
        setSearch(next, { replace: true })
    }

    return (
        <main className="page">
            <div className="page-heading">
                <div>
                    <span className="eyebrow">{live ? 'Running now' : 'Arena and imports'}</span>
                    <h1>Games</h1>
                </div>
                <span className="record-count mono" aria-live="polite">
                    {live
                        ? `${scheduler?.jobs_running ?? 0} of ${scheduler?.slots ?? 0} slots busy`
                        : loading
                          ? 'Loading…'
                          : libraryRange(page, games.length, total)}
                </span>
            </div>
            <div className="tab-row" role="tablist" aria-label="Game view">
                <button
                    role="tab"
                    aria-selected={!live}
                    className={live ? '' : 'on'}
                    onClick={() => setLive(false)}
                >
                    Library
                </button>
                <button
                    role="tab"
                    aria-selected={live}
                    className={live ? 'on' : ''}
                    onClick={() => setLive(true)}
                >
                    Live
                    {scheduler?.jobs_running ? (
                        <em className="live-badge">{scheduler.jobs_running}</em>
                    ) : null}
                </button>
            </div>
            <div className="double-rule" />
            {live ? (
                <LiveView />
            ) : (
                <div className="library-layout">
                    <section>
                        <div className="filters">
                            <label>
                                Search
                                <input
                                    value={q}
                                    onChange={(event) => updateFilter(setQ, event.target.value)}
                                    placeholder="Player or game ID"
                                />
                            </label>
                            <label>
                                Source
                                <select
                                    value={source}
                                    onChange={(event) => updateFilter(setSource, event.target.value)}
                                >
                                    <option value="">All sources</option>
                                    <option value="arena">Arena</option>
                                    <option value="henhen">henhen</option>
                                    <option value="meaf">meaf</option>
                                    <option value="local">local</option>
                                </select>
                            </label>
                            <label>
                                Result
                                <select
                                    value={result}
                                    onChange={(event) => updateFilter(setResult, event.target.value)}
                                >
                                    <option value="">Any result</option>
                                    <option value="blue">Blue wins</option>
                                    <option value="red">Red wins</option>
                                    <option value="draw">Draws</option>
                                </select>
                            </label>
                            {player && (
                                <button type="button" className="filter-chip" onClick={clearPlayer}>
                                    player: {player} ✕
                                </button>
                            )}
                        </div>
                        {error && (
                            <p className="form-error" role="alert">
                                Could not load the library: {error}
                            </p>
                        )}
                        <GameList games={games} loading={loading} />
                        <nav className="pagination" aria-label="Game pages">
                            <button disabled={loading || page === 0} onClick={() => changePage(page - 1)}>
                                ← Newer
                            </button>
                            <span>
                                Page {page + 1} of {pages}
                            </span>
                            <button
                                disabled={loading || page + 1 >= pages}
                                onClick={() => changePage(page + 1)}
                            >
                                Older →
                            </button>
                        </nav>
                    </section>
                    <aside>
                        <h2>Import</h2>
                        <p className="section-intro">
                            Add a PGN, game ID, line or position from henhen or meaf.
                        </p>
                        <ImportForm />
                        <section className="panel">
                            <div className="panel-head">
                                <b>Arena</b>
                                <span>{scheduler ? `${scheduler.jobs_running} running` : 'offline'}</span>
                            </div>
                            <div className="metric-row">
                                <span>Target half-width</span>
                                <b className="mono">
                                    {scheduler ? `± ${Math.round(scheduler.target)}` : '—'}
                                </b>
                            </div>
                            <div className="metric-row">
                                <span>Next candidate</span>
                                <b className="mono">{scheduler?.next_candidate || '—'}</b>
                            </div>
                            <div className="metric-row">
                                <span>Live games</span>
                                <Link to="/games?live=1">watch</Link>
                            </div>
                        </section>
                    </aside>
                </div>
            )}
        </main>
    )
}
