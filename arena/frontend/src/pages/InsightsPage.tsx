import { useCallback, useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { InsightsResponse } from '../api/types'
import { formatRating } from './runGrouping'

export function InsightsPage() {
    const { player: routePlayer } = useParams()
    return <InsightsLoader key={routePlayer || ''} routePlayer={routePlayer} />
}

function InsightsLoader({ routePlayer }: { routePlayer?: string }) {
    const [player, setPlayer] = useState(routePlayer || '')
    const [data, setData] = useState<InsightsResponse>()
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(true)

    const load = useCallback((name: string) => {
        const trimmed = name.trim()
        if (!trimmed) {
            setError('Enter a player or engine name.')
            setLoading(false)
            return Promise.resolve()
        }
        setLoading(true)
        setError('')
        return api
            .insights(trimmed)
            .then((response) => {
                setData(response)
                setLoading(false)
            })
            .catch((cause: Error) => {
                setData(undefined)
                setError(cause.message)
                setLoading(false)
            })
    }, [])

    useEffect(() => {
        let cancelled = false
        // With no player in the URL, open on the strongest rated player.
        const run = async () => {
            try {
                let name = routePlayer
                if (!name) {
                    const ratings = await api.ratings()
                    name = (ratings.players.find((item) => item.rating !== null) || ratings.players[0])?.name
                    if (!cancelled && name) setPlayer(name)
                }
                if (!name) {
                    if (!cancelled) setLoading(false)
                    return
                }
                const response = await api.insights(name)
                if (!cancelled) {
                    setData(response)
                    setLoading(false)
                }
            } catch (cause) {
                if (!cancelled) {
                    setError(cause instanceof Error ? cause.message : String(cause))
                    setLoading(false)
                }
            }
        }
        void run()
        return () => {
            cancelled = true
        }
    }, [routePlayer])

    const submit = (event: FormEvent) => {
        event.preventDefault()
        void load(player)
    }

    return (
        <main className="page">
            <span className="eyebrow">Arena performance</span>
            <h1>Player insights</h1>
            <p className="lede">
                Results, opponents, endings, and review quality across every recorded game.
            </p>
            <div className="double-rule" />
            <form className="explorer-form" onSubmit={submit}>
                <label>
                    Player or engine
                    <input
                        value={player}
                        onChange={(event) => setPlayer(event.target.value)}
                        placeholder="conv_g128_240"
                    />
                </label>
                <button type="submit" disabled={loading}>
                    {loading ? 'Loading…' : 'Load insights'}
                </button>
            </form>
            {error && (
                <p className="form-error" role="alert">
                    Could not load insights: {error}
                </p>
            )}
            {loading && !data && (
                <div className="loading">
                    <span className="cycle-loader">RSP</span> Loading results…
                </div>
            )}
            {data && !data.games && !loading && (
                <div className="empty-state inline-empty">
                    <h2>No matching games</h2>
                    <p>
                        No recorded game contains “{data.player}”. Try the exact player or engine name shown
                        in the rankings.
                    </p>
                </div>
            )}
            {data && data.games > 0 && (
                <>
                    <section className="insight-stats">
                        <article>
                            <b>{formatRating(data.rating, data.half_width)}</b>
                            <span>rating</span>
                        </article>
                        <article>
                            <b>{data.games}</b>
                            <span>games</span>
                        </article>
                        <article>
                            <b>{Math.round(data.score * 100)}%</b>
                            <span>score</span>
                        </article>
                        <article>
                            <b>
                                {data.wins} / {data.draws} / {data.losses}
                            </b>
                            <span>win · draw · loss</span>
                        </article>
                    </section>
                    <div className="insight-grid">
                        <section className="panel">
                            <div className="panel-head">
                                <b>End reasons</b>
                            </div>
                            {Object.entries(data.by_reason).map(([name, count]) => (
                                <div className="metric-row" key={name}>
                                    <span>{name.replace('_', ' ')}</span>
                                    <b>{count}</b>
                                </div>
                            ))}
                        </section>
                        <section className="panel">
                            <div className="panel-head">
                                <b>Opponents</b>
                                <span>W / D / L · score</span>
                            </div>
                            {data.opponents.slice(0, 12).map((item) => (
                                <div className="metric-row" key={item.name}>
                                    <Link
                                        to={`/matchups/${encodeURIComponent(data.player)}/${encodeURIComponent(item.name)}`}
                                    >
                                        {item.name}
                                    </Link>
                                    <b className="mono">
                                        {item.wins ?? 0} / {item.draws ?? 0} / {item.losses ?? 0}
                                        {item.score === undefined
                                            ? ''
                                            : ` · ${Math.round(item.score * 100)}%`}
                                    </b>
                                </div>
                            ))}
                            {!data.opponents.length && <p className="empty-copy compact">No opponent yet.</p>}
                        </section>
                        <section className="panel">
                            <div className="panel-head">
                                <b>Coverage</b>
                            </div>
                            {Object.entries(data.by_source).map(([name, count]) => (
                                <div className="metric-row" key={name}>
                                    <span>{name}</span>
                                    <b>{count}</b>
                                </div>
                            ))}
                            {Object.entries(data.by_color).map(([name, count]) => (
                                <div className="metric-row" key={name}>
                                    <span>{name} games</span>
                                    <b>{count}</b>
                                </div>
                            ))}
                            <div className="metric-row">
                                <span>Review accuracy</span>
                                <b>{data.accuracy == null ? '—' : `${data.accuracy.toFixed(1)}%`}</b>
                            </div>
                            <div className="metric-row">
                                <span>Engine agreement</span>
                                <b>{data.agreement == null ? '—' : `${Math.round(data.agreement * 100)}%`}</b>
                            </div>
                        </section>
                    </div>
                    <section className="panel recent-games">
                        <div className="panel-head">
                            <b>Recent games</b>
                            <span>{Math.min(10, data.timeline.length)} shown</span>
                        </div>
                        {data.timeline
                            .slice(-10)
                            .reverse()
                            .map((item) => (
                                <Link
                                    className="timeline-row"
                                    to={`/game/${encodeURIComponent(item.game_id)}`}
                                    key={`${item.game_id}-${item.outcome}`}
                                >
                                    <span>
                                        {item.date
                                            ? new Date(item.date).toLocaleString()
                                            : 'Date unavailable'}
                                    </span>
                                    <b className={`outcome ${item.outcome}`}>{item.outcome}</b>
                                    <span>{item.rating == null ? 'Unrated' : Math.round(item.rating)}</span>
                                </Link>
                            ))}
                    </section>
                </>
            )}
        </main>
    )
}
