import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { MatchupPair } from '../api/types'
import { GameList } from '../components/GameList'

export function MatchupPairPage() {
    const { a = '', b = '' } = useParams()
    return <PairLoader key={`${a}/${b}`} a={a} b={b} />
}

function PairLoader({ a, b }: { a: string; b: string }) {
    const [data, setData] = useState<MatchupPair>()
    const [error, setError] = useState('')

    useEffect(() => {
        let cancelled = false
        api.matchup(a, b)
            .then((value) => {
                if (!cancelled) setData(value)
            })
            .catch((cause: Error) => {
                if (!cancelled) setError(cause.message)
            })
        return () => {
            cancelled = true
        }
    }, [a, b])

    const games = data ? data.wins_a + data.draws + data.wins_b : 0
    const score = games ? (data!.wins_a + data!.draws / 2) / games : null

    return (
        <main className="page">
            <div className="page-heading">
                <div>
                    <span className="eyebrow">Pair</span>
                    <h1>
                        {a} <span className="versus">vs</span> {b}
                    </h1>
                </div>
                <Link className="button-link" to="/matchups">
                    ← All matchups
                </Link>
            </div>
            <div className="double-rule" />

            {error && (
                <p className="form-error" role="alert">
                    Could not load this pair: {error}
                </p>
            )}
            {!data && !error && (
                <div className="loading" role="status">
                    <span className="cycle-loader" aria-hidden="true">
                        RSP
                    </span>{' '}
                    Loading pair…
                </div>
            )}

            {data && (
                <>
                    <section className="insight-stats">
                        <article>
                            <b>{games}</b>
                            <span>games</span>
                        </article>
                        <article>
                            <b>
                                {data.wins_a} / {data.draws} / {data.wins_b}
                            </b>
                            <span>
                                {a} win · draw · {b} win
                            </span>
                        </article>
                        <article>
                            <b>{score === null ? '—' : `${Math.round(score * 100)}%`}</b>
                            <span>score for {a}</span>
                        </article>
                        <article>
                            <b>{data.avg_plies == null ? '—' : Math.round(data.avg_plies)}</b>
                            <span>average plies</span>
                        </article>
                    </section>

                    <div className="insight-grid">
                        <section className="panel">
                            <div className="panel-head">
                                <b>By colour</b>
                            </div>
                            {Object.entries(data.by_colour).map(([colour, split]) => (
                                <div className="metric-row" key={colour}>
                                    <span>{colour.replace('_', ' ')}</span>
                                    <b className="mono">
                                        {split.wins_a} / {split.draws} / {split.wins_b}
                                    </b>
                                </div>
                            ))}
                            {!Object.keys(data.by_colour).length && (
                                <p className="empty-copy compact">No colour split recorded.</p>
                            )}
                        </section>
                        <section className="panel">
                            <div className="panel-head">
                                <b>End reasons</b>
                            </div>
                            {Object.entries(data.end_reasons).map(([reason, count]) => (
                                <div className="metric-row" key={reason}>
                                    <span>{reason.replace('_', ' ')}</span>
                                    <b className="mono">{count}</b>
                                </div>
                            ))}
                            {!Object.keys(data.end_reasons).length && (
                                <p className="empty-copy compact">No finished game yet.</p>
                            )}
                        </section>
                        <section className="panel">
                            <div className="panel-head">
                                <b>Elsewhere</b>
                            </div>
                            <div className="metric-row">
                                <span>{a}</span>
                                <Link to={`/insights/${encodeURIComponent(a)}`}>insights</Link>
                            </div>
                            <div className="metric-row">
                                <span>{b}</span>
                                <Link to={`/insights/${encodeURIComponent(b)}`}>insights</Link>
                            </div>
                            <div className="metric-row">
                                <span>Analysis</span>
                                <Link
                                    to={`/analysis?engine=${encodeURIComponent(a)}&engine2=${encodeURIComponent(b)}`}
                                >
                                    compare both engines
                                </Link>
                            </div>
                        </section>
                    </div>

                    <section className="wide-section">
                        <div className="double-rule" />
                        <h2>Games</h2>
                        <GameList games={data.games} empty="No game between these two yet." />
                    </section>
                </>
            )}
        </main>
    )
}
