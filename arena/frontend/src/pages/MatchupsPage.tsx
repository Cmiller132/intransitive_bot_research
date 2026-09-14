import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { MatchupsResponse } from '../api/types'
import { cellIndex, cellKey, cellScoreLabel, matchupTone } from './matchupCells'

const short = (name: string) => (name.length > 14 ? `${name.slice(0, 13)}…` : name)

export function MatchupsPage() {
    const navigate = useNavigate()
    const [data, setData] = useState<MatchupsResponse>()
    const [error, setError] = useState('')
    const [needle, setNeedle] = useState('')

    useEffect(() => {
        let cancelled = false
        api.matchups()
            .then((value) => {
                if (!cancelled) setData(value)
            })
            .catch((cause: Error) => {
                if (!cancelled) setError(cause.message)
            })
        return () => {
            cancelled = true
        }
    }, [])

    const cells = useMemo(() => cellIndex(data?.cells || []), [data])
    const players = useMemo(() => {
        const text = needle.trim().toLowerCase()
        return (data?.players || []).filter((name) => !text || name.toLowerCase().includes(text))
    }, [data, needle])

    return (
        <main className="page">
            <div className="page-heading">
                <div>
                    <span className="eyebrow">Head to head</span>
                    <h1>Matchups</h1>
                </div>
                <span className="record-count mono" aria-live="polite">
                    {data ? `${players.length} players` : 'Loading…'}
                </span>
            </div>
            <p className="lede">
                Each cell is the row player's score against the column player. Green means the pair went
                better than the rating model expected, red worse. Pick a cell for the pair's games.
            </p>
            <div className="double-rule" />

            {error && (
                <p className="form-error" role="alert">
                    Could not load the matrix: {error}
                </p>
            )}
            {!data && !error && (
                <div className="loading" role="status">
                    <span className="cycle-loader" aria-hidden="true">
                        RSP
                    </span>{' '}
                    Counting pairs…
                </div>
            )}

            {data && (
                <>
                    <div className="filters">
                        <label>
                            Filter players
                            <input
                                value={needle}
                                placeholder="Run or name"
                                onChange={(event) => setNeedle(event.target.value)}
                            />
                        </label>
                    </div>
                    <div className="matrix-wrap">
                        <table className="matrix">
                            <thead>
                                <tr>
                                    <th className="corner" scope="col">
                                        row vs column
                                    </th>
                                    {players.map((name) => (
                                        <th key={name} scope="col" title={name}>
                                            <span>{short(name)}</span>
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {players.map((a) => (
                                    <tr key={a}>
                                        <th scope="row" title={a}>
                                            {short(a)}
                                        </th>
                                        {players.map((b) => {
                                            if (a === b)
                                                return (
                                                    <td className="self" key={b}>
                                                        <span aria-hidden="true">·</span>
                                                    </td>
                                                )
                                            const cell = cells.get(cellKey(a, b))
                                            const tone = matchupTone(
                                                cell || { games: 0, score_a: null, expected_a: null },
                                            )
                                            const label = cell ? cellScoreLabel(cell) : '–'
                                            return (
                                                <td
                                                    key={b}
                                                    className={`cell ${tone.level}`}
                                                    style={{ background: tone.background }}
                                                >
                                                    <button
                                                        type="button"
                                                        disabled={!cell?.games}
                                                        title={
                                                            cell?.games
                                                                ? `${a} ${cell.wins_a}–${cell.draws}–${cell.wins_b} ${b}${
                                                                      cell.expected_a === null ||
                                                                      cell.expected_a === undefined
                                                                          ? ''
                                                                          : ` · expected ${Math.round(cell.expected_a * 100)}%`
                                                                  }`
                                                                : `${a} has not met ${b}`
                                                        }
                                                        onClick={() =>
                                                            navigate(
                                                                `/matchups/${encodeURIComponent(a)}/${encodeURIComponent(b)}`,
                                                            )
                                                        }
                                                    >
                                                        <b>{label}</b>
                                                        <em>{cell?.games || 0}</em>
                                                    </button>
                                                </td>
                                            )
                                        })}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    {!players.length && <p className="empty-copy">No player matches that filter.</p>}
                    <div className="matrix-legend">
                        <span className="legend-swatch ahead" /> above expectation
                        <span className="legend-swatch behind" /> below expectation
                        <span>· cell shows the row player's score, small number the games played</span>
                    </div>
                </>
            )}
        </main>
    )
}
