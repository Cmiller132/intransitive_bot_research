import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { RatedPlayer, RatingsHistory, RatingsResponse } from '../api/types'
import { LineChart, type ChartSeries } from '../components/LineChart'
import {
    NO_RECORD,
    checkpointSeries,
    formatHeadToHead,
    formatRating,
    sortByRating,
    summariseCounts,
} from './runGrouping'

const wdl = (player: RatedPlayer) => `${player.wins} / ${player.draws} / ${player.losses}`
const score = (player: RatedPlayer) =>
    player.games ? `${Math.round(((player.wins + player.draws / 2) / player.games) * 100)}%` : '—'

const HALF_WIDTH_HINT =
    "95% interval from the player's own games against its opponents' fitted ratings (KataGo-style). The anchor's row shows how well the pool is pinned to it."

function status(player: RatedPlayer): { label: string; tone: string } {
    if (player.retired) return { label: 'retired', tone: 'muted' }
    if (player.broken) return { label: 'broken', tone: 'bad' }
    if (player.settled) return { label: 'settled', tone: 'good' }
    return { label: 'measuring', tone: '' }
}

export function RankingsPage() {
    const [data, setData] = useState<RatingsResponse>()
    const [history, setHistory] = useState<{ data?: RatingsHistory; error?: string }>()
    const [selected, setSelected] = useState<string>()
    const [showRetired, setShowRetired] = useState(false)
    const [needle, setNeedle] = useState('')
    const [error, setError] = useState('')

    useEffect(() => {
        let cancelled = false
        api.ratings()
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

    useEffect(() => {
        if (!selected) return
        let cancelled = false
        api.ratingsHistory(selected)
            .then((value) => {
                if (!cancelled) setHistory({ data: value })
            })
            .catch((cause: Error) => {
                if (!cancelled) setHistory({ error: cause.message })
            })
        return () => {
            cancelled = true
        }
    }, [selected])

    const players = useMemo(() => {
        const all = data?.players || []
        const visible = all.filter((player) => (showRetired || !player.retired) && matches(player, needle))
        return sortByRating(visible, (player) => player.name)
    }, [data, showRetired, needle])

    const summary = useMemo(() => {
        if (!data) return ''
        const retired = (player: RatedPlayer) => player.retired
        const hiddenRetired = data.players.filter(retired).length - players.filter(retired).length
        const counts = summariseCounts(players.length, data.players.length, hiddenRetired)
        return `${counts}${data.fitted ? ` · fitted ${data.fitted}` : ''}`
    }, [data, players])

    const checkpoints = useMemo(
        () => checkpointSeries((data?.players || []).filter((player) => showRetired || !player.retired)),
        [data, showRetired],
    )
    const checkpointChart: ChartSeries[] = checkpoints.runs.map((run) => ({
        label: run.run,
        values: run.values,
    }))
    const historyPoints = selected ? history?.data?.players[selected] || [] : []
    const historyError = history?.error || ''

    return (
        <main className="page">
            <div className="page-heading">
                <div>
                    <span className="eyebrow">Bradley-Terry pool</span>
                    <h1>Rankings</h1>
                </div>
                <span className="record-count mono" aria-live="polite">
                    {data ? summary : 'Loading…'}
                </span>
            </div>
            <p className="lede">
                Every model in the arena plays rated games against the rest. A player is settled once it has
                at least eight games and a 95 % interval no wider than the target
                {data ? ` (± ${Math.round(data.target)})` : ''}
                {data?.anchor ? `; ${data.anchor} is the anchor fixed at 0.` : '.'}
            </p>
            <div className="double-rule" />

            {error && (
                <p className="form-error" role="alert">
                    Could not load the ratings: {error}
                </p>
            )}
            {!data && !error && (
                <div className="loading" role="status">
                    <span className="cycle-loader" aria-hidden="true">
                        RSP
                    </span>{' '}
                    Fitting table…
                </div>
            )}

            {data && (
                <>
                    <div className="filters">
                        <label>
                            Search
                            <input
                                value={needle}
                                placeholder="Name, run or spec"
                                onChange={(event) => setNeedle(event.target.value)}
                            />
                        </label>
                        <label className="checkbox">
                            <input
                                type="checkbox"
                                checked={showRetired}
                                onChange={(event) => setShowRetired(event.target.checked)}
                            />
                            Show retired
                        </label>
                    </div>

                    <div className="engine-table-wrap">
                        <table className="engine-table compact-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Player</th>
                                    <th title={HALF_WIDTH_HINT}>Rating ± 95 %</th>
                                    <th>Games</th>
                                    <th>W / D / L</th>
                                    <th>Score</th>
                                    <th
                                        title={`Head-to-head against the anchor${data.anchor ? ` (${data.anchor})` : ''}`}
                                    >
                                        vs anchor
                                    </th>
                                    <th title="Head-to-head against the previous checkpoint of the same run">
                                        vs previous
                                    </th>
                                    <th>Status</th>
                                    <th>Kind</th>
                                    <th>Added</th>
                                </tr>
                            </thead>
                            <tbody>
                                {players.map((player, index) => {
                                    const state = status(player)
                                    const isAnchor = !!data.anchor && player.name === data.anchor
                                    return (
                                        <tr
                                            key={player.name}
                                            className={selected === player.name ? 'selected-row' : ''}
                                            onClick={() => setSelected(player.name)}
                                        >
                                            <td data-label="#" className="mono">
                                                {index + 1}
                                            </td>
                                            <td data-label="Player">
                                                <b>{player.name}</b>
                                                <small className="mono">{player.spec}</small>
                                            </td>
                                            <td
                                                data-label="Rating ± 95 %"
                                                className="mono"
                                                title={HALF_WIDTH_HINT}
                                            >
                                                {formatRating(player.rating, player.half_width)}
                                            </td>
                                            <td data-label="Games" className="mono">
                                                {player.games}
                                            </td>
                                            <td data-label="W / D / L" className="mono">
                                                {wdl(player)}
                                            </td>
                                            <td data-label="Score" className="mono">
                                                {score(player)}
                                            </td>
                                            <td
                                                data-label="vs anchor"
                                                className="mono"
                                                title={
                                                    isAnchor
                                                        ? `${player.name} is the anchor`
                                                        : data.anchor || undefined
                                                }
                                            >
                                                {isAnchor ? NO_RECORD : formatHeadToHead(player.vs_anchor)}
                                            </td>
                                            <td
                                                data-label="vs previous"
                                                className="mono"
                                                title={player.parent || undefined}
                                            >
                                                {formatHeadToHead(player.vs_parent)}
                                            </td>
                                            <td data-label="Status">
                                                <span className={`state-badge ${state.tone}`}>
                                                    {state.label}
                                                </span>
                                            </td>
                                            <td data-label="Kind">
                                                <span className="kind-badge">{player.kind}</span>
                                            </td>
                                            <td data-label="Added" className="mono">
                                                {player.added ? player.added.slice(0, 10) : '—'}
                                            </td>
                                        </tr>
                                    )
                                })}
                            </tbody>
                        </table>
                        {!players.length && <p className="empty-copy">No player matches these filters.</p>}
                    </div>

                    <div className="chart-grid">
                        <section className="panel chart-panel">
                            <div className="panel-head">
                                <b>Rating by checkpoint</b>
                                <span>{checkpointChart.length} runs</span>
                            </div>
                            {checkpointChart.length ? (
                                <LineChart
                                    x={checkpoints.x}
                                    series={checkpointChart}
                                    xLabel="checkpoint"
                                    yLabel="rating"
                                />
                            ) : (
                                <p className="empty-copy compact">
                                    No run has two rated checkpoints yet. Names like{' '}
                                    <span className="mono">conv_g128_240</span> are plotted against their
                                    number.
                                </p>
                            )}
                        </section>
                        <section className="panel chart-panel">
                            <div className="panel-head">
                                <b>Rating history</b>
                                <span>{selected || 'select a player'}</span>
                            </div>
                            {historyError && (
                                <p className="form-error" role="alert">
                                    {historyError}
                                </p>
                            )}
                            {selected && historyPoints.length > 1 && (
                                <LineChart
                                    x={historyPoints.map((_, index) => index)}
                                    series={[
                                        {
                                            label: selected,
                                            values: historyPoints.map((point) => point.rating),
                                        },
                                    ]}
                                    xLabel="fit"
                                    yLabel="rating"
                                />
                            )}
                            {selected && historyPoints.length <= 1 && !historyError && (
                                <p className="empty-copy compact">
                                    {historyPoints.length
                                        ? 'Only one fit so far for'
                                        : 'No fit recorded yet for'}{' '}
                                    {selected}.
                                </p>
                            )}
                            {!selected && (
                                <p className="empty-copy compact">
                                    Pick a row above to follow how its rating moved.
                                </p>
                            )}
                            {selected && (
                                <div className="form-actions">
                                    <Link
                                        className="button-link"
                                        to={`/insights/${encodeURIComponent(selected)}`}
                                    >
                                        Insights
                                    </Link>
                                    <Link
                                        className="button-link"
                                        to={`/games?player=${encodeURIComponent(selected)}`}
                                    >
                                        Games
                                    </Link>
                                    <Link className="button-link" to="/matchups">
                                        Matchups
                                    </Link>
                                </div>
                            )}
                        </section>
                    </div>
                </>
            )}
        </main>
    )
}

function matches(player: RatedPlayer, needle: string): boolean {
    const text = needle.trim().toLowerCase()
    if (!text) return true
    return `${player.name} ${player.spec} ${player.kind}`.toLowerCase().includes(text)
}
